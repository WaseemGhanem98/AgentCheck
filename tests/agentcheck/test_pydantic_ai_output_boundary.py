"""Executable output declarations must be refused before reconstruction."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from functools import partial
from typing import Annotated, Any, Literal, NewType, TypedDict, Union

import pytest
from pydantic import (
    AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer,
    PlainValidator, WrapValidator, computed_field, field_serializer,
    WithJsonSchema, field_validator, model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_ai import Agent, NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.output import StructuredDict
from typing_extensions import TypeAliasType

from agentcheck.adapters import PydanticAIAdapter, UnsupportedTargetError


def _output_form(name: str, callback: Any) -> Any:
    if name == "plain":
        return callback
    if name == "partial":
        return partial(callback)
    if name == "tool":
        return ToolOutput(callback)
    if name == "text":
        return TextOutput(callback)
    if name == "native":
        return NativeOutput(callback)
    if name == "prompted":
        return PromptedOutput(callback)
    if name == "list":
        return [str, callback]
    if name == "tuple":
        return (str, callback)
    if name == "nested":
        return [str, [ToolOutput(callback)]]
    if name == "native-alternatives":
        return NativeOutput([int, callback])
    if name == "prompted-alternatives":
        return PromptedOutput([int, callback])
    if name == "union":
        return Union[callback, str]
    if name == "annotated-union":
        return Annotated[Union[callback, str], "synthetic metadata"]
    if name == "alias-union":
        return TypeAliasType("OutputAlternatives", Union[callback, str])
    if name == "list-annotation":
        return list[callback]
    if name == "dict-annotation":
        return dict[str, callback]
    if name == "tuple-annotation":
        return tuple[callback, ...]
    if name == "sequence-annotation":
        return typing.Sequence[callback]
    if name == "nested-annotation":
        return list[dict[str, callback]]
    raise AssertionError(name)


@pytest.mark.parametrize("async_callback", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("form", [
    "plain", "partial", "tool", "text", "native", "prompted", "list", "tuple",
    "nested", "native-alternatives", "prompted-alternatives", "union",
    "annotated-union", "alias-union",
    "list-annotation", "dict-annotation", "tuple-annotation", "sequence-annotation",
    "nested-annotation",
])
def test_output_functions_are_refused_before_preparation(form: str, async_callback: bool) -> None:
    calls: list[str] = []

    def process(value: str = "synthetic") -> str:
        calls.append(value)
        raise AssertionError("the original output function must never run")

    async def process_async(value: str = "synthetic") -> str:
        calls.append(value)
        raise AssertionError("the original async output function must never run")

    output = _output_form(form, process_async if async_callback else process)
    target = Agent(output_type=output)
    adapter = PydanticAIAdapter()
    report = adapter.preflight(target)
    assert [(issue.code, issue.location) for issue in report.issues] == [
        ("unsupported_output_function", "agent.output_type")
    ]
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        adapter.prepare(target, object())
    assert calls == []


def test_bound_output_method_is_refused_without_calling_it() -> None:
    class Processor:
        def process(self, value: str) -> str:
            raise AssertionError("bound output method must never run")

    target = Agent(output_type=ToolOutput(Processor().process))
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        PydanticAIAdapter().prepare(target, object())


@pytest.mark.parametrize("wrapper", ["direct", "tool", "native", "prompted"])
@pytest.mark.parametrize("form", [
    "before", "after", "plain", "wrap", "serializer", "field-validator",
    "model-validator", "function-field", "model-post-init", "data-validator",
    "data-post-init", "model-factory", "data-factory", "custom-init",
    "field-serializer", "computed-field", "nested-model",
])
def test_schema_callbacks_are_refused_without_execution(form: str, wrapper: str) -> None:
    calls: list[str] = []

    def callback(value: str = "synthetic") -> str:
        calls.append(value)
        raise AssertionError("original schema callback must never run")

    def wrapped(value: Any, handler: Any) -> Any:
        return callback(value)

    class ValidatedModel(BaseModel):
        value: str
        check = field_validator("value")(callback)

    class ModelValidator(BaseModel):
        value: str

        @model_validator(mode="after")
        def check(self) -> Any:
            return callback(self.value)

    class FunctionField(BaseModel):
        value: callback

    class PostInitModel(BaseModel):
        value: str

        def model_post_init(self, context: Any) -> None:
            callback(self.value)

    @pydantic_dataclass
    class ValidatedData:
        value: str
        check = field_validator("value")(callback)

    @dataclass
    class PostInitData:
        value: str

        def __post_init__(self) -> None:
            callback(self.value)

    class ModelFactory(BaseModel):
        value: str = Field(default_factory=callback)

    @dataclass
    class DataFactory:
        value: str = field(default_factory=callback)

    class CustomInit(BaseModel):
        value: str

        def __init__(self, **data: Any) -> None:
            callback()

    class SerializedModel(BaseModel):
        value: str
        serialize = field_serializer("value")(callback)

    class ComputedModel(BaseModel):
        value: str

        @computed_field
        @property
        def computed(self) -> str:
            return callback(self.value)

    forms = {
        "before": Annotated[str, BeforeValidator(callback)],
        "after": Annotated[str, AfterValidator(callback)],
        "plain": Annotated[str, PlainValidator(callback)],
        "wrap": Annotated[str, WrapValidator(wrapped)],
        "serializer": Annotated[str, PlainSerializer(callback)],
        "field-validator": ValidatedModel,
        "model-validator": ModelValidator,
        "function-field": FunctionField,
        "model-post-init": PostInitModel,
        "data-validator": ValidatedData,
        "data-post-init": PostInitData,
        "model-factory": ModelFactory,
        "data-factory": DataFactory,
        "custom-init": CustomInit,
        "field-serializer": SerializedModel,
        "computed-field": ComputedModel,
        "nested-model": list[ValidatedModel],
    }
    output = forms[form]
    if wrapper != "direct":
        output = {"tool": ToolOutput, "native": NativeOutput, "prompted": PromptedOutput}[wrapper](output)
    target = Agent(output_type=output)
    report = PydanticAIAdapter().preflight(target)
    assert [(issue.code, issue.location) for issue in report.issues] == [
        ("unsupported_output_function", "agent.output_type")
    ]
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        PydanticAIAdapter().prepare(target, object())
    assert calls == []


@pytest.mark.parametrize("form", ["method", "protocol", "extras", "annotation", "nested-annotation"])
def test_schema_generation_hooks_are_refused_before_inspection_or_preparation(form: str) -> None:
    calls: list[str] = []

    class MethodModel(BaseModel):
        value: str

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Any:
            calls.append("method")
            return super().model_json_schema(*args, **kwargs)

    class ProtocolModel(BaseModel):
        value: str

        @classmethod
        def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
            calls.append("protocol")
            return handler(schema)

    def extras(schema: Any) -> None:
        calls.append("extras")

    class ExtrasModel(BaseModel):
        value: str
        model_config = ConfigDict(json_schema_extra=extras)

    class Annotation:
        def __get_pydantic_json_schema__(self, schema: Any, handler: Any) -> Any:
            calls.append("annotation")
            return handler(schema)

    annotated = Annotated[str, Annotation()]

    class NestedModel(BaseModel):
        value: annotated

    output = {"method": MethodModel, "protocol": ProtocolModel, "extras": ExtrasModel,
              "annotation": annotated, "nested-annotation": NestedModel}[form]
    target = Agent(output_type=output)
    # SDK construction is outside AgentCheck's boundary and may call schema
    # hooks. The refusal must add no calls of its own after target admission.
    calls.clear()
    adapter = PydanticAIAdapter()
    assert not adapter.preflight(target).supported
    adapter.inspect(target)
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        adapter.prepare(target, object())
    assert calls == []


@pytest.mark.parametrize("container", ["marker", "list", "agent"])
def test_changed_output_declaration_is_refused(container: str) -> None:
    class ChangedModel(BaseModel):
        value: str

        @field_validator("value")
        @classmethod
        def validate_value(cls, value: str) -> str:
            raise AssertionError("changed output validator must never run")

    output: Any = ToolOutput(Result) if container == "marker" else [Result]
    target = Agent(output_type=output)
    if container == "marker":
        output.output = ChangedModel
    elif container == "list":
        output[0] = ChangedModel
    else:
        target._output_type = ChangedModel
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        PydanticAIAdapter().prepare(target, object())


def test_unknown_validator_is_refused_without_reduction_dispatch() -> None:
    class UnknownValidator:
        def __reduce__(self) -> Any:
            raise AssertionError("an unknown validator must not be reduced")

    target = Agent(output_type=Result)
    target._output_schema.processor.validator = UnknownValidator()
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        PydanticAIAdapter().prepare(target, object())


class Result(BaseModel):
    value: str


@dataclass
class DataResult:
    value: str


class DictResult(TypedDict):
    value: str


class DefaultResult(BaseModel):
    value: str = "synthetic"
    items: list[str] = Field(default_factory=list)
    model_config = ConfigDict(json_schema_extra={"description": "static metadata"})


@dataclass
class DefaultDataResult:
    value: str = "synthetic"
    items: list[str] = field(default_factory=list)


class RecursiveResult(BaseModel):
    children: list[RecursiveResult] = []


@pytest.mark.parametrize("output", [
    str, int, bool, Result, DataResult, DictResult, Literal["accepted", "refused"],
    list[Result], dict[str, int], Result | None, Union[Result, DataResult],
    list[Literal["accepted", "refused"]],
    Annotated[int, "synthetic metadata"], NewType("ResultName", str),
    TypeAliasType("ResultAlias", Union[Result, DataResult]),
    ToolOutput(Result), NativeOutput(Result), PromptedOutput(Result),
    NativeOutput([Result, DataResult]), PromptedOutput([Result, DataResult]),
    [str, Result], (str, Result), [str, [ToolOutput(Result)]],
    StructuredDict({"type": "object", "properties": {"value": {"type": "string"}}}),
    DefaultResult, DefaultDataResult, RecursiveResult,
    Annotated[str, WithJsonSchema({"type": "string"})],
])
def test_data_output_declarations_keep_their_preparation_contract(output: Any) -> None:
    target = Agent(output_type=output)
    adapter = PydanticAIAdapter()
    assert adapter.preflight(target).supported
    prepared = adapter.prepare(target, object())
    assert prepared.runtime_agent.output_type is output


@pytest.mark.parametrize("kind", ["marker", "sequence", "cycle", "missing-marker-field"])
def test_unknown_output_containers_are_refused_without_accessors(kind: str) -> None:
    class CustomMarker(ToolOutput):
        @property
        def output(self) -> Any:
            raise AssertionError("custom marker property must not run")

    class CustomList(list):
        def __iter__(self):
            raise AssertionError("custom output sequence must not be iterated")

    target = Agent()
    # Unknown-state probes intentionally replace stored declarations after SDK
    # construction; they do not claim these malformed forms construct natively.
    if kind == "marker":
        output: Any = object.__new__(CustomMarker)
    elif kind == "sequence":
        output = CustomList([str])
    elif kind == "cycle":
        output = []
        output.append(output)
    else:
        output = ToolOutput(Result)
        del output.output
    target._output_type = output
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        PydanticAIAdapter().prepare(target, object())


@pytest.mark.skipif(not hasattr(typing, "TypeAliasType"), reason="native type aliases require Python 3.12")
def test_native_alias_union_cannot_hide_an_output_function() -> None:
    def process(value: str) -> str:
        raise AssertionError("aliased output callback must never run")

    alias = typing.TypeAliasType("NativeOutputAlternatives", Union[process, str])
    target = Agent(output_type=alias)
    with pytest.raises(UnsupportedTargetError, match="unsupported_output_function"):
        PydanticAIAdapter().prepare(target, object())
