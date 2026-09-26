"""Invariant-category mutations, isolated imports, clean baseline required.

Recipes target semantic transformations rather than line numbers. The old 54
recipes remain archived with their source revision; they are not counted as new
kills after the architecture they modified has been removed.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TESTS = sorted(
    str(p.relative_to(ROOT))
    for p in (ROOT / "tests/agentcheck").glob("test_fabricated_success*.py")
)
RECIPES = ROOT / "tests/fixtures/fabricated_success/semantic_mutations.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pytest_args = ["-p", "no:cacheprovider", *TESTS, "-q", "--tb=short"]
    baseline = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", *pytest_args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if baseline.returncode:
        print(baseline.stdout + baseline.stderr)
        return 2
    recipes = json.loads(RECIPES.read_text())
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "agentcheck").rglob("*.py")
    }
    results = []
    for recipe in recipes:
        module = "agentcheck.evaluate." + recipe["module"]
        path = ROOT / "agentcheck/evaluate" / (recipe["module"] + ".py")
        source = path.read_text()
        if "append" in recipe:
            mutant = source + recipe["append"]
        else:
            if source.count(recipe["old"]) != 1:
                raise RuntimeError("Invalid mutation anchor: " + recipe["name"])
            mutant = source.replace(recipe["old"], recipe["new"])
        child = f"""
import sys, importlib.abc, importlib.machinery, importlib.util
sys.path.insert(0,{str(ROOT)!r})
class Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return compile({mutant!r}, self.path, 'exec')
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {module!r}:
            return importlib.util.spec_from_file_location(fullname,{str(path)!r},loader=Loader(fullname,{str(path)!r}))
sys.meta_path.insert(0,Finder())
import json
from pathlib import Path
from agentcheck.domain import Scenario, CanonicalRun
from agentcheck.evaluate import evaluate_run
checks=[]
for c in json.loads(Path({str(ROOT / "tests/fixtures/fabricated_success/review_01a9c45_guards.json")!r}).read_text())["cases"]:
    v=evaluate_run(Scenario.model_validate_json(json.dumps(c["scenario"])),CanonicalRun.model_validate_json(json.dumps(c["run"]))).verdict.value
    checks.append(dict(id=c["id"],allowed=c["allowed"],actual=v))
print("DIFFERENTIALS="+json.dumps(checks),flush=True)
import pytest
raise SystemExit(pytest.main({pytest_args!r}))
"""
        done = subprocess.run(
            [sys.executable, "-B", "-c", child],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=90,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = done.stdout + done.stderr
        killed = (
            done.returncode == 1
            and "AssertionError" in output
            and "ERROR collecting" not in output
            and "ERROR at setup" not in output
        )
        row = {
            **recipe,
            "classification": "KILLED"
            if killed
            else "SURVIVED"
            if done.returncode == 0
            else "ERROR",
            "exit_code": done.returncode,
            "guard_differentials": [
                json.loads(line.split("=", 1)[1])
                for line in output.splitlines()
                if line.startswith("DIFFERENTIALS=")
            ],
            "summary": output.splitlines()[-1:],
            "failing_tests": [
                line for line in output.splitlines() if line.startswith("FAILED ")
            ],
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "mutant_sha256": hashlib.sha256(mutant.encode()).hexdigest(),
            "stdout_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }
        results.append(row)
        print(
            json.dumps(
                {k: row[k] for k in ["name", "category", "classification", "summary"]}
            ),
            flush=True,
        )
        args.output.write_text(
            json.dumps(
                {
                    "baseline": baseline.stdout.splitlines()[-1:],
                    "tests": TESTS,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "agentcheck").rglob("*.py")
    }
    return 0 if all(r["classification"] == "KILLED" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
