"""Fail the build if a distribution contains anything it should not ship.

A wheel is the thing users actually install, and an sdist is the thing that
gets mirrored. Both are easy to pollute by accident: a stray run artifact, a
local checkout, an .env someone was testing with. This runs over the built
artifacts rather than the source tree, because the source tree is not what
gets published.

PyPI also renders the README from distribution metadata, where repository-
relative Markdown links resolve against the package index instead of the repo.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tarfile
import zipfile


# Anything matching these must never appear in a built artifact.
FORBIDDEN = (
    (re.compile(r"(^|/)\.env($|\.)"), "environment file"),
    (re.compile(r"(^|/)\.agentcheck/"), "local evaluation run artifacts"),
    (re.compile(r"(^|/)migrations/"), "database migrations"),
    (re.compile(r"(^|/)\.venv/|(^|/)venv/"), "virtual environment"),
    (re.compile(r"(^|/)__pycache__/|\.pyc$"), "bytecode cache"),
    (re.compile(r"(^|/)\.git/"), "git metadata"),
    (re.compile(r"(^|/)node_modules/"), "frontend dependencies"),
    (re.compile(r"\.sqlite3?$"), "local database"),
    (re.compile(r"(^|/)dist/"), "nested distributions"),
)

MARKDOWN_DESTINATION = re.compile(
    r"\]\(\s*(?P<destination><[^>\r\n]+>|[^\s)\r\n]+)"
)
ABSOLUTE_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")

# The wheel is the installed package. Tests, examples, and docs belong in the
# sdist and the repository, not in every user's site-packages.
#
# The import package and the distribution have different names on purpose --
# `agentcheck` is imported, `agentcheck-ai` is installed -- and a wheel's
# dist-info directory is named after the distribution, with dashes normalised to
# underscores. Matching that shape rather than a literal keeps this check honest
# if either name changes again.
WHEEL_ALLOWED = re.compile(r"^(agentcheck/|[A-Za-z0-9_.]+-[^/]*\.dist-info/)")
WHEEL_LICENSE_PATH = re.compile(
    r"^[A-Za-z0-9_.]+-[^/]*\.dist-info/licenses/(?P<filename>[^/]+)$"
)
REQUIRED_DISTRIBUTION_FILES = ("LICENSE", "NOTICE")

# An sdist legitimately carries contributor material as well as the package,
# but a new top-level directory must be reviewed. This generic allowlist catches
# accidental sibling/private packages without naming or depending on any host
# repository.
SDIST_ALLOWED_ROOTS = frozenset(
    {
        ".github",
        "AGENTS.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "MANIFEST.in",
        "NOTICE",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "agentcheck",
        "agentcheck_ai.egg-info",
        "docs",
        "examples",
        "pyproject.toml",
        "scripts",
        "setup.cfg",
        "tests",
    }
)


def _members(path: pathlib.Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        # Strip the leading "agentcheck-0.1.0/" component an sdist always has.
        return [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]


def _metadata(path: pathlib.Path) -> tuple[str, str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if re.fullmatch(r"[^/]+\.dist-info/METADATA", name)
            ]
            if len(names) != 1:
                raise ValueError(f"expected one wheel METADATA file, found {len(names)}")
            name = names[0]
            return name, archive.read(name).decode("utf-8")

    with tarfile.open(path) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and re.fullmatch(r"[^/]+/PKG-INFO", member.name)
        ]
        if len(members) != 1:
            raise ValueError(f"expected one sdist PKG-INFO file, found {len(members)}")
        member = members[0]
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"could not read {member.name}")
        return member.name, extracted.read().decode("utf-8")


def _relative_markdown_destinations(text: str) -> set[str]:
    destinations: set[str] = set()
    for match in MARKDOWN_DESTINATION.finditer(text):
        destination = match.group("destination").strip("<>")
        if not destination.startswith("#") and ABSOLUTE_URI.match(destination) is None:
            destinations.add(destination)
    return destinations


def main() -> int:
    dist = pathlib.Path("dist")
    artifacts = sorted(dist.glob("*.whl")) + sorted(dist.glob("*.tar.gz"))
    if not artifacts:
        print("FAIL: no artifacts in dist/; run `python -m build` first")
        return 1

    failures: list[str] = []
    for artifact in artifacts:
        names = _members(artifact)
        print(f"\n=== {artifact.name}: {len(names)} entries ===")
        for name in sorted(names):
            print(f"  {name}")

        for name in names:
            for pattern, label in FORBIDDEN:
                if pattern.search(name):
                    failures.append(f"{artifact.name}: {name} ({label})")
            if artifact.suffix == ".whl":
                if not WHEEL_ALLOWED.match(name):
                    failures.append(f"{artifact.name}: {name} (not part of the package)")
            else:
                root = name.split("/", 1)[0]
                if root not in SDIST_ALLOWED_ROOTS:
                    failures.append(
                        f"{artifact.name}: {name} (unexpected sdist top-level path)"
                    )

        required_counts = dict.fromkeys(REQUIRED_DISTRIBUTION_FILES, 0)
        if artifact.suffix == ".whl":
            for name in names:
                match = WHEEL_LICENSE_PATH.fullmatch(name)
                if match is not None and match.group("filename") in required_counts:
                    required_counts[match.group("filename")] += 1
        else:
            for filename in required_counts:
                required_counts[filename] = names.count(filename)
        for filename, count in required_counts.items():
            if count != 1:
                failures.append(
                    f"{artifact.name}: expected exactly one packaged {filename}, found {count}"
                )

        try:
            metadata_name, metadata = _metadata(artifact)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            failures.append(f"{artifact.name}: invalid distribution metadata ({error})")
        else:
            for destination in sorted(_relative_markdown_destinations(metadata)):
                failures.append(
                    f"{artifact.name}: {metadata_name}: relative Markdown destination "
                    f"{destination!r}"
                )

    if failures:
        print("\nFAIL: distribution audit failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"\nPASS: {len(artifacts)} artifact(s) contain only expected files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
