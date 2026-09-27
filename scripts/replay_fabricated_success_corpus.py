"""Replay immutable historical labels separately from current contract tests."""

import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agentcheck.domain import CanonicalRun, Scenario  # noqa: E402
from agentcheck.evaluate import evaluate_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for case in json.loads(
        (ROOT / "tests/fixtures/fabricated_success/semantic_corpus.json").read_text()
    )["cases"]:
        result = evaluate_run(
            Scenario.model_validate_json(json.dumps(case["scenario"])),
            CanonicalRun.model_validate_json(json.dumps(case["run"])),
        )
        rows.append(
            {
                "dataset": case["dataset"],
                "id": case["id"],
                "allowed": case["allowed"],
                "actual": result.verdict.value,
                "error": result.verdict.value not in case["allowed"],
            }
        )
    counts = {}
    for name in dict.fromkeys(r["dataset"] for r in rows):
        group = [r for r in rows if r["dataset"] == name]
        wrong = collections.Counter(r["actual"] for r in group if r["error"])
        counts[name] = {
            "n": len(group),
            "false_pass": wrong["PASS"],
            "false_fail": wrong["FAIL"],
            "wrong_inconclusive": wrong["INCONCLUSIVE"],
        }
    counts["combined"] = {
        k: sum(v[k] for v in counts.values())
        for k in ("n", "false_pass", "false_fail", "wrong_inconclusive")
    }
    args.output.write_text(
        json.dumps({"counts": counts, "rows": rows}, indent=2) + "\n"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
