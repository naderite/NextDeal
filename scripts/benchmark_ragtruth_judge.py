"""Run NextDeal's live groundedness judge on a labeled RAGTruth QA test sample."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = "https://aclanthology.org/2024.acl-long.585/"


def sample(data, size):
    with (data / "source_info.jsonl").open(encoding="utf-8") as stream:
        source = {item["source_id"]: item["source_info"] for line in stream
                  if (item := json.loads(line))["task_type"] == "QA"}
    with (data / "response.jsonl").open(encoding="utf-8") as stream:
        eligible = [item for line in stream if (item := json.loads(line))["split"] == "test"
                    and item["quality"] == "good" and item["source_id"] in source]
    rng = random.Random(42)
    rng.shuffle(eligible)
    cases, seen = [], set()
    remaining = {True: size // 2, False: size // 2}
    for response in eligible:
        grounded = not response["labels"]
        key = response["source_id"]
        if not remaining[grounded] or key in seen:
            continue
        remaining[grounded] -= 1
        seen.add(key)
        cases.append((response, grounded, source[key]["passages"]))
    if any(remaining.values()):
        raise ValueError("Too few distinct sources for the requested balanced sample")
    return cases


def write_result(path, result):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def benchmark(data, size, output):
    from backend.agent import judge_draft
    from backend.config import OLLAMA_MODEL

    cases = sample(data, size)
    signature = hashlib.sha256(json.dumps(cases, ensure_ascii=False).encode()).hexdigest()
    judge_source = Path(sys.modules[judge_draft.__module__].__file__)
    signature += ":" + hashlib.sha256(judge_source.read_bytes()).hexdigest()
    result = {"source": SOURCE, "model": OLLAMA_MODEL, "sample_size": size, "sample_seed": 42,
              "sample_and_judge_signature": signature, "cases": [], "status": "in_progress",
              "human_label": "Grounded when the response has no annotated unsupported spans.",
              "target_kappa": .60,
              "scope": "Balanced QA pilot from the publisher's human-labeled test split, one response per source. "
                       "Measures agreement on general QA groundedness, not advisor acceptance."}
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["sample_and_judge_signature"] != signature or previous["model"] != OLLAMA_MODEL:
            raise ValueError("Output checkpoint differs from current benchmark configuration")
        result["cases"] = previous["cases"]
    for i, (response, grounded, context) in enumerate(cases[len(result["cases"]):], start=len(result["cases"]) + 1):
        judged = judge_draft(response["response"], [{"chunk_id": "source", "text": context}])
        result["cases"].append({"id": response["id"], "human_grounded": grounded,
                                "judge_grounded": judged["grounded"]})
        write_result(output, result)
        print(f"Judged {i}/{size}", flush=True)
    truth = [case["human_grounded"] for case in result["cases"]]
    predicted = [case["judge_grounded"] for case in result["cases"]]
    result.update({"status": "complete", "test_kappa": float(cohen_kappa_score(truth, predicted)),
                   "confusion_matrix_labels": [False, True],
                   "confusion_matrix": confusion_matrix(truth, predicted, labels=[False, True]).tolist()})
    write_result(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size < 2 or args.sample_size % 2:
        parser.error("Use a positive even sample size")
    benchmark(args.data, args.sample_size, args.output)


if __name__ == "__main__":
    main()
