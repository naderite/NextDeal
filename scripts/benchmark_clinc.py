"""Run the NextDeal intent classifier on the official in-scope CLINC150 split."""
import argparse
import json
from pathlib import Path
import sys

from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nlp import IntentClassifier

SOURCE = "https://aclanthology.org/D19-1131/"


def benchmark(data):
    train = data["train"]
    test = data["test"]
    if len(train) != 15000 or len(test) != 4500 or len({label for _, label in train}) != 150:
        raise ValueError("Expected the publisher's full CLINC150 train and test splits")
    classifier = IntentClassifier().fit(*zip(*train))
    questions, truth = zip(*test)
    predicted = classifier.predict(questions)
    f1 = float(f1_score(truth, predicted, average="macro"))
    return {"source": SOURCE, "dataset": "CLINC150", "intents": 150, "train": len(train), "test": len(test),
            "macro_f1": f1, "accuracy": float(accuracy_score(truth, predicted)),
            "target_macro_f1": .80, "target_met": f1 >= .80,
            "scope": "Original in-scope English intent labels; excludes out-of-scope rows and NextDeal's legacy rules."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    args.output.write_text(json.dumps(benchmark(data), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
