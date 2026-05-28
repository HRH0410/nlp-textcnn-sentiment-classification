import json
from pathlib import Path


def main() -> None:
    rows = []
    for path in sorted(Path("outputs").glob("*_test_metrics.json")):
        data = json.load(open(path, "r", encoding="utf-8"))
        run_name = path.name.replace("_test_metrics.json", "")
        rows.append((run_name, data["accuracy"], data["macro_f1"], data["weighted_f1"]))

    if not rows:
        print("No test metrics found in outputs/.")
        return

    print("| run | accuracy | macro_f1 | weighted_f1 |")
    print("| --- | ---: | ---: | ---: |")
    for run_name, accuracy, macro_f1, weighted_f1 in sorted(rows, key=lambda x: x[1], reverse=True):
        print(f"| {run_name} | {accuracy:.4f} | {macro_f1:.4f} | {weighted_f1:.4f} |")


if __name__ == "__main__":
    main()
