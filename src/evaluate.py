import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import LABEL_NAMES, SentimentDataset, load_vocab
from src.metrics import compute_metrics
from src.models import build_model
from src.utils import ensure_dir, get_device, load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/textcnn.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device)
    data_dir = Path(config["data"]["data_dir"])
    output_dir = ensure_dir(config["output"]["output_dir"])
    run_name = config["output"].get("run_name", config["model"]["name"])

    vocab = load_vocab(data_dir / config["data"]["vocab_file"])
    dataset = SentimentDataset(data_dir / f"{args.split}.csv", vocab, config["data"]["max_len"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    checkpoint_path = args.checkpoint or str(Path(config["output"]["checkpoint_dir"]) / f"{run_name}_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(config, len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            batch_predictions = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(batch_predictions)
            labels.extend(batch["label"].tolist())

    metrics = compute_metrics(labels, predictions)
    metrics["checkpoint"] = checkpoint_path
    metrics["split"] = args.split
    save_json(metrics, output_dir / f"{run_name}_{args.split}_metrics.json")

    prediction_path = output_dir / f"{run_name}_{args.split}_predictions.csv"
    with open(prediction_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold", "prediction", "gold_name", "prediction_name"])
        for gold, pred in zip(labels, predictions):
            writer.writerow([gold, pred, LABEL_NAMES[gold], LABEL_NAMES[pred]])

    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    metrics_path = output_dir / f"{run_name}_{args.split}_metrics.json"
    print(f"metrics_file={metrics_path}")
    print(f"predictions_file={prediction_path}")


if __name__ == "__main__":
    main()
