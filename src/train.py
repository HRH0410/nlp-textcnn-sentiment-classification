import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import SentimentDataset, load_vocab, make_class_weights
from src.embeddings import load_pretrained_embeddings
from src.metrics import compute_metrics
from src.models import build_model
from src.utils import ensure_dir, get_device, load_config, save_json, set_seed


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_grad_norm: float | None = None,
) -> Tuple[float, Dict[str, object]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(dataloader, leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=-1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_predictions)
    return avg_loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/textcnn.yaml")
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = get_device(args.device)

    data_dir = Path(config["data"]["data_dir"])
    output_dir = ensure_dir(config["output"]["output_dir"])
    checkpoint_dir = ensure_dir(config["output"]["checkpoint_dir"])
    run_name = config["output"].get("run_name", config["model"]["name"])

    vocab = load_vocab(data_dir / config["data"]["vocab_file"])
    train_dataset = SentimentDataset(data_dir / "train.csv", vocab, config["data"]["max_len"])
    dev_dataset = SentimentDataset(data_dir / "dev.csv", vocab, config["data"]["max_len"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    pretrained_embeddings = None
    embedding_stats = None
    pretrained_path = config["model"].get("pretrained_embeddings")
    if pretrained_path:
        pretrained_embeddings, embedding_stats = load_pretrained_embeddings(
            pretrained_path,
            vocab,
            config["model"]["embedding_dim"],
            config["training"]["seed"],
        )
        print(
            f"loaded_pretrained_embeddings={pretrained_path} "
            f"coverage={embedding_stats['coverage']:.4f} "
            f"matched={embedding_stats['matched_tokens']}/{embedding_stats['vocab_size'] - 2}"
        )

    model = build_model(config, len(vocab), pretrained_embeddings=pretrained_embeddings).to(device)
    class_weights = None
    if config["training"].get("class_weight", False):
        class_weights = make_class_weights(train_dataset.examples, config["data"]["num_classes"]).to(device)
        print(f"class_weights={[round(x, 4) for x in class_weights.detach().cpu().tolist()]}")
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["training"].get("label_smoothing", 0.0),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["training"].get("lr_reduce_factor", 0.5),
        patience=config["training"].get("lr_patience", 2),
    )

    best_accuracy = -1.0
    best_macro_f1 = -1.0
    patience_counter = 0
    early_stopping_patience = config["training"].get("early_stopping_patience", 8)
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    history = []
    start_time = time.time()

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            max_grad_norm=config["training"].get("max_grad_norm"),
        )
        dev_loss, dev_metrics = run_epoch(model, dev_loader, criterion, device)
        scheduler.step(dev_metrics["accuracy"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "dev_loss": dev_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_metrics['accuracy']:.4f} dev_macro_f1={dev_metrics['macro_f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        if dev_metrics["accuracy"] > best_accuracy:
            best_accuracy = dev_metrics["accuracy"]
            best_macro_f1 = dev_metrics["macro_f1"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocab_size": len(vocab),
                    "dev_metrics": dev_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"early_stopping_epoch={epoch}")
                break

    result = {
        "model": config["model"]["name"],
        "run_name": run_name,
        "device": str(device),
        "best_dev_accuracy": best_accuracy,
        "best_dev_macro_f1": best_macro_f1,
        "embedding_stats": embedding_stats,
        "training_seconds": round(time.time() - start_time, 2),
        "history": history,
        "best_checkpoint": str(best_path),
    }
    save_json(result, output_dir / f"{run_name}_train_metrics.json")
    print(f"best_dev_accuracy={best_accuracy:.4f}")
    print(f"saved_checkpoint={best_path}")


if __name__ == "__main__":
    main()
