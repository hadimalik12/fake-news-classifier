"""Validation-only search for frozen and fully fine-tuned DistilBERT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from deep_truth.config import Config, grid_size, iter_grid
from deep_truth.data.loaders import make_loaders, prepare_distilbert_data
from deep_truth.metrics import compute_metrics
from deep_truth.models.distilbert import RUNS, build_distilbert_model
from deep_truth.runtime import prepare_run
from deep_truth.seed import set_seed
from deep_truth.selection import selection_key


def evaluate(model, loader, config, device):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    labels = [config.labels[index] for index in predicted]
    return compute_metrics(loader.dataset.metadata["label"], labels, config.labels), labels


def build_optimizer(model, parameters, settings, run_name, total_steps):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=parameters["learning_rate"], weight_decay=settings["weight_decay"],
    )
    scheduler = None
    if run_name == "distilbert_finetuned":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=math.ceil(settings["finetuned_warmup_ratio"] * total_steps),
            num_training_steps=total_steps,
        )
    return optimizer, scheduler


def train(config: Config, run_name: str, artifact_dir: Path, results_dir: Path,
          device: str = "cpu", smoke_test: bool = False) -> dict:
    if run_name not in RUNS:
        raise ValueError(f"Unknown DistilBERT run: {run_name}")
    settings = config.data["distilbert"]
    section = settings["frozen" if run_name == "distilbert_frozen" else "finetuned"]
    max_epochs = 1 if smoke_test else int(section["max_epochs"])
    if max_epochs < 1 or int(section["patience"]) < 1:
        raise ValueError("max_epochs and patience must be positive.")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_name"])
    data = prepare_distilbert_data(config, tokenizer=tokenizer)
    if not len(data.train) or not len(data.validation):
        raise ValueError("Training and validation splits must be nonempty.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    grid = list(iter_grid(section["grid"]))
    if smoke_test:
        grid = grid[:1]
    records = []
    best_record = None
    checkpoint_path = artifact_dir / "model"
    for cell, parameters in enumerate(grid, 1):
        set_seed(config.seed)
        loaders = make_loaders(data, parameters["batch_size"], config.seed)
        print(f"[{cell}/{len(grid)}] {run_name}: {parameters}; loading model on {device}", flush=True)
        model = build_distilbert_model(config, run_name).to(device)
        optimizer, scheduler = build_optimizer(
            model, parameters, settings, run_name, len(loaders.train) * max_epochs,
        )
        criterion = nn.CrossEntropyLoss()
        history = []
        cell_best = None
        cell_state = None
        patience_best = float("-inf")
        stale_epochs = 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(loaders.train, 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device)).logits
                loss = criterion(logits, batch["labels"].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item() * len(batch["labels"])
                if step % 50 == 0:
                    print(f"  epoch {epoch}: batch {step}/{len(loaders.train)}", flush=True)
            metrics, _ = evaluate(model, loaders.validation, config, device)
            record = {"parameters": parameters, "validation": metrics, "epoch": epoch}
            history.append({"epoch": epoch, "train_loss": total_loss / len(data.train),
                            "validation": metrics, "learning_rate": optimizer.param_groups[0]["lr"]})
            if cell_best is None or selection_key(record, config) < selection_key(cell_best, config):
                cell_best = record
                cell_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
            # Checkpoint selection and patience use distinct improvement rules.
            if metrics["macro_f1"] - patience_best >= config.data["metric"]["early_stopping_min_delta"]:
                patience_best = metrics["macro_f1"]
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(f"[{cell}/{len(grid)}] epoch {epoch}: loss={history[-1]['train_loss']:.4f} "
                  f"validation macro-F1={metrics['macro_f1']:.4f} "
                  f"accuracy={metrics['accuracy']:.4f}", flush=True)
            if stale_epochs >= section["patience"]:
                break
        records.append({**cell_best, "history": history})
        (results_dir / "search.json").write_text(json.dumps(records, indent=2) + "\n")
        if best_record is None or selection_key(cell_best, config) < selection_key(best_record, config):
            best_record = cell_best
            model.load_state_dict(cell_state)
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            (artifact_dir / "selection.json").write_text(json.dumps({
                "run": run_name, "smoke_test": smoke_test, "selected": best_record,
                "seed": config.seed, "config": config.data,
            }, indent=2) + "\n")
        # Release the old model and optimizer before loading the next search cell.
        del model, optimizer, scheduler, cell_state
    if best_record is None:
        raise ValueError("DistilBERT search grid is empty.")
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path).to(device)
    loaders = make_loaders(data, best_record["parameters"]["batch_size"], config.seed)
    restored_metrics, predictions = evaluate(model, loaders.validation, config, device)
    if restored_metrics != best_record["validation"]:
        raise RuntimeError("Restored checkpoint does not reproduce selected validation metrics.")
    output = data.validation.metadata.copy()
    output["prediction"] = predictions
    output.to_csv(results_dir / "validation_predictions.csv", index=False)
    summary = {"run": run_name, "seed": config.seed, "device": device,
               "smoke_test": smoke_test, "selected": best_record,
               "search_configurations": len(records), "model_path": str(checkpoint_path),
               "config": config.data}
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{**r["parameters"], "epoch": r["epoch"],
                   **{k: r["validation"][k] for k in (
                       "accuracy", "macro_precision", "macro_recall", "macro_f1")}}
                  for r in records]).to_csv(results_dir / "search.csv", index=False)
    print(f"Saved {'smoke-test' if smoke_test else 'selected'} model: {checkpoint_path}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="One setting, one full epoch; separate outputs.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    config = prepare_run(args.run)
    section = "frozen" if args.run == "distilbert_frozen" else "finetuned"
    if args.dry_run:
        print(f"{args.run}: seed {config.seed}, grid size "
              f"{grid_size(config.data['distilbert'][section]['grid'])}; dry run.")
        return
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    output_name = args.run + ("_smoke" if args.smoke_test else "")
    train(config, args.run, config.path(f"artifacts/{output_name}"),
          config.path(f"results/{output_name}"), device, args.smoke_test)


if __name__ == "__main__":
    main()
