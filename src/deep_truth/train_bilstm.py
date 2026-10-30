"""Train and select the GloVe-initialized BiLSTM on validation data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from deep_truth.config import Config, grid_size, iter_grid
from deep_truth.data.loaders import make_loaders, prepare_bilstm_data
from deep_truth.metrics import compute_metrics
from deep_truth.models.bilstm import BiLSTM
from deep_truth.runtime import prepare_run
from deep_truth.seed import set_seed
from deep_truth.selection import selection_key


def evaluate(model, loader, config, device):
    model.eval()
    predicted = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device), batch['length'])
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    labels = [config.labels[index] for index in predicted]
    return compute_metrics(loader.dataset.metadata['label'], labels, config.labels), labels


def train(config: Config, artifact_dir: Path, results_dir: Path,
          device: str = 'cpu', smoke_test: bool = False) -> dict:
    data = prepare_bilstm_data(config)
    if torch.any(data.datasets.train.lengths == 0) or torch.any(data.datasets.validation.lengths == 0):
        raise ValueError('BiLSTM requires nonempty statements.')
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    settings = config.data['bilstm']
    grid = list(iter_grid(settings['grid']))
    if smoke_test:
        grid = grid[:1]
    best_record = None
    records = []
    for cell, parameters in enumerate(grid, 1):
        # Reset initialization and shuffle order for each search cell.
        set_seed(config.seed)
        loaders = make_loaders(data.datasets, parameters['batch_size'], config.seed)
        model = BiLSTM(data.embeddings.weights, parameters['hidden_size'],
                       parameters['dropout'], len(config.labels),
                       data.vocabulary.pad_index).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'],
                                     weight_decay=settings['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        cell_best = None
        cell_state = None
        history = []
        patience_best = float('-inf')
        stale_epochs = 0
        for epoch in range(1, (1 if smoke_test else settings['max_epochs']) + 1):
            model.train()
            total_loss = 0.0
            for batch in loaders.train:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch['input_ids'].to(device), batch['length'])
                loss = criterion(logits, batch['labels'].to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'])
                optimizer.step()
                total_loss += loss.item() * len(batch['labels'])
            metrics, _ = evaluate(model, loaders.validation, config, device)
            record = {'parameters': parameters, 'validation': metrics, 'epoch': epoch}
            history.append({'epoch': epoch, 'train_loss': total_loss / len(data.datasets.train),
                            'validation': metrics})
            if cell_best is None or selection_key(record, config) < selection_key(cell_best, config):
                cell_best = record
                cell_state = {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}
            # Patience has its own threshold, independent of checkpoint selection.
            if metrics['macro_f1'] - patience_best >= config.data['metric']['early_stopping_min_delta']:
                patience_best = metrics['macro_f1']
                stale_epochs = 0
            else:
                stale_epochs += 1
            print(f"[{cell}/{len(grid)}] epoch {epoch}: loss={history[-1]['train_loss']:.4f} "
                  f"validation macro-F1={metrics['macro_f1']:.4f} "
                  f"accuracy={metrics['accuracy']:.4f}", flush=True)
            if stale_epochs >= settings['patience']:
                break
        if cell_best is None:
            raise ValueError('bilstm.max_epochs must be positive.')
        records.append({**cell_best, 'history': history})
        (results_dir / 'search.json').write_text(json.dumps(records, indent=2) + '\n')
        if best_record is None or selection_key(cell_best, config) < selection_key(best_record, config):
            best_record = cell_best
            torch.save({
                'state_dict': cell_state, 'parameters': parameters,
                'epoch': cell_best['epoch'], 'validation': cell_best['validation'],
                'index_to_token': list(data.vocabulary.index_to_token),
                'config': config.data, 'smoke_test': smoke_test,
            }, artifact_dir / 'model.pt')
    if best_record is None:
        raise ValueError('BiLSTM search grid is empty.')
    # Reload the serialized winner before exporting its validation predictions.
    checkpoint = torch.load(artifact_dir / 'model.pt', map_location='cpu', weights_only=True)
    parameters = checkpoint['parameters']
    model = BiLSTM(data.embeddings.weights, parameters['hidden_size'],
                   parameters['dropout'], len(config.labels), data.vocabulary.pad_index)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    loaders = make_loaders(data.datasets, parameters['batch_size'], config.seed)
    restored_metrics, predictions = evaluate(model, loaders.validation, config, device)
    if restored_metrics != best_record['validation']:
        raise RuntimeError('Restored checkpoint does not reproduce selected validation metrics.')
    output = data.datasets.validation.metadata.copy()
    output['prediction'] = predictions
    output.to_csv(results_dir / 'validation_predictions.csv', index=False)
    summary = {'run': 'bilstm', 'seed': config.seed, 'device': device,
               'smoke_test': smoke_test, 'selected': best_record,
               'search_configurations': len(records), 'config': config.data,
               'vocabulary_size': len(data.vocabulary),
               'glove_matched_tokens': data.embeddings.matched_tokens}
    (results_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    pd.DataFrame([{**r['parameters'], 'epoch': r['epoch'],
                   **{k: r['validation'][k] for k in ('accuracy', 'macro_f1')}}
                  for r in records]).to_csv(results_dir / 'search.csv', index=False)
    print(f"Saved {'smoke-test' if smoke_test else 'selected'} model: {artifact_dir / 'model.pt'}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='One configuration, one epoch; separate outputs.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4, help='CPU worker threads (default: 4).')
    args = parser.parse_args(argv)
    config = prepare_run('bilstm')
    if args.dry_run:
        print(f"bilstm: seed {config.seed}, grid size {grid_size(config.data['bilstm']['grid'])}; dry run.")
        return
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    run = 'bilstm_smoke' if args.smoke_test else 'bilstm'
    train(config, config.path(f'artifacts/{run}'), config.path(f'results/{run}'), device, args.smoke_test)


if __name__ == '__main__':
    main()
