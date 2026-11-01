"""Offline tests using an actual tiny DistilBERT and local tokenizer."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

from deep_truth.data.loaders import make_loaders, prepare_distilbert_data
from deep_truth.metrics import compute_metrics
from deep_truth.models.distilbert import build_distilbert_model
from deep_truth.train_distilbert import build_optimizer, evaluate, train
from tests.helpers import make_config


class DistilBertTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = make_config(self.root, [
            ('a', 'false', 'wrong claim'), ('b', 'true', 'correct claim'),
            ('c', 'false', 'wrong statement'), ('d', 'true', 'correct statement'),
        ], [('v', 'true', 'correct validationonly')])
        self.config.split_path('test').unlink()
        local = self.root / 'pretrained'
        local.mkdir()
        (local / 'vocab.txt').write_text(
            '[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwrong\ncorrect\nclaim\nstatement\nvalidationonly\n'
        )
        tokenizer = BertTokenizerFast(vocab_file=str(local / 'vocab.txt'), do_lower_case=True)
        tokenizer.save_pretrained(local)
        torch.manual_seed(42)
        tiny = DistilBertConfig(vocab_size=10, max_position_embeddings=16,
                               n_layers=1, n_heads=2, dim=8, hidden_dim=16,
                               num_labels=6, dropout=0.0, attention_dropout=0.0,
                               seq_classif_dropout=0.0)
        DistilBertForSequenceClassification(tiny).save_pretrained(local)
        self.config.data['distilbert']['model_name'] = str(local)
        for mode in ('frozen', 'finetuned'):
            self.config.data['distilbert'][mode].update(
                max_epochs=5, patience=2,
                grid={'learning_rate': [0.001], 'batch_size': [2]},
            )

    def test_encoder_updates_only_when_finetuned_and_head_always_updates(self):
        for run in ('distilbert_frozen', 'distilbert_finetuned'):
            with self.subTest(run=run):
                model = build_distilbert_model(self.config, run)
                before = {name: p.detach().clone() for name, p in model.named_parameters()}
                # Isolate gradient behavior from the warmup scheduler (which starts at zero).
                optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001)
                loss = model(input_ids=torch.tensor([[2, 6, 7, 3, 0]]),
                             attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
                             labels=torch.tensor([5])).loss
                loss.backward()
                optimizer.step()
                changed = {name for name, p in model.named_parameters()
                           if not torch.equal(before[name], p)}
                self.assertTrue(any(name.startswith('classifier.') for name in changed))
                encoder_changed = any(name.startswith('distilbert.') for name in changed)
                self.assertEqual(encoder_changed, run == 'distilbert_finetuned')
                self.assertEqual(model.config.label2id, self.config.label_to_index)

    def test_scheduler_is_constant_for_frozen_and_warms_up_then_decays_for_full(self):
        for run in ('distilbert_frozen', 'distilbert_finetuned'):
            model = build_distilbert_model(self.config, run)
            optimizer, scheduler = build_optimizer(
                model, {'learning_rate': 0.001}, self.config.data['distilbert'], run, 10,
            )
            rates = [optimizer.param_groups[0]['lr']]
            for _ in range(10):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                rates.append(optimizer.param_groups[0]['lr'])
            if run.endswith('frozen'):
                self.assertIsNone(scheduler)
                self.assertEqual(set(rates), {0.001})
            else:
                self.assertEqual(rates[0], 0.0)
                self.assertEqual(max(rates), 0.001)
                self.assertEqual(rates[-1], 0.0)

    def test_both_modes_train_save_and_reload_without_test_data(self):
        for run in ('distilbert_frozen', 'distilbert_finetuned'):
            with self.subTest(run=run):
                artifacts, results = self.root / run, self.root / (run + '_results')
                summary = train(self.config, run, artifacts, results, smoke_test=True)
                self.assertTrue(summary['smoke_test'])
                self.assertEqual(summary['search_configurations'], 1)
                restored = DistilBertForSequenceClassification.from_pretrained(artifacts / 'model')
                tokenizer = BertTokenizerFast.from_pretrained(artifacts / 'model')
                data = prepare_distilbert_data(self.config, tokenizer=tokenizer)
                loader = make_loaders(data, 2, self.config.seed).validation
                metrics, predictions = evaluate(restored, loader, self.config, 'cpu')
                self.assertEqual(metrics, summary['selected']['validation'])
                output = pd.read_csv(results / 'validation_predictions.csv', keep_default_na=False, dtype=str)
                self.assertEqual(output.prediction.tolist(), predictions)
                self.assertEqual(output.id.tolist(), ['v'])
                self.assertEqual(json.loads((artifacts / 'selection.json').read_text())['run'], run)

    def test_patience_restores_first_epoch_on_validation_plateau(self):
        metrics = compute_metrics(['true'], ['true'], self.config.labels)
        with patch('deep_truth.train_distilbert.evaluate', return_value=(metrics, ['true'])):
            summary = train(self.config, 'distilbert_frozen', self.root / 'artifacts', self.root / 'results')
        history = json.loads((self.root / 'results/search.json').read_text())[0]['history']
        self.assertEqual(len(history), 3)
        self.assertEqual(summary['selected']['epoch'], 1)

