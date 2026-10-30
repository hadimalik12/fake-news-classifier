from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from deep_truth.models.bilstm import BiLSTM
from deep_truth.metrics import compute_metrics
from deep_truth.train_bilstm import train
from tests.helpers import make_config


class BiLSTMTests(unittest.TestCase):
    def test_padding_invariance_batch_order_and_embedding_gradients(self):
        torch.manual_seed(42)
        embeddings = torch.randn(8, 3)
        embeddings[0] = 0
        model = BiLSTM(embeddings, 4, 0.3, 6).eval()
        short = model(torch.tensor([[2, 3]]), torch.tensor([2]))
        padded = model(torch.tensor([[4, 5, 6, 0], [2, 3, 0, 0]]), torch.tensor([3, 2]))
        torch.testing.assert_close(short[0], padded[1])
        self.assertEqual(tuple(padded.shape), (2, 6))
        padded.sum().backward()
        self.assertGreater(model.embedding.weight.grad[2].abs().sum().item(), 0)
        self.assertEqual(model.embedding.weight.grad[0].abs().sum().item(), 0)
        torch.testing.assert_close(embeddings, model.embedding.weight.detach())
        with self.assertRaises(ValueError):
            model(torch.tensor([[0]]), torch.tensor([0]))

    def test_training_stops_on_plateau_and_saves_reloadable_winner_without_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, [
                ('a', 'false', 'wrong claim'), ('b', 'true', 'correct claim'),
                ('c', 'false', 'wrong statement'), ('d', 'true', 'correct statement'),
            ], [('v', 'true', 'correct validationonly')])
            config.split_path('test').unlink()
            config.data['bilstm']['grid'] = {
                'learning_rate': [0.001], 'hidden_size': [4], 'dropout': [0.3], 'batch_size': [2],
            }
            config.data['bilstm'].update(max_epochs=6, patience=2)
            glove = config.path(config.data['embeddings']['glove']['path'])
            glove.write_text('correct 0.1 0.2 0.3\nwrong 0.3 0.2 0.1\n')
            metrics = compute_metrics(['true'], ['true'], config.labels)
            with patch('deep_truth.train_bilstm.evaluate', return_value=(metrics, ['true'])):
                summary = train(config, root / 'artifacts', root / 'results')
            checkpoint = torch.load(root / 'artifacts/model.pt', weights_only=True)
            self.assertNotIn('validationonly', checkpoint['index_to_token'])
            self.assertEqual(checkpoint['epoch'], 1)
            import json
            records = json.loads((root / 'results/search.json').read_text())
            self.assertEqual(len(records[0]['history']), 3)
            self.assertEqual(summary['search_configurations'], 1)
            weights = checkpoint['state_dict']['embedding.weight']
            model = BiLSTM(weights, 4, 0.3, 6)
            model.load_state_dict(checkpoint['state_dict'])
            self.assertTrue(torch.isfinite(model(torch.tensor([[1, 0]]), torch.tensor([1]))).all())
            self.assertEqual(pd.read_csv(root / 'results/validation_predictions.csv').iloc[0]['id'], 'v')
