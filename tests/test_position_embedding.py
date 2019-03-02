from unittest import TestCase
import torch
from torch_position_embedding import PositionEmbedding


class TestPositionEmbedding(TestCase):

    def test_mode_expand(self):
        net = PositionEmbedding(num_embeddings=5, embedding_dim=10, mode=PositionEmbedding.MODE_EXPAND)
        print(net)
        x = torch.Tensor([[0, 1, -1, 100, -100], [1, 2, 3, 4, 5]])
        y = net(x)
        self.assertEqual(torch.Size([2, 5, 10]), y.size())
        self.assertTrue(net.weight[-1].allclose(y[0, -2]), (net.weight[-1], y[0, -2]))

    def test_mode_add(self):
        net = PositionEmbedding(num_embeddings=5, embedding_dim=10, mode=PositionEmbedding.MODE_ADD)
        print(net)
        x = torch.randn(3, 4, 10)
        y = net(x)
        self.assertEqual(torch.Size([3, 4, 10]), y.size())

    def test_mode_concat(self):
        net = PositionEmbedding(num_embeddings=5, embedding_dim=10, mode=PositionEmbedding.MODE_CONCAT)
        print(net)
        x = torch.randn(3, 4, 6)
        y = net(x)
        self.assertEqual(torch.Size([3, 4, 16]), y.size())

    def test_mode_invalid(self):
        with self.assertRaises(NotImplementedError):
            net = PositionEmbedding(num_embeddings=5, embedding_dim=10, mode='INVALID')
            x = torch.randn(3, 4, 6)
            y = net(x)
            self.assertEqual(torch.Size([3, 4, 16]), y.size())
