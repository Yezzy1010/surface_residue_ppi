"""Synthetic checks, never reported as real-protein pipeline validation."""
import unittest
import torch
from surface_residue_ppi.adapters import neighborhoods, SurfaceAdapter


class AdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.atoms = torch.tensor([[0., 0, 0], [2., 0, 0], [20., 0, 0]], device="cuda")
        self.ids = torch.tensor([0, 1, 2], device="cuda")
        self.points = torch.tensor([[1., 0, 0], [0., 1, 0], [1., 1, 0]], device="cuda")
        self.h = torch.randn(4, 8, device="cuda")
        self.f = torch.randn(3, 6, device="cuda")

    def test_exact_geometry_and_empty_residues(self):
        a, b, owner = neighborhoods(self.atoms, self.ids, self.points, 4, 3, 0)
        self.assertEqual(owner.tolist(), [0, 0, 0])
        self.assertEqual(len(a.residue), 3)
        self.assertEqual(len(b.residue), 5)
        for mode, edges in [("A", a), ("B", b), ("C", b)]:
            adapter = SurfaceAdapter(mode, 8, 6).cuda()
            output = adapter(self.h, self.f, edges)
            self.assertTrue(torch.equal(output[2:], self.h[2:]))
            output.square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in adapter.parameters()))
            self.assertGreater(adapter.value.weight.grad.abs().sum().item(), 0)

    def test_point_permutation_and_rigid_motion(self):
        a, b, _ = neighborhoods(self.atoms, self.ids, self.points, 4)
        permutation = torch.tensor([2, 0, 1], device="cuda")
        rotation = torch.tensor([[0., -1, 0], [1., 0, 0], [0., 0, 1]], device="cuda")
        other_a, other_b, _ = neighborhoods(self.atoms @ rotation + 5, self.ids,
                                           self.points[permutation] @ rotation + 5, 4)
        for mode, original, other in [("A", a, other_a), ("B", b, other_b), ("C", b, other_b)]:
            adapter = SurfaceAdapter(mode, 8, 6).cuda().eval()
            self.assertTrue(torch.allclose(adapter(self.h, self.f, original),
                            adapter(self.h, self.f[permutation], other), atol=1e-6))

    def test_empty_surface(self):
        a, b, owner = neighborhoods(self.atoms, self.ids, self.points[:0], 4)
        self.assertEqual(len(owner), 0)
        for mode in ("A", "B", "C"):
            self.assertTrue(torch.equal(SurfaceAdapter(mode, 8, 6).cuda()(self.h,
                            self.f[:0], b), self.h))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("Run with sbatch on a GPU node")
    unittest.main()
