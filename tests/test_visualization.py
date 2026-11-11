"""Unit tests for 2-simplex visualization."""

import unittest
from pathlib import Path
import numpy as np

from simplexity.visualization import plot_mess3_ground_truth, simplex_2d_projection


class TestVisualization(unittest.TestCase):
    def test_simplex_projection_vertices(self):
        # Vertex 0: (1, 0, 0) -> (0.0, 0.0)
        # Vertex 1: (0, 1, 0) -> (1.0, 0.0)
        # Vertex 2: (0, 0, 1) -> (0.5, sqrt(3)/2)
        beliefs = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        x, y = simplex_2d_projection(beliefs)
        np.testing.assert_allclose(x, [0.0, 1.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(y, [0.0, 0.0, np.sqrt(3.0) / 2.0], atol=1e-12)

    def test_gate4_mess3_plot_generation(self):
        out_path = Path("figures/test_GT_mess3.png")
        if out_path.exists():
            out_path.unlink()

        # Generate plot for length <= 4 for speed
        saved = plot_mess3_ground_truth(output_path=out_path, max_length=4)
        self.assertTrue(saved.exists())
        self.assertGreater(saved.stat().st_size, 5000)

        # Cleanup
        out_path.unlink()


if __name__ == "__main__":
    unittest.main()
