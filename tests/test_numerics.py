"""Reference math checks using analytic identities and explicit edge cases."""
import math
import unittest

import numpy as np

from llm_lab.numerics import (attention, quantize_symmetric, rms_norm, rope,
                              run_checks, softmax, token_loss)


class NumericalTests(unittest.TestCase):
    def test_softmax_shift_invariance_and_masked_zero_mass(self):
        values = np.array([[1000., 1001., -np.inf], [-1001., -1000., -np.inf]])
        weights = softmax(values)
        np.testing.assert_allclose(weights, softmax(values - 900))
        np.testing.assert_allclose(weights.sum(axis=1), 1)
        np.testing.assert_array_equal(weights[:, 2], 0)

    def test_softmax_rejects_empty_nan_positive_infinity_and_all_masked(self):
        for values in [[], [float("nan"), 0], [float("inf"), 0], [-np.inf, -np.inf]]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    softmax(values)

    def test_softmax_explicit_axis(self):
        values = np.array([[1., 2.], [3., 4.]])
        np.testing.assert_allclose(softmax(values, axis=0).sum(axis=0), 1)
        np.testing.assert_allclose(softmax(values, axis=1).sum(axis=1), 1)

    def test_rms_norm_matches_definition_without_centering(self):
        x = np.array([[3., 4.]])
        expected = x * np.array([2., 0.5]) / math.sqrt(12.5 + 1e-6)
        np.testing.assert_allclose(rms_norm(x, [2., 0.5]), expected)
        self.assertGreater(rms_norm(x, [1., 1.]).mean(), 0)
        np.testing.assert_array_equal(rms_norm(np.zeros((2, 3)), np.ones(3)), 0)

    def test_rms_norm_rejects_bad_scale_and_nonfinite_epsilon(self):
        with self.assertRaises(ValueError):
            rms_norm([[1., 2.]], [1.])
        for epsilon in [0, -1, float("nan"), float("inf")]:
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    rms_norm([[1., 2.]], [1., 1.], epsilon)

    def test_rope_preserves_norms_and_relative_position_inner_product(self):
        x = np.array([[1., 2., 3., 4.], [-2., 1., 0.5, 3.]])
        first = rope(x, [2, 5])
        shifted = rope(x, [9, 12])
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.linalg.norm(x, axis=1))
        self.assertAlmostEqual(float(first[0] @ first[1]), float(shifted[0] @ shifted[1]), places=12)
        np.testing.assert_array_equal(rope(x, [0, 0]), x)

    def test_rope_rejects_odd_width_and_bad_position_count(self):
        for x, positions in [(np.ones((2, 3)), [0, 1]), (np.ones((2, 4)), [0])]:
            with self.assertRaises(ValueError):
                rope(x, positions)

    def test_attention_uniform_keys_average_values(self):
        output, weights = attention([[1., 2.]], [[3., 4.], [3., 4.]], [[2.], [8.]])
        np.testing.assert_allclose(weights, [[0.5, 0.5]])
        np.testing.assert_allclose(output, [[5.]])

    def test_attention_causal_mask_blocks_future_keys_and_values(self):
        rng = np.random.default_rng(8)
        q, k, v = [rng.normal(size=(4, 3)) for _ in range(3)]
        mask = np.tril(np.ones((4, 4), dtype=bool))
        before, weights = attention(q, k, v, mask)
        k[-1] += 100
        v[-1] -= 100
        after, _ = attention(q, k, v, mask)
        np.testing.assert_allclose(before[:-1], after[:-1])
        np.testing.assert_array_equal(weights[~mask], 0)

    def test_attention_rejects_nonboolean_wrong_shape_and_empty_rows(self):
        for mask in [[[1, 0]], [[True]], [[False, False]]]:
            with self.subTest(mask=mask):
                with self.assertRaises(ValueError):
                    attention([[1., 2.]], np.eye(2), [[2.], [3.]], mask)

    def test_masked_token_loss_uses_only_supervised_denominator(self):
        logits = np.array([[[2., 0.], [0., 2.], [100., -100.]]])
        targets = np.array([[0, 1, 1]])
        loss, count = token_loss(logits, targets, np.array([[True, True, False]]))
        self.assertEqual(count, 2)
        self.assertAlmostEqual(loss, math.log1p(math.exp(-2)), places=13)
        logits[:, 2] *= 100
        self.assertEqual(token_loss(logits, targets, np.array([[True, True, False]])), (loss, count))

    def test_token_loss_rejects_invalid_targets_and_masks(self):
        logits = np.zeros((1, 2, 3))
        for targets in [np.array([[3, 0]]), np.array([[-1, 0]]),
                        np.array([[0., 1.]]), np.array([[True, False]]), np.array([[0]])]:
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    token_loss(logits, targets)
        for mask in [np.array([[False, False]]), np.array([[1, 1]]), np.array([[True]])]:
            with self.assertRaises(ValueError):
                token_loss(logits, np.array([[0, 1]]), mask)

    def test_quantization_has_half_step_error_bound_and_finite_zero_case(self):
        x = np.linspace(-1, 1, 97)
        codes, scale, restored = quantize_symmetric(x, bits=4)
        self.assertLessEqual(float(np.max(np.abs(x - restored))), scale / 2 + 1e-12)
        self.assertLessEqual(int(np.abs(codes).max()), 7)
        zeros, zero_scale, restored_zero = quantize_symmetric(np.zeros(7))
        self.assertTrue(np.isfinite(zero_scale))
        np.testing.assert_array_equal(zeros, 0)
        np.testing.assert_array_equal(restored_zero, 0)

    def test_quantization_rejects_bad_bits_and_nonfinite_data(self):
        for bits in [True, 1, 17, 4.0]:
            with self.assertRaises(ValueError):
                quantize_symmetric([1], bits)
        for values in [[], [float("nan")], [float("inf")]]:
            with self.assertRaises(ValueError):
                quantize_symmetric(values)

    def test_reference_check_report_is_consistent(self):
        report = run_checks()
        self.assertTrue(report["passed"])
        self.assertEqual(report["total"], len(report["checks"]))
        self.assertTrue(all(item["passed"] for item in report["checks"]))


if __name__ == "__main__":
    unittest.main()
