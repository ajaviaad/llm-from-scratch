"""Independent derivative, causality, and KV-cache tests for the decoder."""
import copy
import unittest
import warnings

import numpy as np

from llm_lab.model import Config, TinyDecoder, generate, norm_backward, norm_forward
from llm_lab.numerics import token_loss


def small_config(**changes):
    values = dict(vocabulary=8, context=4, width=4, heads=2,
                  feed_forward=8, layers=2)
    values.update(changes)
    return Config(**values)


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyDecoder(small_config(), seed=29)
        self.tokens = np.array([[1, 2, 1], [3, 2, 4]])
        self.targets = np.array([[2, 1, 3], [2, 4, 5]])

    def test_every_parameter_matches_central_finite_difference(self):
        """Check every element, including shared embeddings and both blocks."""
        _, analytic = self.model.loss_grad(self.tokens, self.targets)
        self.assertEqual(set(analytic), set(self.model.parameters))
        epsilon = 1e-5
        for name, parameter in self.model.parameters.items():
            numerical = np.empty_like(parameter)
            original = parameter.copy()
            for index in np.ndindex(parameter.shape):
                old = parameter[index]
                try:
                    parameter[index] = old + epsilon
                    high, _ = self.model.loss_grad(self.tokens, self.targets)
                    parameter[index] = old - epsilon
                    low, _ = self.model.loss_grad(self.tokens, self.targets)
                finally:
                    parameter[index] = old
                numerical[index] = (high - low) / (2 * epsilon)
            with self.subTest(parameter=name):
                np.testing.assert_allclose(analytic[name], numerical,
                                           rtol=3e-4, atol=2e-8)
                np.testing.assert_array_equal(parameter, original)

    def test_reported_loss_matches_independent_cross_entropy(self):
        logits, _ = self.model.forward(self.tokens)
        expected, count = token_loss(logits, self.targets)
        actual, _ = self.model.loss_grad(self.tokens, self.targets)
        self.assertEqual(count, self.targets.size)
        self.assertAlmostEqual(actual, expected, places=13)

    def test_future_tokens_cannot_change_earlier_logits(self):
        original = np.array([[1, 2, 3, 4]])
        changed = np.array([[1, 2, 6, 7]])
        before, _ = self.model.forward(original)
        after, _ = self.model.forward(changed)
        np.testing.assert_allclose(before[:, :2], after[:, :2], atol=1e-13)
        self.assertGreater(float(np.max(np.abs(before[:, 2:] - after[:, 2:]))), 1e-6)

    def test_cache_matches_full_forward_for_all_chunk_partitions(self):
        tokens = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
        expected, _ = self.model.forward(tokens)
        for sizes in [(1, 1, 1, 1), (2, 2), (1, 3), (3, 1)]:
            cache, offset, pieces = None, 0, []
            for size in sizes:
                logits, cache = self.model.forward(tokens[:, offset:offset + size], cache)
                pieces.append(logits)
                offset += size
            with self.subTest(chunks=sizes):
                np.testing.assert_allclose(np.concatenate(pieces, axis=1), expected,
                                           rtol=1e-11, atol=1e-12)

    def test_cache_input_is_not_mutated_by_continuation(self):
        _, cache = self.model.forward(self.tokens[:, :1])
        original = copy.deepcopy(cache)
        self.model.forward(self.tokens[:, 1:], cache)

        def compare(left, right):
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            elif isinstance(left, dict):
                self.assertEqual(set(left), set(right))
                for key in left:
                    compare(left[key], right[key])
            elif isinstance(left, list):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    compare(a, b)
            else:
                self.assertEqual(left, right)
        compare(cache, original)

    def test_batch_duplication_preserves_mean_loss_and_gradients(self):
        loss, gradients = self.model.loss_grad(self.tokens, self.targets)
        doubled_loss, doubled = self.model.loss_grad(np.tile(self.tokens, (2, 1)),
                                                      np.tile(self.targets, (2, 1)))
        self.assertAlmostEqual(loss, doubled_loss, places=13)
        for name in gradients:
            np.testing.assert_allclose(gradients[name], doubled[name], atol=1e-12)

    def test_unused_embedding_and_position_rows_have_zero_gradient(self):
        _, gradients = self.model.loss_grad(self.tokens, self.targets)
        np.testing.assert_array_equal(gradients["embedding"][[0, 5, 6, 7]], 0)
        np.testing.assert_array_equal(gradients["position"][3:], 0)

    def test_tiny_gradient_step_decreases_loss(self):
        before, gradients = self.model.loss_grad(self.tokens, self.targets)
        for name, value in self.model.parameters.items():
            value -= 1e-3 * gradients[name]
        after, _ = self.model.loss_grad(self.tokens, self.targets)
        self.assertLess(after, before)

    def test_bad_token_shapes_types_and_values_are_rejected(self):
        for value in [np.array([1, 2]), np.empty((1, 0), int),
                      np.array([[True]]), np.array([[1.0]]),
                      np.array([[-1]]), np.array([[8]])]:
            with self.subTest(tokens=value):
                with self.assertRaises(ValueError):
                    self.model.forward(value)

    def test_bad_targets_are_rejected(self):
        for value in [self.targets[:, :1], self.targets.astype(float),
                      self.targets.astype(bool), np.full_like(self.targets, 8),
                      np.full_like(self.targets, -1)]:
            with self.subTest(targets=value):
                with self.assertRaises(ValueError):
                    self.model.loss_grad(self.tokens, value)

    def test_context_limit_applies_to_full_and_cached_forward(self):
        with self.assertRaises(ValueError):
            self.model.forward(np.ones((1, 5), dtype=int))
        _, cache = self.model.forward(np.ones((1, 3), dtype=int))
        with self.assertRaises(ValueError):
            self.model.forward(np.ones((1, 2), dtype=int), cache)
        with self.assertRaises(ValueError):
            self.model.forward(np.ones((1, 1), dtype=int), cache, retain=True)

    def test_malformed_or_nonfinite_cache_is_rejected(self):
        _, cache = self.model.forward(np.array([[1, 2]]))
        invalid = [{}, {**cache, "length": True}, {**cache, "layers": []}]
        wrong_axes = copy.deepcopy(cache)
        wrong_axes["layers"][0]["k"] = np.ones((2, 2, 2, 2))
        invalid.append(wrong_axes)
        nonfinite = copy.deepcopy(cache)
        nonfinite["layers"][0]["v"][0, 0, 0, 0] = np.nan
        invalid.append(nonfinite)
        for value in invalid:
            with self.assertRaises(ValueError):
                self.model.forward(np.array([[3]]), value)

    def test_invalid_configuration_is_rejected(self):
        for changes in [{"width": 5, "heads": 2}, {"layers": 0},
                        {"context": True}, {"vocabulary": 0},
                        {"epsilon": float("nan")}, {"epsilon": 0}]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    small_config(**changes)

    def test_rms_backward_matches_directional_derivatives(self):
        rng = np.random.default_rng(101)
        x, direction = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 4))
        scale, scale_direction = rng.normal(size=4), rng.normal(size=4)
        upstream = rng.normal(size=x.shape)
        _, tape = norm_forward(x, scale, 1e-6)
        dx, dg = norm_backward(upstream, tape)
        epsilon = 1e-5
        plus, _ = norm_forward(x + epsilon * direction, scale + epsilon * scale_direction, 1e-6)
        minus, _ = norm_forward(x - epsilon * direction, scale - epsilon * scale_direction, 1e-6)
        numerical = float(np.sum(upstream * (plus - minus)) / (2 * epsilon))
        analytic = float(np.sum(dx * direction) + np.sum(dg * scale_direction))
        self.assertAlmostEqual(numerical, analytic, places=7)

    def test_tiny_positive_temperature_matches_unique_greedy_without_warnings(self):
        model = TinyDecoder(small_config(vocabulary=256, context=8), seed=71)
        greedy = generate(model, "a", max_new_tokens=4, temperature=0)
        prefix = [ord("a")]
        for token in greedy["generated_token_ids"]:
            logits, _ = model.forward(np.array([prefix]))
            largest = np.sort(logits[0, -1])[-2:]
            self.assertGreater(float(largest[1] - largest[0]), 1e-6)
            prefix.append(token)
        with warnings.catch_warnings(), np.errstate(all="raise"):
            warnings.simplefilter("error")
            sampled = generate(model, "a", max_new_tokens=4,
                               temperature=1e-320, seed=123)
        self.assertEqual(sampled["generated_token_ids"], greedy["generated_token_ids"])
        self.assertEqual(sampled["text"], greedy["text"])


if __name__ == "__main__":
    unittest.main()
