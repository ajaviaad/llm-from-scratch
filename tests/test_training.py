"""Optimizer arithmetic, document boundaries, learning, and exact resume."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from llm_lab.model import (Adam, Config, TinyDecoder, evaluation_loss,
                           generate, load_checkpoint, read_documents, sample_batch,
                           save_checkpoint, train_steps)
from llm_lab.tokenization import BPETokenizer, ByteTokenizer


def tiny_model():
    return TinyDecoder(Config(vocabulary=8, context=4, width=4, heads=2,
                              feed_forward=8, layers=2), seed=71)


class TrainingTests(unittest.TestCase):
    def test_adam_first_two_steps_match_bias_corrected_scalar_arithmetic(self):
        parameters = {"w": np.array([0.5, -0.2])}
        optimizer = Adam(parameters, learning_rate=0.01)
        expected = parameters["w"].copy()
        m, v = np.zeros(2), np.zeros(2)
        for step, gradient in enumerate([np.array([2., -3.]), np.array([-1., 4.])], 1):
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient ** 2
            expected -= 0.01 * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            norm = optimizer.update(parameters, {"w": gradient}, clip=100)
            self.assertAlmostEqual(norm, float(np.linalg.norm(gradient)))
            np.testing.assert_allclose(parameters["w"], expected, rtol=1e-13, atol=1e-14)
            self.assertEqual(optimizer.step, step)

    def test_global_norm_clipping_is_shared_across_parameter_groups(self):
        parameters = {"a": np.array([0.]), "b": np.array([0.])}
        optimizer = Adam(parameters)
        norm = optimizer.update(parameters, {"a": np.array([3.]), "b": np.array([4.])}, clip=1)
        self.assertEqual(norm, 5)
        np.testing.assert_allclose(optimizer.m["a"], [0.06])
        np.testing.assert_allclose(optimizer.m["b"], [0.08])
        self.assertAlmostEqual(float(optimizer.v["a"][0] + optimizer.v["b"][0]), 0.001)

    def test_large_finite_gradients_clip_without_overflow(self):
        parameters = {"w": np.zeros(2)}
        optimizer = Adam(parameters)
        with np.errstate(over="raise", invalid="raise"):
            norm = optimizer.update(parameters, {"w": np.array([1e300, -1e300])})
        self.assertTrue(np.isfinite(norm))
        self.assertLess(parameters["w"][0], 0)
        self.assertGreater(parameters["w"][1], 0)

    def test_invalid_gradients_fail_before_optimizer_or_weights_change(self):
        for gradients in [{"w": np.array([np.nan, 0.])}, {"w": np.array([1.])},
                          {"other": np.ones(2)}]:
            parameters = {"w": np.array([0.5, -0.2])}
            optimizer = Adam(parameters)
            with self.assertRaises(ValueError):
                optimizer.update(parameters, gradients)
            np.testing.assert_array_equal(parameters["w"], [0.5, -0.2])
            np.testing.assert_array_equal(optimizer.m["w"], 0)
            np.testing.assert_array_equal(optimizer.v["w"], 0)
            self.assertEqual(optimizer.step, 0)

    def test_sample_windows_shift_targets_without_crossing_documents(self):
        documents = [np.arange(0, 20), np.arange(100, 120)]
        x, y = sample_batch(documents, np.random.default_rng(99), context=4, batch=32)
        self.assertEqual(x.shape, (32, 4))
        np.testing.assert_array_equal(y, x + 1)
        self.assertTrue(all((row < 20).all() or (row >= 100).all() for row in x))
        np.testing.assert_array_equal(x[:, 1:], y[:, :-1])

    def test_sample_batch_rejects_invalid_documents_or_budgets(self):
        rng = np.random.default_rng(1)
        for documents in [[], [np.array([1, 2, 3, 4])], [np.ones((5, 1), int)], [np.ones(5, float)]]:
            with self.assertRaises(ValueError):
                sample_batch(documents, rng, 4)
        for context, batch in [(0, 4), (4, 0), (True, 4), (4, False)]:
            with self.assertRaises(ValueError):
                sample_batch([np.arange(20)], rng, context, batch)

    def test_document_reader_keeps_blank_line_boundaries(self):
        class Encoder:
            def encode(self, text):
                return list(text.encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("first document\n\nsecond document", encoding="utf-8")
            documents = read_documents(path, context=4, tokenizer=Encoder())
            self.assertEqual(len(documents), 2)
            self.assertEqual(bytes(documents[0].tolist()), b"first document")
            self.assertEqual(bytes(documents[1].tolist()), b"second document")
            path.write_text("short\n\nx", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_documents(path, context=4, tokenizer=Encoder())

    def test_training_reduces_loss_on_a_constructed_periodic_corpus(self):
        model = tiny_model()
        documents = [np.tile(np.array([0, 1, 2, 3]), 32)]
        before = evaluation_loss(model, documents)
        history = train_steps(model, Adam(model.parameters, 0.02), documents,
                              np.random.default_rng(19), 100)
        after = evaluation_loss(model, documents)
        self.assertLess(after, before * 0.45)
        self.assertEqual(history[-1]["step"], 100)
        self.assertTrue(all(np.isfinite(row["loss"]) for row in history))

    def test_evaluation_is_repeatable_and_does_not_modify_model(self):
        model = tiny_model()
        documents = [np.arange(40) % 8]
        before = {key: value.copy() for key, value in model.parameters.items()}
        self.assertEqual(evaluation_loss(model, documents), evaluation_loss(model, documents))
        for key in before:
            np.testing.assert_array_equal(model.parameters[key], before[key])

    def test_checkpoint_resume_matches_uninterrupted_training_exactly(self):
        model = tiny_model()
        optimizer = Adam(model.parameters, 0.008)
        rng = np.random.default_rng(111)
        documents = [np.arange(40) % 8, np.arange(30)[::-1] % 8]
        train_steps(model, optimizer, documents, rng, 3)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, rng, {"train": "synthetic-test-data"})
            restored, restored_optimizer, restored_rng, metadata = load_checkpoint(directory)
            self.assertEqual(metadata["data_digests"], {"train": "synthetic-test-data"})
            original_history = train_steps(model, optimizer, documents, rng, 7)
            resumed_history = train_steps(restored, restored_optimizer, documents, restored_rng, 7)
            self.assertEqual(original_history, resumed_history)
            self.assertEqual(optimizer.step, restored_optimizer.step)
            for original, loaded in [(model.parameters, restored.parameters),
                                     (optimizer.m, restored_optimizer.m),
                                     (optimizer.v, restored_optimizer.v)]:
                for key in original:
                    np.testing.assert_array_equal(original[key], loaded[key])
            np.testing.assert_array_equal(rng.integers(1000, size=20), restored_rng.integers(1000, size=20))

    def test_checkpoint_rejects_overwrite_and_corrupted_archive(self):
        model = tiny_model()
        optimizer = Adam(model.parameters)
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, rng, {})
            with self.assertRaises(FileExistsError):
                save_checkpoint(directory, model, optimizer, rng, {})
            path = Path(directory) / "checkpoint.npz"
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_checkpoint(directory)

    def test_checkpoint_rejects_invalid_tensor_even_with_recomputed_digest(self):
        model = tiny_model()
        optimizer = Adam(model.parameters)
        rng = np.random.default_rng(7)
        for mutation in ["shape", "negative_second_moment", "missing_tensor"]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                save_checkpoint(directory, model, optimizer, rng, {})
                archive_path = Path(directory) / "checkpoint.npz"
                with np.load(archive_path, allow_pickle=False) as loaded:
                    arrays = {key: loaded[key].copy() for key in loaded.files}
                if mutation == "shape":
                    arrays["p_embedding"] = np.zeros((1, 1))
                elif mutation == "negative_second_moment":
                    arrays["v_embedding"][0, 0] = -1
                else:
                    del arrays["m_embedding"]
                np.savez_compressed(archive_path, **arrays)
                metadata_path = Path(directory) / "checkpoint.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["weights_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_checkpoint(directory)

    def test_training_rejects_invalid_step_count(self):
        model = tiny_model()
        optimizer = Adam(model.parameters)
        for steps in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                train_steps(model, optimizer, [np.arange(20) % 8], np.random.default_rng(7), steps)

    def test_greedy_generation_matches_recomputed_full_context(self):
        tokenizer = BPETokenizer([(ord("a"), ord("b"))])
        model = TinyDecoder(Config(vocabulary=tokenizer.vocab_size, context=8,
                                   width=4, heads=2, feed_forward=8, layers=2), seed=91)
        result = generate(model, "abab", max_new_tokens=4, tokenizer=tokenizer)
        prefix = tokenizer.encode("abab")
        expected = []
        for _ in range(4):
            logits, _ = model.forward(np.array([prefix + expected]))
            expected.append(int(logits[0, -1].argmax()))
        self.assertEqual(result["generated_token_ids"], expected)
        self.assertEqual(result["text"], tokenizer.decode(prefix + expected))
        self.assertEqual(result["prompt_tokens"], 2)

    def test_seeded_sampling_is_repeatable_and_generation_validates_budget(self):
        model = TinyDecoder(Config(context=8, width=4, heads=2, feed_forward=8))
        first = generate(model, "a", max_new_tokens=3, temperature=0.7, seed=123)
        second = generate(model, "a", max_new_tokens=3, temperature=0.7, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["output_tokens"], 3)
        self.assertTrue(all(0 <= token < 256 for token in first["generated_token_ids"]))
        for kwargs in [{"prompt": "", "max_new_tokens": 1},
                       {"prompt": "abcdef", "max_new_tokens": 3},
                       {"prompt": "a", "max_new_tokens": True},
                       {"prompt": "a", "max_new_tokens": 1, "temperature": -1}]:
            with self.assertRaises(ValueError):
                generate(model, **kwargs)

    def test_checkpoint_binds_tokenizer_identity_and_vocabulary(self):
        tokenizer = BPETokenizer([(ord("a"), ord("b"))])
        model = TinyDecoder(Config(vocabulary=tokenizer.vocab_size, context=4,
                                   width=4, heads=2, feed_forward=8))
        optimizer, rng = Adam(model.parameters), np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(directory, model, optimizer, rng, {}, tokenizer=tokenizer)
            restored, _, _, metadata = load_checkpoint(directory)
            self.assertEqual(metadata["tokenizer_sha256"], tokenizer.digest())
            self.assertEqual(restored.config.vocabulary, tokenizer.vocab_size)
            path = Path(directory) / "checkpoint.json"
            metadata["tokenizer"] = ByteTokenizer().to_dict()
            metadata["tokenizer_sha256"] = ByteTokenizer().digest()
            path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                load_checkpoint(directory)

    def test_document_reader_uses_bpe_tokens_before_window_sampling(self):
        tokenizer = BPETokenizer([(ord("a"), ord("b"))])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.txt"
            path.write_text("abababababab", encoding="utf-8")
            documents = read_documents(path, context=4, tokenizer=tokenizer)
            self.assertEqual(documents[0].tolist(), [256] * 6)
            x, y = sample_batch(documents, np.random.default_rng(2), 4, batch=2)
            np.testing.assert_array_equal(x, np.full((2, 4), 256))
            np.testing.assert_array_equal(y, x)


if __name__ == "__main__":
    unittest.main()
