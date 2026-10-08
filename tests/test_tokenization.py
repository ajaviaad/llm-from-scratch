"""Behavioral tests for byte/BPE encoding and checkpoint metadata boundaries."""
import hashlib
import json
import random
import unittest

from llm_lab.tokenization import (
    BPETokenizer, ByteTokenizer, MAX_TOKEN_BYTES, MAX_TRAINING_BYTES,
    MAX_TRAINING_DOCUMENTS, MAX_VOCABULARY, tokenizer_from_dict,
)


class ByteTokenizerTests(unittest.TestCase):
    def test_byte_values_and_unicode_round_trip(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.vocab_size, 256)
        self.assertEqual(tokenizer.encode('é'), [195, 169])
        for text in ('', 'ASCII\x00\n', '你好 🐍 café', 'e\u0301', 'é'):
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text), errors='strict'), text)
        self.assertNotEqual(tokenizer.encode('e\u0301'), tokenizer.encode('é'))

    def test_generated_invalid_utf8_has_explicit_error_policy(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.decode([255]), '\ufffd')
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([255], errors='strict')
        with self.assertRaises(ValueError):
            tokenizer.decode([255], errors='ignore')

    def test_token_ids_are_strict_and_bounded(self):
        tokenizer = ByteTokenizer()
        for ids in ([True], [False], [-1], [256], [1.0], [float('nan')], ['1'], None):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                tokenizer.decode(ids)
        self.assertEqual(tokenizer.decode(iter([65, 66])), 'AB')

    def test_text_validation_does_not_silently_coerce(self):
        for value in (None, b'bytes', 12, ['text']):
            with self.subTest(value=value), self.assertRaises(TypeError):
                ByteTokenizer().encode(value)
        with self.assertRaises(UnicodeEncodeError):
            ByteTokenizer().encode('\ud800')

    def test_byte_metadata_is_stable_and_digest_binds_kind(self):
        tokenizer = tokenizer_from_dict({'kind': 'byte', 'version': 1})
        self.assertEqual(tokenizer.to_dict(), {'kind': 'byte', 'version': 1})
        canonical = b'{"kind":"byte","version":1}'
        self.assertEqual(tokenizer.digest(), hashlib.sha256(canonical).hexdigest())
        self.assertNotEqual(tokenizer.digest(), BPETokenizer().digest())


class BPETokenizerTests(unittest.TestCase):
    def test_frequency_and_lexicographic_ties_are_deterministic(self):
        tokenizer = BPETokenizer.train(['ab', 'ac'], vocab_size=258, min_frequency=1)
        self.assertEqual(tokenizer.merges, ((97, 98), (97, 99)))
        self.assertEqual(tokenizer.encode('abac'), [256, 257])
        reverse = BPETokenizer.train(['ac', 'ab'], vocab_size=258, min_frequency=1)
        self.assertEqual(tokenizer.to_dict(), reverse.to_dict())

    def test_document_boundaries_do_not_supply_training_pairs(self):
        tokenizer = BPETokenizer.train(['a', 'b', 'a', 'b'], 320, 1)
        self.assertEqual(tokenizer.merges, ())
        self.assertEqual(tokenizer.encode('ab'), [97, 98])
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_overlapping_counts_and_left_to_right_replacement(self):
        tokenizer = BPETokenizer.train(['aaaaaa'], 260, 2)
        self.assertEqual(tokenizer.merges, ((97, 97), (256, 256)))
        self.assertEqual(tokenizer.encode('aaaaaa'), [257, 256])
        self.assertEqual(tokenizer.encode('aaaaa'), [257, 97])
        self.assertEqual(tokenizer.decode([257, 256], errors='strict'), 'aaaaaa')

    def test_training_only_data_and_unseen_unicode_are_reversible(self):
        documents = ['cat sat on mat\n', 'cat sat on mat\n']
        original = documents.copy()
        tokenizer = BPETokenizer.train(documents, 290, 2)
        self.assertEqual(documents, original)
        self.assertLess(len(tokenizer.encode(documents[0])), len(documents[0].encode()))
        for unseen in ('🧠 東京 café', '\x00\x01hello', 'e\u0301', ''):
            with self.subTest(unseen=unseen):
                encoded = tokenizer.encode(unseen)
                self.assertTrue(all(type(i) is int and 0 <= i < tokenizer.vocab_size for i in encoded))
                self.assertEqual(tokenizer.decode(encoded, errors='strict'), unseen)

    def test_empty_and_256_token_baselines_stop_cleanly(self):
        for documents in ([], [''], ['', '']):
            self.assertEqual(BPETokenizer.train(documents).vocab_size, 256)
        tokenizer = BPETokenizer.train(['banana'], vocab_size=256)
        self.assertEqual(tokenizer.encode('banana'), ByteTokenizer().encode('banana'))
        self.assertEqual(BPETokenizer.train(['ab'], min_frequency=2).merges, ())

    def test_rank_encoding_agrees_with_independent_single_merge_reference(self):
        tokenizer = BPETokenizer.train(['banana bandana', 'banana ananas'], 280, 1)
        ranks = {pair: rank for rank, pair in enumerate(tokenizer.merges)}

        def reference(text):
            tokens = list(text.encode('utf-8'))
            while True:
                candidates = [(ranks[pair], i) for i, pair in enumerate(zip(tokens, tokens[1:]))
                              if pair in ranks]
                if not candidates:
                    return tokens
                rank, index = min(candidates)
                tokens[index:index + 2] = [256 + rank]

        rng = random.Random(23)
        texts = ['banana bandana', 'ananas', 'banana ananas', '🧠']
        texts += [''.join(rng.choice('abn s') for _ in range(30)) for _ in range(20)]
        for text in texts:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.encode(text), reference(text))

    def test_json_round_trip_preserves_ranks_ids_and_digest(self):
        tokenizer = BPETokenizer.train(['hello hello world'], 280, 1)
        restored = tokenizer_from_dict(json.loads(json.dumps(tokenizer.to_dict())))
        self.assertIsInstance(restored, BPETokenizer)
        self.assertEqual(restored.merges, tokenizer.merges)
        self.assertEqual(restored.digest(), tokenizer.digest())
        self.assertEqual(restored.encode('hello 🐍'), tokenizer.encode('hello 🐍'))
        metadata = tokenizer.to_dict()
        metadata['merges'][0][0] = 0
        self.assertNotEqual(metadata, tokenizer.to_dict())
        self.assertNotEqual(tokenizer.digest(), BPETokenizer([(97, 98)]).digest())

    def test_invalid_merge_references_and_shapes_are_rejected(self):
        bad_merges = (
            [[256, 97]], [[97, 256]], [[-1, 97]], [[True, 97]], [[97.0, 98]],
            [[float('nan'), 98]], [[float('inf'), 98]], [[97]], [[97, 98, 99]],
            [[97, 98], [97, 98]], ['ab'], [(97, 98)], 'not a list',
        )
        for merges in bad_merges:
            metadata = {'kind': 'bpe', 'version': 1, 'merges': merges}
            with self.subTest(merges=merges), self.assertRaises(ValueError):
                BPETokenizer.from_dict(metadata)

    def test_metadata_expansion_and_vocabulary_are_bounded(self):
        merges = [[97, 97]]
        while 2 ** len(merges) <= MAX_TOKEN_BYTES:
            last_id = 255 + len(merges)
            merges.append([last_id, last_id])
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict({'kind': 'bpe', 'version': 1, 'merges': merges})
        with self.assertRaises(ValueError):
            BPETokenizer([[]] * (MAX_VOCABULARY - 255))

    def test_training_contract_and_limits(self):
        for size in (255, MAX_VOCABULARY + 1, True, 256.0, float('nan')):
            with self.subTest(size=size), self.assertRaises(ValueError):
                BPETokenizer.train(['ab'], vocab_size=size)
        for frequency in (0, -1, True, 2.0, float('inf')):
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                BPETokenizer.train(['ab'], min_frequency=frequency)
        for documents in ('text', ('text',), None, [''] * (MAX_TRAINING_DOCUMENTS + 1)):
            with self.subTest(documents_type=type(documents)), self.assertRaises(ValueError):
                BPETokenizer.train(documents)
        with self.assertRaises(TypeError):
            BPETokenizer.train(['text', b'bytes'])
        with self.assertRaises(ValueError):
            BPETokenizer.train(['x' * (MAX_TRAINING_BYTES + 1)])

    def test_bpe_decode_validates_ids_before_utf8(self):
        tokenizer = BPETokenizer([(97, 98)])
        self.assertEqual(tokenizer.decode([256]), 'ab')
        for ids in ([True], [257], [-1], [256.0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                tokenizer.decode(ids)
        self.assertEqual(tokenizer.decode([255]), '\ufffd')
        with self.assertRaises(UnicodeDecodeError):
            tokenizer.decode([255], errors='strict')


class MetadataBoundaryTests(unittest.TestCase):
    def test_unknown_missing_extra_and_wrong_version_fields_fail_closed(self):
        malformed = (
            None, [], {}, {'kind': True, 'version': 1},
            {'kind': 'word', 'version': 1}, {'kind': 'byte'},
            {'kind': 'byte', 'version': True}, {'kind': 'byte', 'version': 1.0},
            {'kind': 'byte', 'version': 2}, {'kind': 'byte', 'version': 1, 'extra': None},
            {'kind': 'bpe', 'version': 1},
            {'kind': 'bpe', 'version': 1, 'merges': [], 'extra': None},
            {'kind': 'bpe', 'version': float('nan'), 'merges': []},
        )
        for data in malformed:
            with self.subTest(data=data), self.assertRaises(ValueError):
                tokenizer_from_dict(data)
        with self.assertRaises(ValueError):
            ByteTokenizer.from_dict({'kind': 'bpe', 'version': 1, 'merges': []})
        with self.assertRaises(ValueError):
            BPETokenizer.from_dict({'kind': 'byte', 'version': 1})


if __name__ == '__main__':
    unittest.main()
