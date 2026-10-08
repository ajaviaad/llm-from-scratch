# Observed examples

These files are output from the release validator on Python 3.12.14 / NumPy 2.3.5 / macOS ARM64. They are examples of the actual report format, not hand-written expected answers.

- [Byte training](byte-training.json): 120 updates from random weights on the included training file.
- [BPE training](bpe-training.json): a separate 120-update run with 300 tokens in its vocabulary.
- [Byte generation](byte-generation.json) and [BPE generation](bpe-generation.json): eight generated tokens from the corresponding saved checkpoint.
- [Validation summary](validation.json): observed tests, workflow stages, environment and source hashes.

The repository contains no pretrained tensor archive. Run the training command to create your own checkpoint. Generated text from this miniature corpus may be repetitive or incoherent; the observations demonstrate execution, not useful assistant behavior.

The two loss reports use different tokenizations. Compare improvement within each run; do not directly rank their per-token loss or perplexity. See [verification](../docs/verification.md) for scope and [training](../docs/training.md) for the exact commands and sampling policy.
