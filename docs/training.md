# Train a language model from random initialization

This guide takes you from text files to a trained checkpoint and a generated continuation. Read the setup commands in [README](../README.md) first. All commands here run from the repository root; `python` means the virtual-environment interpreter. The full path `.venv/bin/python` on macOS/Linux or `.\.venv\Scripts\python.exe` on Windows works without activation.

## 1. Establish the baseline

```sh
python scripts/validate.py
python -m llm_lab check
python -m llm_lab tokenize --text "A café and 雨"
```

Validation runs the automated tests and both byte and BPE training paths. The tokenization command prints IDs and decodes them back to the input. A UTF-8 byte is not necessarily a character; the non-ASCII examples make this visible. The numerical checks report the exact properties checked rather than a language-quality score.

The default model configuration is:

| Setting | Value | Meaning |
|---|---:|---|
| Layers | 2 | Two separately parameterized decoder blocks |
| Context | 32 | Maximum positions in training or generation |
| Width | 16 | Features in the residual stream |
| Heads | 2 | Attention heads; width must divide into equal heads |
| Feed-forward width | 32 | Width of each SwiGLU expanded branch |
| Vocabulary | 256 | UTF-8 bytes; changes when BPE is selected |
| Parameters | 13,904 | All trained, including positions and normalization scales |
| Batch | 4 | Fixed four sampled document windows per update |
| Learning rate | 0.003 | Constant Adam learning rate |
| Clipping | 1.0 | Global gradient-norm threshold |
| Precision | float64 | Numerical reference calculations on CPU |

There are no biases, dropout, weight tying, weight decay, learning-rate schedule, special tokens or padded training examples in this baseline. These are deliberate design choices you can extend after verifying the reference behavior.

## 2. Train and inspect a run

```sh
python -m llm_lab train --steps 120 --output results/byte-120
```

Every run begins with random weights unless `--resume` is supplied. The initialization seed defaults to 17; the training sampler uses the next integer, 18. Each update samples a document uniformly, then a legal start within that document, for four windows. Input and target windows are shifted by one token. Windows never cross a document boundary.

The trainer evaluates the held-out documents before and after fitting. It uses eight deterministic batches with seed 2026 so those two values refer to the same windows. This is a sampled-window validation estimate. It is not a pass over every token in the validation corpus and is not a final independent test of a repeatedly tuned model.

Open these files after training:

| File | Contents |
|---|---|
| `training.json` | Configuration, parameter count, data/tokenizer digests, initial/final validation loss and all training steps |
| `checkpoint.json` | Model configuration, step, learning rate, tokenizer, corpus digests, NumPy version and sampler state |
| `checkpoint.npz` | Model arrays and both Adam moment arrays; loaded with pickle disabled |

The metadata binds the tokenizer and tensor archive to SHA-256 digests. These identify contents and detect changes; they are unsigned, so they are not proof of who produced a file. Keep the two checkpoint files together. Generation requires only those two files and the implementation; the training report remains useful evidence.

A training step's loss comes from its current random minibatch. It can rise even when average learning is improving. Compare validation under the same conditions, inspect the gradient norms, and repeat a controlled comparison across seeds before claiming an improvement. The packaged [example reports](../examples/README.md) provide one observed baseline.

## 3. Evaluate fixed weights and generate

```sh
python -m llm_lab evaluate results/byte-120
python -m llm_lab generate results/byte-120 --prompt "The policy" --max-new-tokens 16
python -m llm_lab generate results/byte-120 --prompt "The policy" --max-new-tokens 16 --temperature 0.8 --seed 23
```

Evaluation prints loss in natural-log units per token and perplexity `exp(loss)`, plus the data digest and checkpoint step. With default data it should match the final validation loss in the training report. Use `--data path/to/test.txt` for another split whose documents satisfy the length requirements below. Reserve a separate test split before tuning if you need final evaluation evidence.

Temperature zero selects the largest logit. A positive temperature samples from the scaled-logit distribution using a local seeded generator. The implementation deliberately does not add top-k, top-p, repetition penalties, beam search, or EOS stopping. It generates exactly the requested number of new tokens if the input contract is valid.

The prompt token count plus the requested output token count must fit the context. In the byte baseline, count UTF-8 bytes; in BPE, count encoded tokens. For context 32, a ten-byte prompt plus sixteen new bytes fits. A longer request fails explicitly rather than silently truncating evidence or resetting learned positions.

Generation prefills the prompt, then appends new keys and values at each layer. It never refits weights. Arbitrary generated bytes can form invalid UTF-8; the displayed text uses replacement characters while `generated_token_ids` retains the exact sequence. Nonsensical text is expected from a tiny corpus and model. It is not an implementation failure if numerical and state checks still pass.

## 4. Resume the same optimization trajectory

```sh
python -m llm_lab train --resume results/byte-120 --steps 80 --output results/byte-200
```

`--steps 80` means 80 additional updates. The new checkpoint is at step 200. Resume restores architecture, tokenizer, weights, optimizer moments, learning rate, step and sampler state. The caller must provide the same training and validation files, verified by digest; pass the custom file paths again if the original run used them.

Do not combine resume with fresh-run options such as `--width`, `--seed`, `--tokenizer` or `--learning-rate`. The CLI rejects them instead of pretending that a changed run is an exact continuation. If you intentionally want a new optimization regime, implement and document a separate fine-tuning operation rather than modifying checkpoint metadata by hand.

Exact resume tests compare split and uninterrupted runs under the same local numerical environment. Floating-point behavior can differ after changing NumPy, CPU, numerical kernels or code. A saved version number records an environment; it does not freeze every binary library inside it.

## 5. Learn and use BPE

```sh
python -m llm_lab tokenize --tokenizer bpe --vocab-size 300 --text "The policy requires evidence."
python -m llm_lab train --tokenizer bpe --vocab-size 300 --steps 120 --output results/bpe-120
python -m llm_lab generate results/bpe-120 --prompt "The policy" --max-new-tokens 12
```

BPE starts with all 256 bytes. It learns the most frequent adjacent pair, breaking ties by token-ID order, and merges nonoverlapping occurrences left to right. Training documents are never joined for pair counting. Encoding applies the stored merge ranks; unseen valid Unicode text remains representable as bytes.

The tokenizer trains only on the training file. Validation uses its frozen merges. The requested vocabulary is a maximum: training stops earlier when no pair reaches the default minimum frequency of two. Inspect the reported actual vocabulary rather than assuming it reached the requested size.

This readable BPE implementation rescans the small corpus for each merge. It accepts vocabulary targets 256–4096, at most 10,000 training documents and at most 1,000,000 UTF-8 training bytes. A merged token may contain at most 4096 bytes. These are explicit teaching-implementation limits, not recommendations for an industrial tokenizer. The byte tokenizer has no learned merge table and does not impose the BPE corpus-size limit.

Do not compare byte and BPE perplexities as if they shared a token unit. For a fair model comparison, keep tokenization fixed, or add an explicitly normalized metric such as a carefully defined byte-level likelihood accounting.

## 6. Bring your own text

Create separate UTF-8 files for training and validation. Use a blank-line-delimited text block for each document. Keep only single newlines inside a document; those single newlines are preserved. Every document must have at least `context + 1` tokens after the selected tokenizer is applied. The trainer rejects a short document rather than silently dropping it or borrowing from the next document.

Split source documents before constructing windows. Repeated passages and related documents can leak between splits even when file hashes differ. The CLI rejects identical files and exact repeated whole documents across training and validation, but it cannot detect paraphrases, near duplicates, shared authors, temporal leakage or an unsuitable scientific split. Design those boundaries for your intended use.

```sh
python -m llm_lab train --train-file my-data/train.txt --validation-file my-data/validation.txt --context 64 --width 32 --heads 4 --layers 3 --feed-forward 64 --steps 1000 --output results/my-model
```

For the command above, all documents must contain at least 65 tokens. This creates a new random model; it is not a resize of an old checkpoint. Width must be divisible by head count. Vocabulary follows the tokenizer. Increasing layers, width, context, batch size in source, or training steps changes different costs and should be tested separately.

The sampler currently gives each document equal selection probability, then samples a start uniformly. Long documents are not weighted by their token count. State this sampling policy when interpreting a custom corpus. Data is read into memory; this is not a streaming dataset loader.

Use text you have the right to process. The repository's two sample files are original synthetic teaching documents and contain no downloaded corpus or customer records.

## 7. Inspect and change the implementation

Read [architecture](architecture.md), then run the complete derivative and cache probes in [experiments](experiments.md). Change one mechanism at a time: tokenizer size, width, head count, depth, context, learning rate, data distribution, or number of updates. Save a new output directory and state the expected observation before running it.

The provided code trains all parameters, including the embedding table, learned positions, all attention projections, both SwiGLU branches, normalization scales and the output head. Replacing a learned operation with a frozen feature extractor would change the experiment. Extending to RoPE, grouped-query attention, dropout, packed sequences or mixed precision requires matching forward, backward, checkpoint and generation changes—not just one class option.

## Troubleshooting

| Symptom | Check |
|---|---|
| `No module named numpy` | Install `requirements.txt` with the same interpreter used to run the commands |
| `No module named llm_lab` | Run from the directory containing `llm_lab/` and README |
| Output directory is not empty | Choose a new run name; existing records are intentionally preserved |
| A document is too short | Measure tokens with the selected tokenizer; lower context or provide longer documents |
| Context limit exceeded | Count prompt and output tokens together; train a new model with larger positions if needed |
| Resume data mismatch | Restore the exact original train/validation contents and pass their paths |
| Checkpoint digest mismatch | Keep the original metadata/array pair together and compare with an intact copy |
| Loss becomes nonfinite | Reproduce at small dimensions, inspect data and gradients, and verify learning rate before continuing |
| Text remains poor | Evaluate the tiny model's actual scale and data; numerical correctness does not imply fluent language |

The [limitations](limitations.md) explain which parts of a larger training system would require new work and new evidence.
