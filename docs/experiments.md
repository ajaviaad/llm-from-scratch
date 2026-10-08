# Experiments: from a tensor calculation to a trained model

Complete the [README setup](../README.md) first. Run all commands from the repository root using that environment's Python. The [training guide](training.md) is the canonical reference for command options, data formats, and output files. The exercises below explain what to test and how to interpret it.

Keep an unmodified baseline, save each experiment in a new output directory, and change one condition at a time. Before running an experiment, write a prediction. Afterward, distinguish the result from the explanation you hoped to obtain.

## 1. Inspect tokenization before fitting anything

**Question.** What does one model token represent?

Start with these commands:

```console
python -m llm_lab tokenize --text "The policy is clear."
python -m llm_lab tokenize --text "café"
python -m llm_lab tokenize --text "The policy is clear." --tokenizer bpe --vocab-size 300
```

The output reports `token_ids`, `tokens`, `utf8_bytes`, `decoded`, and `tokenizer_sha256`. Check exact recovery in `decoded`. The BPE inspection command learns its rules from the default training file, not from the text being inspected. Repeat with punctuation and a newline through the Python tokenizer interface if your shell's quoting behavior is unfamiliar.

Record the input's character count, UTF-8 byte count, token count, and tokenizer identity. An accented character can use more than one UTF-8 byte; a frequent BPE piece can contain more than one character. The model's context limit counts tokens under its saved tokenizer, not an informal word count.

For BPE, increase the allowed merge budget on the same training corpus and compare sequence lengths on frozen validation text. Shorter sequences can reduce computation for that text but add vocabulary parameters and do not automatically improve predictive quality. Preserve the learned merge rules with the model; do not retrain the tokenizer independently at generation time.

## 2. Verify the backward pass with an independent numerical method

**Hypothesis.** The analytic gradient predicts the local loss change caused by a small parameter perturbation.

Save this complete program as `probe_gradients.py` in the repository root, then run `python probe_gradients.py`. It constructs a tiny model and checks one active coordinate from each parameter tensor. It restores every perturbed value before moving on.

```python
import numpy as np
from llm_lab.model import Config, TinyDecoder

model = TinyDecoder(Config(context=8, width=8, heads=2, feed_forward=12), seed=17)
inputs = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int64)
targets = np.array([[1, 2, 3, 4], [2, 1, 0, 4]], dtype=np.int64)
loss, analytic = model.loss_grad(inputs, targets)
print(f"Unperturbed loss: {loss:.9f}")

for epsilon in (1e-4, 1e-5, 1e-6):
    worst = 0.0
    for name, parameter in model.parameters.items():
        gradient = analytic[name]
        index = np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
        old = float(parameter[index])
        try:
            parameter[index] = old + epsilon
            high, _ = model.loss_grad(inputs, targets)
            parameter[index] = old - epsilon
            low, _ = model.loss_grad(inputs, targets)
        finally:
            parameter[index] = old
        numerical = (high - low) / (2.0 * epsilon)
        expected = float(gradient[index])
        error = abs(numerical - expected) / max(1.0, abs(numerical), abs(expected))
        worst = max(worst, error)
    print(f"epsilon={epsilon:g}, maximum scaled error={worst:.3e}")
    assert worst < 1e-6, (epsilon, worst)
```

The scale in this diagnostic is bounded below by one, so small derivatives are judged by an absolute error while larger ones use a relative scale. The assertion is a local numerical check, not a quality threshold for language modeling.

**Controlled changes.** Use repeated token IDs to exercise embedding-gradient accumulation. Change the input order without changing the tensor shape. Inspect a different coordinate in a parameter tensor. A zero gradient can be correct for an unused embedding row; it is a poor choice if your purpose is to test an active path.

If the comparison fails, first inspect the loss denominator, matrix orientation, residual-gradient addition, and feature axis of normalization. Increasing the tolerance before finding a cause hides evidence. The repository's automated tests remain the broader baseline; this small probe explains the technique.

## 3. Test causality and the KV cache directly

**Hypotheses.** A future token cannot alter earlier logits, and incremental cached decoding agrees with full-prefix computation.

Save the following as `probe_cache.py` in the repository root and run `python probe_cache.py`. No training is needed: these properties should hold even for randomly initialized weights.

```python
import numpy as np
from llm_lab.model import Config, TinyDecoder

model = TinyDecoder(Config(context=8, width=8, heads=2, feed_forward=12), seed=17)
tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
full_logits, _ = model.forward(tokens)

changed = tokens.copy()
changed[0, -1] = 99
changed_logits, _ = model.forward(changed)
causal_error = float(np.max(np.abs(full_logits[:, :-1] - changed_logits[:, :-1])))
assert np.allclose(full_logits[:, :-1], changed_logits[:, :-1], rtol=1e-10, atol=1e-10)

cache = None
pieces = []
for position in range(tokens.shape[1]):
    logits, cache = model.forward(tokens[:, position:position + 1], cache=cache)
    pieces.append(logits)
incremental_logits = np.concatenate(pieces, axis=1)
cache_error = float(np.max(np.abs(full_logits - incremental_logits)))
assert np.allclose(full_logits, incremental_logits, rtol=1e-10, atol=1e-10)
print({"causal_max_abs_error": causal_error, "cache_max_abs_error": cache_error})
```

Errors should be zero or at ordinary floating-point rounding scale under this reference. Compare logits, not only the chosen next token. Two incorrect distributions can share an argmax.

**Controlled changes.** Test a longer supported prefix and a two-example batch. Compare a prompt processed in one chunk with the same prompt processed in several chunks. Keep positions and weights unchanged. A context-overflow rejection is the supported boundary, not permission to silently reset position indices.

## 4. Train the documented small baseline

**Hypothesis.** Correctly shifted next-token supervision and gradient updates reduce loss on the generated teaching corpus.

Run a byte-tokenized baseline and evaluate its fixed checkpoint:

```console
python -m llm_lab check
python -m llm_lab train --steps 120 --output results/experiment-baseline
python -m llm_lab evaluate results/experiment-baseline --data data/validation.txt
python -m json.tool results/experiment-baseline/training.json
```

The output directory must be new or empty. It contains `training.json`, `checkpoint.json`, and `checkpoint.npz`; keep them together. Read the recorded configuration, tokenizer identity, corpus hashes, `start_step`, `end_step`, `history`, `initial_validation_nll`, and `final_validation_nll`.

The default architecture uses two blocks, residual width 16, two heads, feed-forward width 32, and a 32-token context. The default byte model has 13,904 learned parameters. Evaluation uses eight deterministic batches of four windows with seed 2026. The same selected validation windows are used before and after training. This is a reproducible sampled validation loss, not full-corpus perplexity.

Interpret three observations separately:

| Observation | What it supports | What it does not establish |
| --- | --- | --- |
| Finite loss and gradients | The sampled numerical path remained finite. | Correct derivatives or useful language behavior. |
| Lower training loss | The model fits the sampled training objective better. | Generalization outside the training corpus. |
| Lower validation loss on the same procedure | Improved prediction on the declared validation evidence. | An independent final test result after repeated tuning. |

A uniform distribution over the byte vocabulary has negative log likelihood `log(256) ≈ 5.545` per token. The initialized model need not equal this exactly. The final value depends on architecture, tokenizer, corpus, optimization, and training length; do not treat an arbitrary low number as a universal success threshold.

**Controlled changes.** Compare two training budgets using the same configuration and data. State whether you continued the same optimization trajectory or started independent runs. Those designs answer different questions. A single minibatch loss can fluctuate while longer-run learning improves.

## 5. Diagnose learning rate and clipping

**Question.** Does a faster update produce faster useful learning, or instability?

Reuse the default-rate baseline from Experiment 4, then add two controlled runs:

```console
python -m llm_lab train --steps 120 --learning-rate 0.001 --seed 17 --output results/experiment-lr-small
python -m llm_lab train --steps 120 --learning-rate 0.01 --seed 17 --output results/experiment-lr-large
```

The baseline learning rate is 0.003. These fresh runs retain the same default architecture, data, and seed. Record training/validation loss and `gradient_norm_before_clip` from each history. The comparison holds step count fixed; it is not a claim about equally tuned optimizers.

A lower rate may make steady but slow progress. A larger rate may improve initially, oscillate, or become unstable. Frequent clipping shows that the update input often exceeds the chosen norm threshold; it does not prove the underlying derivatives are correct. A finite but high loss is different from a NaN or infinity and should be diagnosed separately.

Choose the next experiment from the evidence. If both losses remain high, check supervision and optimization before increasing model capacity. If training improves while validation worsens, investigate data size, repeated patterns, and capacity. If gradients explode, inspect scaling, normalization, and the changed update size.

## 6. Compare byte and BPE training fairly

**Question.** How do representation and sequence length change the learning problem?

Compare the byte baseline with a separate BPE run:

```console
python -m llm_lab train --steps 120 --tokenizer bpe --vocab-size 300 --seed 17 --output results/experiment-bpe
python -m llm_lab evaluate results/experiment-bpe --data data/validation.txt
```

Both runs use the same underlying training and validation documents. BPE learns merges only from the training partition. Its actual vocabulary can be smaller than the requested cap if the merge conditions stop early. Read the actual `config.vocabulary` and `parameter_count` from the output; preserve each tokenizer with its own checkpoint.

A fair comparison requires naming the budget: equal optimizer steps, equal input tokens, equal input bytes, and equal elapsed time are different experiments. BPE can fit more characters inside a fixed token context, so changing tokenizer while keeping context length numerically equal also changes the accessible text span.

Do not compare raw perplexity across these tokenizers as though it used the same unit. Report token-level loss within each tokenizer, sequence lengths on a shared text set, runtime on the same machine, and generated output under fixed prompts. A common-unit likelihood comparison needs an explicitly implemented accounting procedure; do not infer it from token count alone.

## 7. Preserve training state, then continue

**Hypothesis.** A supported resume restores the intended optimization state rather than merely loading weights.

Create and continue a split run:

```console
python -m llm_lab train --steps 60 --seed 17 --output results/experiment-split-60
python -m llm_lab train --resume results/experiment-split-60 --steps 60 --output results/experiment-split-120
```

Resume restores configuration, tokenizer, optimizer, and random state. Omit fresh-run options such as `--seed`, `--width`, and `--learning-rate` on the resume command. The training and validation file contents must match the checkpoint's recorded hashes; provide the same custom paths if the first run used custom files.

Preserve the earlier checkpoint and write continued training to a new destination. Inspect the step count, optimizer moments, random state, tokenizer, and data identity in the saved metadata. Do not overwrite the baseline merely to make file names convenient.

The strongest controlled comparison divides a run into two pieces: train `a + b` steps uninterrupted, then separately train `a`, save/reload, and continue `b` under the same inputs and supported state contract. Compare the resulting parameters and losses using the repository's tolerance policy. If you only compare generated text, a changed trajectory can remain hidden behind identical greedy tokens.

Changing learning rate, corpus, tokenizer, or architecture at resume is a new experiment. Some such changes may be unsupported; follow the actual interface rather than editing metadata to force compatibility.

## 8. Generate and interpret continuations

**Question.** What changes when selection policy changes but model parameters stay fixed?

Use one saved checkpoint and the same short prompt:

```console
python -m llm_lab generate results/experiment-baseline --prompt "The policy" --max-new-tokens 16 --temperature 0
python -m llm_lab generate results/experiment-baseline --prompt "The policy" --max-new-tokens 16 --temperature 0.7 --seed 23
python -m llm_lab generate results/experiment-baseline --prompt "The policy" --max-new-tokens 16 --temperature 0.7 --seed 24
```

The prompt uses 10 byte tokens, so the requested 16-token continuation fits the default 32-token context. Preserve the prompt, output token IDs, selection settings, and decoded text. The checkpoint supplies the tokenizer; generation does not independently relearn it.

Greedy selection repeatedly chooses the highest-scoring token. Positive temperature changes sampling probabilities and can increase variety, including malformed or repetitive output. It does not add knowledge or repair training. For a byte model, a replacement character in displayed text can indicate invalid generated UTF-8; token IDs retain the underlying output evidence.

Keep prompt tokens plus requested continuation within the model's supported context. Do not compare one run that truncates input with another that uses the complete prompt without documenting the difference. Training a small corpus model should produce an inspectable learning exercise; it should not be presented to users as a source of factual answers.

## Record an experiment in six lines

Use this compact record for every controlled change:

1. **Hypothesis:** the mechanism and expected direction.
2. **Baseline:** code, data, tokenizer, configuration, seed, and checkpoint identity.
3. **Change:** one input or implementation decision.
4. **Measurement:** units, evaluation procedure, and retained output directory.
5. **Result:** the observation, including failures and unexpected behavior.
6. **Conclusion:** the supported claim, its limitation, and the next discriminating test.

Run the repository validation after source changes and before interpreting a new training result. The [limitations](limitations.md) explain why gradient correctness, learning, text quality, and production suitability require different evidence.
