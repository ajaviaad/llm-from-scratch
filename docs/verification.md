# Verification record

Release 1.0.0 was validated on 8 October 2026 using Python 3.12.14, NumPy 2.3.5 and macOS 26.5.2 ARM64. The recorded results below are actual local observations. They qualify this small CPU reference implementation, not a production training cluster or a large-model capability claim.

## Correctness and state

All 69 automated tests passed. They cover tokenizer round trips and malformed merge rejection; all 420 scalar derivatives in a two-layer, small-dimension decoder; repeated-token gradient accumulation; causal noninterference; full versus cached logits across chunk boundaries; Adam arithmetic and stable clipping; invalid shapes, masks and targets; training loss reduction on a constructed periodic corpus; checkpoint corruption and incompatible-state rejection; and command-level preservation of existing outputs.

The checkpoint tests compare split and uninterrupted runs. Weights, first and second optimizer moments, sampled batches, step histories and subsequent random-generator samples agree exactly under the same tested environment. BPE metadata remains bound to the model's vocabulary and tensor archive. The generation tests compare cached greedy output with full recomputation and exercise extremely small positive temperatures without overflow warnings.

Eight additional numerical reference checks pass through `python -m llm_lab check`. These cover softmax translation invariance, attention arithmetic, causal masking, RoPE norm preservation, RMS normalization, floating-point LoRA merge equivalence, a quantization-error bound, and the supervised-token loss denominator. RoPE, LoRA and quantization here are separate arithmetic probes, not active options in the trained model.

## Executed training workflows

Both tokenizers were exercised through training, saving, reloading, evaluation, eight generated tokens, and five resumed updates. The byte run uses vocabulary 256; the BPE run requests and reaches vocabulary 300. All other model dimensions use the documented defaults. Each first run performs 120 optimizer updates; resumed runs finish at step 125.

| Run | Parameters | Initial sampled validation NLL | Final sampled validation NLL |
|---|---:|---:|---:|
| Byte | 13,904 | 5.5820484602 | 2.7286137598 |
| BPE 300 | 15,312 | 5.7826806964 | 3.8815011062 |

Each row compares the same fixed validation windows before and after training within one tokenizer. Token units differ between rows, so these figures do not rank byte and BPE modeling quality. The corpus contains four training and two validation documents written for instruction. It is too small and too narrow to support a general language-quality claim.

The [example records](../examples/README.md) preserve the observed training histories and validator summary. Their source digests identify the implementation used. No pretrained model, GPU, external dataset, online API or automatic-differentiation framework was involved.

## Reproduce

After the dependency setup in [README](../README.md):

```sh
python scripts/verify_integrity.py
python scripts/validate.py --output results/my-verification
```

Use a new or empty directory. The validator retains unit-test logs, numerical-check output, both complete trained checkpoints and training reports, generation/evaluation output, resumed checkpoints, and `validation.json`. Without `--output`, it removes these temporary artifacts after reporting the result.

The archive manifest is unsigned. It checks every listed release file and intentionally fails after a listed file is edited. Additional unlisted files do not cause failure. A digest match detects content changes; it does not establish the identity of an author.

## Platform and dependency scope

The local run used the pinned NumPy installation already available in the execution environment. A clean package download was not tested. Windows and Linux command syntax is provided, but execution on those systems was not observed here.

The [CI workflow](../.github/workflows/ci.yml) configures nine hosted combinations: three operating systems with Python 3.11, 3.12 and 3.13. It installs the pinned NumPy dependency and invokes the same validator. Hosted jobs have not been executed during ZIP preparation; a configured job is not a passing job. Its action revisions are pinned to the official [checkout v7.0.0 release](https://github.com/actions/checkout/releases/tag/v7.0.0) and [setup-python v7.0.0 release](https://github.com/actions/setup-python/releases/tag/v7.0.0).

Changing dimensions, dtype, tokenizer, data, NumPy build or hardware should produce a new record. Exact local resume and numerical tolerances do not imply bitwise identity across all systems. GPU kernels, automatic differentiation, mixed precision, distributed training and large-corpus throughput are outside this release's implementation and evidence.
