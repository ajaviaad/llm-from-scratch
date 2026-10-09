# LLM From Scratch

**Build, train, save, resume, and generate with a language model you can understand end to end.**

LLM From Scratch is a standalone implementation of a small decoder language model
in Python and NumPy. It trains every parameter from random initialization and
makes the tokenizer, forward pass, gradients, optimizer, checkpoints, and generation
path available as readable source code.

NumPy supplies array operations. The repository implements causal attention,
RMSNorm, SwiGLU, backpropagation, Adam, byte-level BPE, and KV-cached generation
itself. No automatic-differentiation framework, pretrained weights, API keys, or
GPU are required.

Use it to learn the mechanics behind larger language models, inspect numerical
behavior, or establish a small reference before extending an implementation.
The included text is original synthetic teaching data.

## Install from this repository

### 1. Clone the repository

Copy this repository's HTTPS or SSH clone URL from its **Code** menu. Replace
`REPOSITORY_URL` below with that URL:

```sh
git clone REPOSITORY_URL llm-from-scratch
cd llm-from-scratch
```

You need Git and Python. The documented interpreter range is **Python 3.11–3.13**;
the recorded verification environment uses **Python 3.12.14** and **NumPy 2.3.5**.
Python 3.12 is a suitable starting point for reproducing those examples.

This project runs directly from its source checkout. Install its dependencies
below, then run commands from the repository root containing `llm_lab/`, `data/`,
and `requirements.txt`.

### 2. Create an environment and install dependencies

**macOS / Linux**

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt` directly, then
replace `python` with `.\.venv\Scripts\python.exe` in the commands below.
On macOS/Linux, `.venv/bin/python` also works without activation.

The requirements file pins NumPy to the documented numerical environment.
Installation may need network access to obtain it. After dependency installation,
the repository's training, evaluation, generation, and validation commands run
locally without a service connection or model download.

### 3. Train your first model

With the environment active and the terminal at the repository root:

```sh
python -m llm_lab --version
python -m llm_lab check
python -m llm_lab train --steps 120 --output results/first-model
python -m llm_lab evaluate results/first-model
python -m llm_lab generate results/first-model --prompt "The policy" --max-new-tokens 16
```

The numerical check reports the reference properties it tests. Training updates
all model parameters, saves a checkpoint, and reports validation loss before and
after optimization. Generation loads the checkpoint and produces a continuation.

Use a **new or empty output directory** for every training run. The trainer
preserves existing nonempty directories. A short run demonstrates learning and
state management; fluent or useful assistant behavior is outside this model's scale.

## What you build

| Component | Source | Main ideas |
| --- | --- | --- |
| Byte and learned BPE tokenizers | [tokenization.py](llm_lab/tokenization.py) | Reversible UTF-8 encoding, ranked merges, and tokenizer identity |
| Decoder forward pass | [model.py](llm_lab/model.py) | Embeddings, learned positions, causal multi-head attention, residual paths, RMSNorm, and SwiGLU |
| Explicit backward pass | [model.py](llm_lab/model.py) | Chain rule, softmax derivatives, repeated-token accumulation, and gradients through each block |
| Training and checkpoints | [model.py](llm_lab/model.py) | Shifted targets, document-safe sampling, Adam, clipping, and saved optimization state |
| Autoregressive generation | [model.py](llm_lab/model.py) | Greedy and temperature sampling, position offsets, and per-layer KV caches |
| Numerical reference checks | [numerics.py](llm_lab/numerics.py) | Stable softmax, masking, normalization, RoPE arithmetic, LoRA equivalence, and quantization error |
| Command-line workflow | [__main__.py](llm_lab/__main__.py) | Tokenization, training, evaluation, resume, and generation |

The trained decoder uses **learned absolute positions** and ordinary multi-head
attention. RoPE, LoRA, and quantization are separate mathematical probes; they
are not enabled features of the trained model.

## Default configuration

| Setting | Default |
| --- | ---: |
| Decoder blocks | 2 |
| Context length | 32 tokens |
| Model width | 16 |
| Attention heads | 2 |
| Feed-forward width | 32 |
| Byte vocabulary | 256 |
| Trainable parameters | 13,904 |
| Training batch size | 4 |
| Adam learning rate | 0.003 |
| Global gradient-norm clipping threshold | 1.0 |
| Numerical precision | float64 on CPU |

These defaults keep the implementation small enough to inspect and run on a
laptop. Increasing model dimensions changes memory use and runtime; treat each
larger configuration as a new experiment.

## Continue the workflow

### Inspect tokenization

```sh
python -m llm_lab tokenize --text "A café and 雨"
python -m llm_lab tokenize --tokenizer bpe --vocab-size 300 --text "The policy requires evidence."
```

The output includes token IDs, token count, vocabulary size, and decoded text.
A Unicode character, a UTF-8 byte, and a BPE token are different units. The BPE
tokenizer learns merges from training documents only; validation uses frozen merges.

### Resume a saved run

```sh
python -m llm_lab train --resume results/first-model --steps 120 --output results/continued-model
```

`--steps 120` means **120 additional updates**, so this continuation reaches step
240 after the initial 120-step run. Resume restores the architecture, tokenizer,
weights, Adam moments, learning rate, step count, and sampler state.

Use the same training and validation contents. If the first run used custom data,
pass those file paths again. Fresh-run options such as `--width`, `--seed`,
`--tokenizer`, and `--learning-rate` cannot be combined with resume.

### Train with BPE

```sh
python -m llm_lab train --tokenizer bpe --vocab-size 300 --steps 120 --output results/bpe-model
python -m llm_lab evaluate results/bpe-model
python -m llm_lab generate results/bpe-model --prompt "The policy" --max-new-tokens 12
```

The requested vocabulary is a maximum; fewer tokens may be learned if the corpus
has insufficient repeated pairs. Byte and BPE losses use different token units,
so their per-token losses and perplexities should not be directly ranked.

### Change generation behavior

```sh
python -m llm_lab generate results/first-model --prompt "The policy" --max-new-tokens 16 --temperature 0.8 --seed 23
```

Temperature `0` selects the highest-scoring token; a positive temperature samples.
The prompt length plus requested output length must fit the checkpoint's context
window. The baseline uses UTF-8 bytes as tokens. Generation produces the requested
number of tokens and does not use an end-of-sequence stopping rule.

Run `python -m llm_lab --help`, or add `--help` after a subcommand, for its options.

## Understand the saved files

Each successful training run produces:

| File | Contents |
| --- | --- |
| `training.json` | Configuration, parameter count, step history, data/tokenizer identity, and initial/final validation measurements |
| `checkpoint.json` | Configuration, tokenizer, optimization metadata, corpus digests, and random-generator state |
| `checkpoint.npz` | Model parameters and Adam moment arrays |

Keep the checkpoint's JSON and NPZ files together. They are bound by content
digests; numerical arrays load with pickle disabled. The report is useful for
comparison and investigation even though generation uses the checkpoint files.

Validation evaluates eight deterministic batches, using the same sampled windows
before and after training. It is a **sampled-window estimate**, not an exact
full-corpus perplexity calculation. Preserve a separate test split if you need an
independent final evaluation after tuning.

## Bring your own text

Create separate UTF-8 files for training and validation. Separate documents with
blank lines, and keep single newlines within a document. Every document must
contain at least `context + 1` tokens after tokenization.

For example, after creating suitable files:

```sh
python -m llm_lab train --train-file my-data/train.txt --validation-file my-data/validation.txt --context 64 --width 32 --heads 4 --layers 3 --feed-forward 64 --steps 1000 --output results/my-model
```

This configuration requires at least 65 tokens per document. Width must be
divisible by the head count. Training windows stay within document boundaries;
data is read into memory.

The trainer rejects identical input files and exact whole-document overlap
between training and validation. Construct the split before forming windows and
check related sources and near duplicates yourself. See the
[training guide](docs/training.md) for sampling behavior, BPE limits, and all options.

## Inspect the model in Python

Run this from the repository root using the configured environment:

```python
import numpy as np

from llm_lab.model import Config, TinyDecoder
from llm_lab.tokenization import ByteTokenizer

tokenizer = ByteTokenizer()
ids = tokenizer.encode("The policy")

# Each target is the next token after its corresponding input.
inputs = np.array([ids[:-1]], dtype=np.int64)
targets = np.array([ids[1:]], dtype=np.int64)

model = TinyDecoder(Config(), seed=17)
logits, cache = model.forward(inputs)
loss, gradients = model.loss_grad(inputs, targets)

assert logits.shape == (1, len(ids) - 1, 256)
assert set(gradients) == set(model.parameters)
assert np.isfinite(loss)

print("Parameter count:", sum(value.size for value in model.parameters.values()))
print("Initial next-token loss:", loss)
print("Cached positions:", cache["length"])
```

This computes a forward pass and explicit gradients for an untrained model.
The CLI training workflow applies optimizer updates and records the resulting state.

## Learning and verification

Start with the [learning paths](docs/learning-paths.md). Beginners can follow text
through tokenization, shifted targets, a decoder block, training, and generation.
Experienced readers can use the [architecture guide](docs/architecture.md) and
[experiments](docs/experiments.md) to investigate gradients, causality, cache
equivalence, and exact local continuation before changing the implementation.

```sh
python -m unittest discover -s tests -v
python scripts/validate.py --output results/validation
```

The validator runs the tests, numerical reference checks, and byte/BPE workflows
through training, evaluation, generation, and resume. Use a new or empty output
directory. Without `--output`, it removes its temporary evidence after reporting.

The [recorded release verification](docs/verification.md) reports **69 passing
tests and eight numerical checks** on Python 3.12.14, NumPy 2.3.5, and macOS ARM64.
It includes finite differences for all 420 scalar parameters in a small two-layer
test configuration and comparisons of resumed versus uninterrupted optimization.

The recorded 120-step byte run reduced sampled validation loss from **5.5820 to
2.7286 nats per token** on the included synthetic corpus. These are local baseline
observations, not a general language-quality benchmark. See
[example results](examples/README.md) for the actual reports.

GitHub Actions is configured for Python 3.11, 3.12, and 3.13 on Linux, macOS, and
Windows. The stored verification record distinguishes that configuration from
hosted jobs actually executed.

An optional file check is available with `python scripts/verify_integrity.py`.
It compares listed files against `SHA256SUMS` and reports changes after those files
are edited. The manifest is unsigned and does not authenticate a publisher.

## Repository guide

```text
llm_lab/             Tokenizers, decoder, backward pass, optimizer, and CLI
data/                Separate synthetic training and validation documents
docs/                Architecture, training, experiments, and learning guides
tests/               Numerical, state, tokenizer, and command-level tests
scripts/             Workflow validation and file-integrity checks
examples/            Recorded training results and verification evidence
.github/workflows/   CPU validation configuration
requirements.txt     Pinned numerical dependency
release.json         Recorded release metadata
SHA256SUMS           File-integrity manifest
```

## Troubleshooting

| Problem | Resolution |
| --- | --- |
| `No module named llm_lab` | Run from the cloned repository root containing `llm_lab/`. |
| `No module named numpy` | Install `requirements.txt` with the same interpreter used to run the command. |
| Output directory is not empty | Choose a new run directory; existing runs are preserved. |
| A document is too short | Provide at least `context + 1` tokens per document, or reduce context for a fresh run. |
| Context limit is exceeded | Shorten the prompt or requested generation so their combined token count fits. |
| Resume data does not match | Restore the original corpus contents and pass the original custom file paths. |
| Checkpoint digest does not match | Keep the matching metadata and tensor files together and compare with an intact copy. |
| Generated text is repetitive or incoherent | Interpret output in light of the tiny model and narrow corpus; check numerical behavior separately. |

## Scope and further reading

This is a CPU reference for learning how decoder models work. It does not provide
foundation-model-scale training, distributed execution, mixed precision, or a
production inference service. New architectures and larger workloads need their
own correctness checks and measurements. Read the [limitations](docs/limitations.md)
before extending it.

Readers of *LLM Engineering and Architecture: From Transformer Mathematics to
Reliable Production Systems* by Adeel Javaid can also use the toolkit while
studying the book. The implementation and guides are self-contained.

## License and attribution

Repository code, documentation, and original synthetic text are covered by the
[MIT License](LICENSE). See [NOTICE.md](NOTICE.md) for attribution and dependency
information, and [CONTRIBUTING.md](CONTRIBUTING.md) for development guidance.

Current version: **1.0.0**.
