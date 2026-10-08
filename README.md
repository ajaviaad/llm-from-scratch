# LLM From Scratch

Build, train, save, resume, and generate with a decoder language model whose forward pass, gradients, optimizer, and tokenizer you can read end to end.

**Python 3.11–3.13 · NumPy 2.3.5 · CPU · No pretrained weights or API keys**

This is a complete small language-model implementation, designed for learning how larger decoder models work. It trains every parameter from random initialization. NumPy supplies array operations; the repository implements attention, normalization, SwiGLU, backpropagation, Adam, byte-level BPE, and KV-cached generation itself. No automatic-differentiation or transformer framework is used.

The default model has two decoder blocks and 13,904 parameters. It is small enough for a laptop and intentionally does not claim the capabilities of a large pretrained assistant. The included corpus is original synthetic teaching text. You can run everything without the book; the accompanying explanations introduce the concepts needed to understand the code.

## Run your first model

Open a terminal in the extracted `llm-from-scratch` directory containing this README. Installation needs access to the Python package index unless NumPy is already available locally. After installation, all repository commands run offline.

### macOS and Linux

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_integrity.py
.venv/bin/python scripts/validate.py
.venv/bin/python -m llm_lab train --steps 120 --output results/first-model
.venv/bin/python -m llm_lab generate results/first-model --prompt "The policy" --max-new-tokens 16
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verify_integrity.py
.\.venv\Scripts\python.exe scripts/validate.py
.\.venv\Scripts\python.exe -m llm_lab train --steps 120 --output results/first-model
.\.venv\Scripts\python.exe -m llm_lab generate results/first-model --prompt "The policy" --max-new-tokens 16
```

Direct interpreter paths avoid activation and execution-policy changes. In the shorter commands below, `python` means this virtual environment's interpreter. Use a fresh output directory for each run; the trainer refuses to overwrite an existing nonempty directory.

A successful training run creates `training.json`, `checkpoint.json`, and `checkpoint.npz`. The training report records validation loss before and after optimization, every training step, configuration, data identity, and tokenizer identity. The checkpoint includes optimizer and random-generator state, so training can resume rather than restart with old weights.

## What you will build

| Component | Implementation | What to understand |
|---|---|---|
| UTF-8 bytes and learned BPE | [tokenization.py](llm_lab/tokenization.py) | Reversible encoding, ranked pair merges, frozen tokenizer identity |
| Decoder forward pass | [model.py](llm_lab/model.py) | Embeddings, learned positions, causal multi-head attention, residual paths, RMSNorm and SwiGLU |
| Explicit backward pass | [model.py](llm_lab/model.py) | Chain rule, softmax derivatives, embedding accumulation and gradients through every block |
| Training and checkpoints | [model.py](llm_lab/model.py) | Shifted targets, document-safe batches, Adam, clipping and reproducible continuation |
| Generation | [model.py](llm_lab/model.py) | Greedy/temperature sampling, position offsets, bounded output and KV reuse |
| Numerical probes | [numerics.py](llm_lab/numerics.py) | Stable softmax, masking, RoPE arithmetic, LoRA equivalence and quantization error |
| Complete workflow | [Command interface](llm_lab/__main__.py) | Tokenize, train, evaluate, resume and generate |

The decoder uses learned absolute positions and ordinary multi-head attention. The separate RoPE, LoRA and quantization probes are small mathematical experiments; they do not silently change the trained decoder architecture.

## Continue beyond the first run

```sh
python -m llm_lab tokenize --text "A café and 雨"
python -m llm_lab evaluate results/first-model
python -m llm_lab train --resume results/first-model --steps 120 --output results/continued-model
python -m llm_lab train --tokenizer bpe --vocab-size 300 --steps 120 --output results/bpe-model
```

Use the [training guide](docs/training.md) for your own corpus, configuration flags, output formats and troubleshooting. The [architecture guide](docs/architecture.md) follows each tensor and derivative. The [experiments](docs/experiments.md) give runnable gradient, causality and cache probes, then controlled changes to training. Choose a [learning path](docs/learning-paths.md) appropriate to your experience.

## Verified behavior

On Python 3.12.14 / NumPy 2.3.5 / macOS ARM64:

- 69 automated tests passed, including finite differences for all 420 scalar parameters in a two-layer test configuration.
- Eight independent numerical reference checks passed.
- Byte and BPE models trained, saved, evaluated, generated, and resumed successfully.
- Resumed training matched an uninterrupted run, including weights, Adam moments, and random state.

In the included 120-step byte run, sampled validation loss fell from **5.5820 to 2.7286 nats per token**. The BPE run has a different token unit, so its per-token loss must not be ranked against the byte loss. These observations concern a small synthetic corpus, not general language quality. See the [verification record](docs/verification.md) and [limitations](docs/limitations.md).

```sh
python scripts/validate.py --output results/validation
python -m unittest discover -s tests -v
```

The first command preserves diagnostic evidence; without `--output`, the validator removes its temporary results. The GitHub Actions workflow is configured for Python 3.11, 3.12 and 3.13 on Linux, macOS and Windows. Hosted jobs were not run during preparation; local verification was macOS only.

## Repository layout

```text
llm-from-scratch/
├── README.md
├── requirements.txt
├── llm_lab/             # The complete implementation
├── data/                # Separate synthetic training and validation documents
├── docs/                # Self-contained architecture, training and experiment guides
├── tests/               # Numerical, state and command-level tests
├── scripts/             # Validation and archive-integrity checks
├── examples/            # Recorded training results and validation evidence
├── .github/workflows/   # Cross-platform CPU checks
├── release.json
├── SHA256SUMS
├── CONTRIBUTING.md
├── CHANGELOG.md
├── NOTICE.md
└── LICENSE
```

This repository was developed alongside *LLM Engineering and Architecture: From Transformer Mathematics to Reliable Production Systems* by Adeel Javaid. Its implementation and guides are self-contained; the book is not required or bundled. See [NOTICE](NOTICE.md) for attribution and [LICENSE](LICENSE) for the repository's MIT license.
