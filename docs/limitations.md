# What this model teaches—and what it does not establish

This is a complete small language-model implementation for learning and experimentation. It trains its parameters from scratch on a CPU using NumPy. The forward pass, gradients, Adam updates, checkpoint state, and generation path are readable source code. That makes it useful as an engineering reference; it does not make the resulting model a general-purpose assistant.

## Valid conclusions from a successful run

A decreasing training loss shows that optimization improved next-token prediction on the sampled training objective. A validation loss measures prediction on the specified validation procedure. A finite-difference check provides evidence about local derivative correctness. A cache comparison provides evidence that two computation paths agree for the tested prefixes. A checkpoint reconstruction checks the supported saved-state contract.

These are different claims. A generated sentence that resembles the corpus does not demonstrate factual knowledge, reasoning ability, robust instruction following, or broad language understanding. A good-looking sample can also occur by chance; select models using declared evaluation criteria, not a favorite continuation.

## Scope of the implementation

| Implemented learning mechanism | Boundary that still matters |
| --- | --- |
| Small decoder transformer in NumPy | CPU reference code, not a GPU-optimized framework or efficient large-model runtime. |
| Byte tokenization and a small BPE learning path | Tokenizer quality, vocabulary size, and sequence length depend on the corpus. The implementation does not supply a commercially trained tokenizer. |
| Learned absolute positions | The context limit is a supported input boundary. Longer sequences need an explicit architectural or inference-policy change. |
| Causal multi-head attention | Dense attention stores arrays whose attention dimensions grow quadratically with sequence length during a full forward/training pass. |
| RMSNorm and SwiGLU | The reference arithmetic does not qualify mixed-precision or fused implementations. |
| Manual backward pass | The repository's checks cover specified fixtures; new layers or transformations require new derivative tests. |
| Adam and global gradient clipping | There is no implied AdamW weight decay, large-scale schedule, or distributed optimizer. |
| Synthetic teaching data | Training and validation examples are narrow and can share stylistic patterns. Results do not estimate performance on general text. |
| Checkpointing and continuation | The saved local state is not a complete distributed checkpoint, signed artifact system, or fault-tolerant training platform. |
| Cached autoregressive generation | Numerical cache equivalence is distinct from production paging, cancellation, ownership, concurrency, and cross-device handoff. |

The architecture and supported command options are documented in [architecture.md](architecture.md) and [training.md](training.md). Features absent from those interfaces should not be assumed from terminology used in a paper or book. In particular, this project does not provide foundation-model-scale pretraining, distributed collectives, mixture-of-experts training, multimodal encoders, reinforcement learning from feedback, or a deployed inference service.

## Data and evaluation

The included text exists to make a short run reproducible and understandable. It is not a benchmark corpus. A held-out file gives a useful development signal, but repeated tuning against it makes it part of the development process. If you want an independent final estimate, preserve an additional test corpus and use it only after selecting the procedure.

Split by independent document or source before producing overlapping training windows. Randomly separating windows from the same document can put nearly identical passages on both sides. Learning BPE merges from validation text is another information path that changes the experiment, even though it does not directly use target labels.

The evaluator's sampling procedure determines what the reported loss represents. It must not be presented as the exact average over every token in a corpus unless it actually traverses and weights all those tokens accordingly. Preserve the evaluation corpus, tokenizer, model identity, and evaluation settings beside comparisons.

Byte loss and BPE loss have different denominators. A larger BPE token may represent several bytes. Directly ranking token-level perplexities across these tokenizers can reverse the meaning of a comparison. Use a common text unit and a carefully defined likelihood calculation if making a cross-tokenizer claim; sequence compression alone does not prove better language modeling.

When using your own text, include only material you are entitled to use. Do not put private documents or credentials into a public training fixture or checkpoint. A small model can memorize training fragments, and removing names does not automatically remove sensitive combinations of information.

## Numerical reproducibility

Fixed seeds help reproduce initialization, sampling, and generation, but they are not a substitute for preserving the corpus, tokenizer, configuration, and environment. Floating-point reductions can differ across libraries or hardware. Compare with explicit tolerances and investigate the first divergent intermediate result.

Finite differences have two error sources: a large perturbation is a poor local approximation; a tiny perturbation loses precision through subtraction. Probe several moderate scales and avoid interpreting a single coordinate as a proof of the entire backward pass. Check attention, normalization, residual additions, embeddings with repeated IDs, and output projections as separate mechanisms.

A local checkpoint digest detects inconsistency relative to its companion metadata. If someone can replace both files, the digest does not authenticate their origin. Load only checkpoints you intend to trust. The artifact format uses numerical arrays, but valid array syntax alone does not establish that the data is safe, useful, or compatible with an unrelated implementation.

## Scaling is a new experiment

Increasing width, depth, vocabulary, batch size, or context can increase memory and runtime sharply. A successful small configuration does not establish feasible training at a larger scale. Measure the intended workload and inspect where activations, attention matrices, parameters, gradients, and optimizer moments consume memory.

Moving to PyTorch, another autodifferentiation framework, mixed precision, or accelerators can be a useful next step. First compare a small fixed forward pass, gradients, one optimizer update, cache continuation, and checkpoint resume against the NumPy reference. Then measure throughput. A fast implementation of the wrong mask or denominator is still wrong.

A model suitable for a real application needs additional evidence: representative data, appropriately separated evaluation, useful quality baselines, error analysis, privacy and security controls, infrastructure qualification, and a release process. This repository gives you the mechanics needed to begin that work; its small-model results are not a substitute for it.
