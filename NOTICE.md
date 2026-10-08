# Attribution and scope

The accompanying manuscript is *LLM Engineering and Architecture: From Transformer Mathematics to Reliable Production Systems*, first edition 2026, by Adeel Javaid. The repository uses the manuscript's emphasis on explicit computation, numerical verification and reproducible state as educational context.

The implementation, documentation and synthetic text in this repository are newly written standalone material. The manuscript and its original source listings are not included. The repository's MIT license does not license the separate book or third-party software.

NumPy remains under its own license and is installed separately. The pinned version, 2.3.5, matches the tested CPU environment; it is a reproducibility choice, not a claim to be the newest release. See the [official NumPy 2.3.5 release notes](https://numpy.org/doc/stable/release/2.3.5-notes.html).

The model uses established decoder components: causal attention, learned embeddings, residual additions, RMS normalization, SwiGLU, and Adam. It is a small educational implementation, not a replica of a named commercial model or its training data.
