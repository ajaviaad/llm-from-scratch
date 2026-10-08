# Contributing

Start with the small default model and `python scripts/validate.py`. Preserve a runnable CPU path with NumPy as the only numerical dependency. Keep documentation useful without a separate book.

For a change to a differentiable operation, add an independent finite-difference or analytic check. For a cache or checkpoint change, compare full and cached computation or split and uninterrupted training. State shapes, normalization axes, masking semantics and supported input limits. Do not weaken a failed tolerance merely to make a test pass.

Add features as complete paths: an architectural change needs forward computation, backward computation where trained, serialization, generation behavior and tests. A new tokenizer changes token meanings and must be bound to checkpoints. A new corpus needs a declared split and sampling policy. Never replace held-out validation with a training score while preserving the old label.

Before preparing a release, run unit tests, both tokenizer training paths, evaluation, generation and resume. Check relative Markdown links, JSON artifacts and the actual extracted ZIP. Regenerate SHA256SUMS only after the release files are final. Record observed environments separately from configured CI jobs. Generated development runs belong under the ignored results/ directory.
