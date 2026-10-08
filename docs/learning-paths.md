# Learn to build a language model

You can use this repository without the book. It gives you a complete small model implemented in NumPy, a training loop, saved checkpoints, evaluation, and text generation. Begin with the environment setup in the [README](../README.md). Keep the [training guide](training.md) open for commands and the [architecture guide](architecture.md) open for explanations.

No prior GPU experience is needed. You should be comfortable running Python and reading arrays. If matrix multiplication is unfamiliar, first calculate one dot product by hand and check the dimensions of a `[batch, time, width]` array.

## Beginner path: make every stage visible

| Stage | What to do | What you should be able to explain |
| --- | --- | --- |
| 1. Encode text | Use the tokenizer instructions in the training guide. Compare a short ASCII phrase with text containing an accented character. | A character, UTF-8 byte, and BPE token are different units; decoding should recover the original input. |
| 2. Form a training pair | Write five token IDs, then shift them into four inputs and four next-token targets. | What the model predicts at every position, and why a causal mask is necessary. |
| 3. Trace a forward pass | Read the [architecture](architecture.md) and inspect one small batch. | Embeddings, positions, normalization, attention, residual additions, SwiGLU, and the output vocabulary head. |
| 4. Check one derivative | Run the finite-difference probe in [experiments](experiments.md). | How a small weight perturbation changes loss and how the chain rule predicts that change. |
| 5. Train the baseline | Run the documented short CPU training command. Preserve its entire output directory. | Training loss, validation loss, the optimizer step count, and the purpose of clipping. |
| 6. Generate a continuation | Load the saved checkpoint and use a short prompt within the context limit. | Greedy selection versus sampling, and why a plausible phrase is not a factual answer. |
| 7. Reconstruct a run | Follow the checkpoint/evaluation workflow in [training](training.md). | Why weights, tokenizer, optimizer state, configuration, data identity, and random state belong together. |
| 8. Change one variable | Select one experiment below the baseline in [experiments](experiments.md). | A comparison has meaning only when its changed condition and measurement are explicit. |

For every stage, write four sentences: what the code should do, what input you used, what you observed, and what the result does not prove. That habit turns running a demonstration into understanding it.

Do not begin by increasing the model size. First reproduce a small result, inspect the loss calculation, and confirm that cached and full-prefix logits agree. Larger arrays make elementary mistakes slower to find.

## Intermediate path: learn from controlled changes

After reproducing the baseline, run a short experiment series. Keep each run in a different output directory and record the configuration beside its results.

1. Change the training duration while retaining the same architecture, corpus, tokenizer, and initialization policy. Compare both training and validation loss.
2. Change learning rate while retaining the same planned training budget. Inspect gradient norms and loss behavior; do not select solely by the lowest value on one noisy minibatch.
3. Change context length and rebuild compatible training examples. Describe how the next-token task changes and how sequence-length cost grows.
4. Compare byte and BPE tokenization after learning merges only on training text. Compare sequence lengths and vocabulary size. Report token-level loss separately because its unit has changed.
5. Keep a checkpoint fixed and vary decoding temperature. Separate changes in output variety from changes in the trained distribution.

A useful deliverable is one page per experiment: hypothesis, changed variable, unchanged conditions, observed metrics, representative outputs, interpretation, and next test. Include a failed run when it taught you something. A loss curve that becomes nonfinite is evidence to investigate, not a result to hide.

## Experienced path: qualify the implementation before extending it

Use the small model as a readable reference. The engineering challenge is to preserve its contract while changing implementation or scale.

| Investigation | Required comparison | Common mistake |
| --- | --- | --- |
| Manual gradients | Finite differences at several representative parameter coordinates and perturbation scales. | Checking only an output weight while missing attention, normalization, or embedding accumulation. |
| Causality | Change a future suffix; compare earlier logits with the prefix unchanged. | Inspecting the mask array but never testing its effect on computation. |
| Cached decoding | Compare logits from full recomputation with incremental decoding on the same prefix. | Comparing only generated token IDs or restarting position indices at zero. |
| Resume | Compare continued training with an uninterrupted reference under the supported state contract. | Restoring weights but reinitializing Adam or the data sampler. |
| Tokenizer change | Treat tokenizer identity, vocabulary dimensions, and model weights as one compatibility boundary. | Loading an old checkpoint after changing merges. |
| Larger depth or width | Rerun shape, gradient, numerical, cache, and training checks before interpreting quality. | Assuming a configuration that constructs successfully is numerically correct. |
| New data | Separate documents and evaluation roles before training or learning tokenizer rules. | Randomly splitting overlapping windows from one document across training and validation. |

Only extend architecture fields supported by the command/configuration interface. If you add a new mechanism in source, give it an independent reference test. RoPE, grouped-query attention, tied embeddings, weight decay, padding, mixed precision, and distributed execution each change a contract; none is enabled simply by renaming an existing option.

## A practical completion standard

You understand the baseline when you can tokenize and reconstruct text, derive the shifted target array, trace logits through one block, explain a gradient, reproduce a short training run, load its checkpoint, and account for the generated tokens. You are ready to extend it when you can also create a nearby failing case and explain exactly which invariant it violates.

The [limitations](limitations.md) describe the boundary between this learning implementation and a model or service with independently measured real-world capability.
