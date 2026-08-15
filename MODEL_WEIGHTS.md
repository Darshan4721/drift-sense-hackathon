# Model weight artifact

The optional learned-verifier checkpoint is included in two byte-identical locations:

| Path | Role |
|---|---|
| `checkpoints/best.pt` | Automatic default path used by `localize.py` when no `--checkpoint` is supplied. |
| `model_weights/drift_sense_verifier.pt` | Clearly named copy supplied for reviewer inspection and download. |

The checkpoint contains the `TinyVerifier` state used by the optional learned candidate-verification stage. It is not required for the deterministic classical method. When PyTorch or a valid checkpoint is unavailable, `localize.py` retains its evaluator-safe classical fallback rather than attempting a download or crashing.

Verify the two provided artifacts are identical:

```bash
shasum -a 256 checkpoints/best.pt model_weights/drift_sense_verifier.pt
```
