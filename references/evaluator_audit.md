# Strict evaluator audit

## Findings and risk ratings

| Finding | Risk before fix | Correction |
|---|---:|---|
| Training selected the best checkpoint by pair-classification loss rather than localization error. | Medium | `train.py` now evaluates validation coordinates after classical candidate generation and learned reranking, and saves `best.pt` by validation localization error. |
| Hybrid failure analysis attempted to serialize the complete PyTorch checkpoint payload, including tensors. | High | `load_verifier()` now returns only JSON-safe checkpoint metadata; malformed or incompatible checkpoints fall back safely. |
| Hybrid reranking could displace a strong classical candidate based on a weak learned verifier. | Medium | A classical-score guardrail keeps the classical winner when the learned winner is materially weaker on classical evidence. |
| The rotation sweep did not cover the full relative pose difference from two independent ±1.5° augmentations. | Medium | The sweep now covers −3° through +3°. |
| The generator exposed only three augmentation flags despite claiming all augmentation parameters were configurable. | Medium | `generate_dataset.py` now exposes every `AugmentationConfig` field as a CLI flag. |
| Metadata conflated native-pixel resize factor with final reference-to-search scale. | High | Metadata now reports `scale_factor` and `reference_to_search_scale` as 0.1 and separately records `native_to_final_search_scale`. |
| Rotated target bounding boxes were annotated as unrotated squares. | Low/Medium | Bounding boxes are now computed from transformed target corners in search-image coordinates. |
| RGBA image loading relied on implicit OpenCV behavior. | Low | Four-channel images are explicitly converted with `COLOR_BGRA2GRAY`. |
| Malformed checkpoint behavior was not covered by tests. | Medium | Added an automated fallback test. |
| Optional PyTorch was included in the required evaluator dependency manifest. | Medium | `requirements.txt` is now classical-only; `requirements-deep.txt` contains optional PyTorch. |
| A scratch source-note file disclosed a broken research URL inside the release. | Medium | Removed from both source and release trees. |
| Tiny bundled checkpoint provenance was weak. | Medium | Re-trained the bundled checkpoint on 300 samples with a disjoint 210/45/45 ID split and validation localization selection. |

## Acceptance checks

The corrected repository uses zero-based `(x, y)` image coordinates, with `x` increasing rightward and `y` increasing downward. The generator creates a 200-pixel native reference and a 2000-pixel native search field from one continuous layout, then maps them to 1000×1000 views. The resulting target is approximately 100×100 pixels, so the final-image reference-to-search scale is exactly 0.1.

Inference loads only the two supplied image paths and an optional checkpoint relative to the repository. It does not read the training dataset or use network access. Its default standard output is exactly one line of the form `(integer_x, integer_y)`. Debug details and all errors are written to standard error. RGB, RGBA, PNG, JPEG, and TIFF inputs are handled through the image loader, subject to the 1000×1000 evaluator contract.

Training partitions sample IDs before candidate crops are generated. All candidate crops from a source sample remain in one partition, preventing correlated-crop leakage between train, validation, and test. The evaluator itself uses only the annotated truth after inference to calculate metrics and failure reports; truth coordinates are never passed into `localize.py`.

The corrected validation evidence includes seven passing tests, no-PyTorch fallback output, malformed-checkpoint fallback output, RGBA loading, corrected scale metadata, and successful packaging checks. A 20-sample comparison of the corrected 300-sample checkpoint produced 95% accuracy within 5 pixels for hybrid inference versus 90% for classical-only inference. The full 300-sample classical benchmark remains included as an honest baseline reference; its failures are not hidden.
