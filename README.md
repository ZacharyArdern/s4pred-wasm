---
license: gpl-3.0
tags:
  - protein
  - secondary-structure
  - onnx
  - biology
  - wasm
pipeline_tag: token-classification
---

# S4PRED — ONNX weights for in-browser inference

ONNX export of the [S4PRED](https://github.com/psipred/s4pred) secondary structure predictor by Moffat & Jones, for use with [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).

## Files

| File | Size | Description |
|---|---|---|
| `s4pred_int8.onnx` | 225 MB | int8 weight-quantised ensemble (recommended) |
| `s4pred_fp16.onnx` | 449 MB | fp16 ensemble |

Both files contain the full 5-model GRU ensemble. The int8 variant quantises GRU weight matrices to int8 via direct initialiser surgery, retaining float32 activations — prediction accuracy is effectively identical to fp32 (0.3% of residues differ across 100 test proteins).

## Live app

Try it in your browser (no install, no data sent to server):
**[zacharyardern.github.io/s4pred-wasm](https://zacharyardern.github.io/s4pred-wasm)**

## Original model

> Moffat, L. & Jones, D.T. (2021). Increasing the accuracy of single sequence prediction methods using a deep semi-supervised learning framework. *Bioinformatics*, 37(21), 3744–3750. https://doi.org/10.1093/bioinformatics/btab491

Source code and original weights: https://github.com/psipred/s4pred

Original weights are © the authors, released under GPL-3.0. This repo redistributes them in ONNX format under the same licence.

## Export method

```python
# Float32 export
torch.onnx.export(model, dummy, 's4pred.onnx', dynamo=False, opset_version=17,
                  dynamic_axes={'tokens': {1: 'seq_len'}, 'log_probs': {0: 'seq_len'}})

# int8: direct GRU initialiser quantisation (quantize_dynamic skips GRU nodes)
# see quantize_gru_weights.py in https://github.com/ZacharyArdern/s4pred-wasm
```
