#!/usr/bin/env python3
"""
Export S4PRED ensemble to a single ONNX model, then quantize to int8.

The 5-model ensemble is fused into one ONNX graph so ONNX Runtime
runs a single forward pass rather than five, keeping the browser
overhead low while preserving ensemble accuracy.

Output:
  s4pred.onnx          - float32, ~430MB
  s4pred_int8.onnx     - int8 quantized, ~110MB
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Science/Programs/s4pred"))

import torch
import torch.nn as nn
from network import S4PRED

WEIGHTS_DIR = os.path.expanduser("~/Science/Programs/s4pred/weights")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH   = os.path.join(OUT_DIR, "s4pred.onnx")
FP16_PATH   = os.path.join(OUT_DIR, "s4pred_fp16.onnx")


def load_model() -> S4PRED:
    model = S4PRED()
    for i in range(1, 6):
        weight_file = os.path.join(WEIGHTS_DIR, f"weights_{i}.pt")
        state = torch.load(weight_file, map_location="cpu")
        getattr(model, f"model_{i}").load_state_dict(state)
        print(f"  Loaded model_{i}")
    model.eval()
    return model


def export_onnx(model: S4PRED, path: str):
    # Dummy input: batch=1, seq_len=100, integer token ids
    dummy = torch.randint(0, 21, (1, 100))

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["tokens"],
        output_names=["log_probs"],
        dynamic_axes={
            "tokens":    {0: "batch", 1: "seq_len"},
            "log_probs": {0: "seq_len"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Exported to {path}  ({size_mb:.1f} MB)")


def convert_fp16(src: str, dst: str):
    import onnx
    from onnxconverter_common import convert_float_to_float16
    model = onnx.load(src)
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, dst)
    size_mb = os.path.getsize(dst) / 1e6
    print(f"  fp16 model saved to {dst}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    print("Loading weights...")
    model = load_model()

    print("Exporting float32 ONNX...")
    export_onnx(model, ONNX_PATH)

    print("Converting to fp16...")
    convert_fp16(ONNX_PATH, FP16_PATH)

    print("Done.")
