#!/usr/bin/env python3
"""
Quantize GRU weight initializers directly in the ONNX graph.

Standard ONNX quantization skips GRU ops. This script instead:
  1. Finds all weight/recurrence initializers for GRU nodes
  2. Quantizes them to int8 (per-tensor, symmetric)
  3. Inserts DequantizeLinear nodes so the GRU still receives float32
  4. Saves a smaller model where weights are stored as int8

At inference the runtime dequantizes weights on the fly — same accuracy
as fp32, ~4x smaller weight storage.
"""
import os
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper

IN_PATH  = os.path.join(os.path.dirname(__file__), 's4pred.onnx')
OUT_PATH = os.path.join(os.path.dirname(__file__), 's4pred_int8.onnx')


def quantize_array_int8(arr: np.ndarray):
    """Symmetric per-tensor int8 quantization. Returns (int8_arr, scale_f32)."""
    amax  = np.max(np.abs(arr))
    scale = amax / 127.0
    if scale == 0:
        scale = 1.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, np.float32(scale)


def make_dequant_node(q_name, scale_name, zp_name, out_name):
    return helper.make_node(
        'DequantizeLinear',
        inputs=[q_name, scale_name, zp_name],
        outputs=[out_name],
        axis=0,
    )


def main():
    print(f'Loading {IN_PATH} ...')
    model = onnx.load(IN_PATH)

    # Build initializer lookup
    init_map = {init.name: init for init in model.graph.initializer}

    # GRU nodes have: inputs[1]=W (weight), inputs[2]=R (recurrence weight)
    # inputs[3]=B (bias, small, skip), inputs[5]=initial_h (skip)
    gru_weight_inputs = set()
    for node in model.graph.node:
        if node.op_type == 'GRU':
            if len(node.input) > 1 and node.input[1]: gru_weight_inputs.add(node.input[1])  # W
            if len(node.input) > 2 and node.input[2]: gru_weight_inputs.add(node.input[2])  # R

    print(f'Found {len(gru_weight_inputs)} GRU weight tensors to quantize')

    new_inits   = []
    new_nodes   = []
    remap       = {}   # original_name -> dequantized_name (same name, fed via DQL)

    for name in gru_weight_inputs:
        if name not in init_map:
            print(f'  WARNING: {name} not found in initializers, skipping')
            continue

        init  = init_map[name]
        arr   = numpy_helper.to_array(init)
        q_arr, scale = quantize_array_int8(arr)

        q_name     = name + '_q'
        scale_name = name + '_scale'
        zp_name    = name + '_zp'
        dq_name    = name + '_dq'

        # int8 weight initializer
        q_init = numpy_helper.from_array(q_arr, name=q_name)
        # scale (scalar float32)
        scale_init = numpy_helper.from_array(np.array([scale], dtype=np.float32), name=scale_name)
        # zero point (scalar int8 = 0, symmetric)
        zp_init = numpy_helper.from_array(np.array([0], dtype=np.int8), name=zp_name)

        new_inits.extend([q_init, scale_init, zp_init])
        new_nodes.append(make_dequant_node(q_name, scale_name, zp_name, dq_name))
        remap[name] = dq_name

        orig_mb = arr.nbytes / 1e6
        q_mb    = q_arr.nbytes / 1e6
        print(f'  {name}: {orig_mb:.1f} MB → {q_mb:.1f} MB  (scale={scale:.6f})')

    # Remove original weight initializers (replaced by int8 + DQL)
    kept_inits = [i for i in model.graph.initializer if i.name not in gru_weight_inputs]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    model.graph.initializer.extend(new_inits)

    # Prepend DequantizeLinear nodes
    original_nodes = list(model.graph.node)
    del model.graph.node[:]

    # Patch GRU nodes to use dequantized names
    for node in original_nodes:
        if node.op_type == 'GRU':
            new_inputs = list(node.input)
            for i, inp in enumerate(new_inputs):
                if inp in remap:
                    new_inputs[i] = remap[inp]
            del node.input[:]
            node.input.extend(new_inputs)

    model.graph.node.extend(new_nodes)
    model.graph.node.extend(original_nodes)

    # Add DequantizeLinear opset if needed (it's in opset 13+, already present)
    print('Saving quantized model...')
    onnx.save(model, OUT_PATH)

    orig_mb = os.path.getsize(IN_PATH)  / 1e6
    out_mb  = os.path.getsize(OUT_PATH) / 1e6
    print(f'Original: {orig_mb:.1f} MB')
    print(f'Int8:     {out_mb:.1f} MB  ({100*out_mb/orig_mb:.0f}% of original)')

    # Verify
    print('\nVerifying...')
    import sys
    sys.path.insert(0, os.path.expanduser('~/Science/Programs/s4pred'))
    from utilities import aas2int
    import onnxruntime as ort
    seq = 'MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL'
    ids = np.array([aas2int(seq)], dtype=np.int64)
    sess_ref  = ort.InferenceSession(IN_PATH,  providers=['CPUExecutionProvider'])
    sess_q    = ort.InferenceSession(OUT_PATH, providers=['CPUExecutionProvider'])
    out_ref = sess_ref.run(['log_probs'], {'tokens': ids})[0]
    out_q   = sess_q.run(  ['log_probs'], {'tokens': ids})[0]
    labels  = ['C','H','E']
    ss_ref  = ''.join(labels[p] for p in out_ref.argmax(axis=1))
    ss_q    = ''.join(labels[p] for p in out_q.argmax(axis=1))
    print(f'fp32: {ss_ref}')
    print(f'int8: {ss_q}')
    n_diff = sum(a != b for a, b in zip(ss_ref, ss_q))
    print(f'Differences: {n_diff}/{len(seq)} positions ({100*n_diff/len(seq):.1f}%)')


if __name__ == '__main__':
    main()
