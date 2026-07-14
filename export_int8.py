#!/usr/bin/env python3
"""
Re-export S4PRED with GRU unrolled into explicit Linear (MatMul) ops,
then apply dynamic int8 quantization.

nn.GRU → ONNX GRU node (opaque, not quantizable)
nn.GRUCell in a loop → ONNX Gemm/MatMul nodes (quantizable)
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Science/Programs/s4pred"))

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = os.path.expanduser("~/Science/Programs/s4pred/weights")
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))


# ── Unrolled bidirectional GRU using GRUCell (traces to MatMul ops) ──────────
class UnrolledBiGRU(nn.Module):
    """Drop-in replacement for nn.GRU(input, hidden, num_layers, bidirectional=True).
    Copies weights from a trained nn.GRU.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        # forward and backward cells per layer
        self.fwd_cells = nn.ModuleList()
        self.bwd_cells = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size * 2
            self.fwd_cells.append(nn.GRUCell(in_size, hidden_size))
            self.bwd_cells.append(nn.GRUCell(in_size, hidden_size))

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        B, L, _ = x.shape
        out = x
        for layer in range(self.num_layers):
            fwd_cell = self.fwd_cells[layer]
            bwd_cell = self.bwd_cells[layer]
            h_f = torch.zeros(B, self.hidden_size)
            h_b = torch.zeros(B, self.hidden_size)
            fwd_outs, bwd_outs = [], []
            for t in range(L):
                h_f = fwd_cell(out[:, t, :], h_f)
                fwd_outs.append(h_f)
            for t in range(L - 1, -1, -1):
                h_b = bwd_cell(out[:, t, :], h_b)
                bwd_outs.insert(0, h_b)
            fwd_tensor = torch.stack(fwd_outs, dim=1)   # [B, L, H]
            bwd_tensor = torch.stack(bwd_outs, dim=1)   # [B, L, H]
            out = torch.cat([fwd_tensor, bwd_tensor], dim=-1)  # [B, L, 2H]
        return out

    def load_from_gru(self, gru: nn.GRU):
        """Copy weights from a trained nn.GRU into the cell layers."""
        for layer in range(self.num_layers):
            for cell, direction in [(self.fwd_cells[layer], ''), (self.bwd_cells[layer], '_reverse')]:
                # GRU weight_ih: [3*H, input]  (r, z, n gates stacked)
                wih = getattr(gru, f'weight_ih_l{layer}{direction}')
                whh = getattr(gru, f'weight_hh_l{layer}{direction}')
                bih = getattr(gru, f'bias_ih_l{layer}{direction}')
                bhh = getattr(gru, f'bias_hh_l{layer}{direction}')
                H = self.hidden_size
                # GRUCell stores: weight_ih [3H, input], weight_hh [3H, H]
                cell.weight_ih.data.copy_(wih)
                cell.weight_hh.data.copy_(whh)
                cell.bias_ih.data.copy_(bih)
                cell.bias_hh.data.copy_(bhh)


# ── Unrolled S4PRED model ─────────────────────────────────────────────────────
class S4PRED_Unrolled(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(22, 128, padding_idx=21)
        self.gru     = UnrolledBiGRU(128, 1024, 3)
        self.outlayer = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.embed(x)          # [B, L, 128]
        x = self.gru(x)            # [B, L, 2048]
        x = self.outlayer(x)       # [B, L, 3]
        x = F.log_softmax(x, dim=-1)
        return x.squeeze(0)        # [L, 3]

    def load_weights(self, path_template):
        # load a single GRU model from weights file
        from network import GRUnet
        src = GRUnet()
        state = torch.load(path_template, map_location='cpu')
        src.load_state_dict(state)
        src.eval()

        self.embed.weight.data.copy_(src.embed.weight.data)
        self.outlayer.weight.data.copy_(src.outlayer.weight.data)
        self.outlayer.bias.data.copy_(src.outlayer.bias.data)
        self.gru.load_from_gru(src.lstm)


class S4PRED_Ensemble_Unrolled(nn.Module):
    """5-model ensemble using unrolled GRUs."""
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([S4PRED_Unrolled() for _ in range(5)])

    def forward(self, x):
        outs = [m(x) for m in self.models]
        return sum(o * 0.2 for o in outs)


def load_ensemble(weights_dir):
    model = S4PRED_Ensemble_Unrolled()
    for i, sub in enumerate(model.models, 1):
        path = os.path.join(weights_dir, f'weights_{i}.pt')
        sub.load_weights(path)
        print(f'  Loaded model_{i}')
    model.eval()
    return model


def verify(model, orig_path):
    """Quick sanity check against the original ONNX model."""
    import numpy as np, onnxruntime as ort
    from utilities import aas2int
    seq = 'MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL'
    tokens = torch.tensor([aas2int(seq)])
    with torch.no_grad():
        out_pt = model(tokens).numpy()
    sess = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
    ids = np.array([aas2int(seq)], dtype=np.int64)
    out_ref = sess.run(['log_probs'], {'tokens': ids})[0]
    labels = ['C','H','E']
    ss_pt  = ''.join(labels[p] for p in out_pt.argmax(axis=1))
    ss_ref = ''.join(labels[p] for p in out_ref.argmax(axis=1))
    match = ss_pt == ss_ref
    print(f'  PyTorch unrolled: {ss_pt}')
    print(f'  ONNX reference:   {ss_ref}')
    print(f'  Match: {match}')
    return match


if __name__ == '__main__':
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print('Loading weights into unrolled model...')
    model = load_ensemble(WEIGHTS_DIR)

    onnx_path   = os.path.join(OUT_DIR, 's4pred_unrolled.onnx')
    int8_path   = os.path.join(OUT_DIR, 's4pred_int8.onnx')

    print('Verifying unrolled model against ONNX reference...')
    verify(model, os.path.join(OUT_DIR, 's4pred.onnx'))

    print('Exporting unrolled ONNX...')
    dummy = torch.randint(0, 21, (1, 100))
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['tokens'],
        output_names=['log_probs'],
        dynamic_axes={'tokens': {0: 'batch', 1: 'seq_len'}, 'log_probs': {0: 'seq_len'}},
        opset_version=17,
        dynamo=False,
    )
    print(f'  Unrolled ONNX: {os.path.getsize(onnx_path)/1e6:.1f} MB')

    print('Applying dynamic int8 quantization...')
    quantize_dynamic(onnx_path, int8_path, weight_type=QuantType.QInt8)
    print(f'  int8 ONNX: {os.path.getsize(int8_path)/1e6:.1f} MB')

    print('Verifying int8 model...')
    import onnxruntime as ort, numpy as np
    from utilities import aas2int
    seq = 'MGDIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELMDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL'
    ids = np.array([aas2int(seq)], dtype=np.int64)
    sess_ref  = ort.InferenceSession(os.path.join(OUT_DIR, 's4pred.onnx'), providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
    out_ref  = sess_ref.run(['log_probs'],  {'tokens': ids})[0]
    out_int8 = sess_int8.run(['log_probs'], {'tokens': ids})[0]
    labels = ['C','H','E']
    ss_ref  = ''.join(labels[p] for p in out_ref.argmax(axis=1))
    ss_int8 = ''.join(labels[p] for p in out_int8.argmax(axis=1))
    print(f'  fp32 ref: {ss_ref}')
    print(f'  int8:     {ss_int8}')
    print(f'  Match: {ss_ref == ss_int8}')
    print('Done.')
