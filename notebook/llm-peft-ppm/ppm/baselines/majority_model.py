from __future__ import annotations
import torch
import torch.nn as nn


class MajorityPredictor(nn.Module):
    def __init__(self, n_classes_activity: int, majority_class_id: int, const_next_time: float, const_remaining_time: float, padding_idx: int = 0):
        super().__init__()
        self.n_classes_activity = int(n_classes_activity)
        self.padding_idx = int(padding_idx)
        self.register_buffer("majority_class_id", torch.tensor(int(majority_class_id)))
        self.register_buffer("const_next_time", torch.tensor(float(const_next_time)))
        self.register_buffer("const_remaining_time", torch.tensor(float(const_remaining_time)))
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num, attention_mask):
        B, L = attention_mask.shape
        dev = attention_mask.device
        out = {}
        logits = torch.zeros((B, L, self.n_classes_activity), device=dev)
        logits[..., int(self.majority_class_id.item())] = 8.0
        logits = logits + 0.0 * self._dummy
        out["next_activity"] = logits
        nt = torch.full((B, L), float(self.const_next_time.item()), device=dev)
        rt = torch.full((B, L), float(self.const_remaining_time.item()), device=dev)
        nt = nt + 0.0 * self._dummy
        rt = rt + 0.0 * self._dummy
        out["next_remaining_time"] = rt
        out["next_time_to_next_event"] = nt
        return out, None