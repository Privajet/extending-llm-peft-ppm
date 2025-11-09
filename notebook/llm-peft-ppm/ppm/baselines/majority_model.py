from __future__ import annotations
import torch
import torch.nn as nn

class MajorityPredictor(nn.Module):
    """
    ACT: Mehrheitsklasse → Logits (B,L,C)
    NT : konstanter Wert → (B,L)
    RT : konstanter Wert → (B,L)
    """
    def __init__(self, n_classes_activity: int | None, majority_class_id: int | None,
                 const_next_time: float | None, const_remaining_time: float | None):
        super().__init__()
        self.n_classes_activity = n_classes_activity
        self.register_buffer("majority_class_id", torch.tensor(-1 if majority_class_id is None else int(majority_class_id)))
        self.register_buffer("const_next_time", torch.tensor(float("nan") if const_next_time is None else float(const_next_time)))
        self.register_buffer("const_remaining_time", torch.tensor(float("nan") if const_remaining_time is None else float(const_remaining_time)))
        self.eval()

    @torch.no_grad()
    def forward(self, x_cat, x_num, attention_mask):
        B, L = attention_mask.shape
        dev = attention_mask.device
        out = {}

        if self.n_classes_activity is not None and self.majority_class_id.item() >= 0:
            logits = torch.full((B, L, self.n_classes_activity), -1e9, device=dev)
            logits[..., int(self.majority_class_id.item())] = 1e9
            out["activity"] = logits

        if not torch.isnan(self.const_next_time):
            out["time_to_next_event"] = torch.full((B, L), float(self.const_next_time.item()), device=dev)
        if not torch.isnan(self.const_remaining_time):
            out["remaining_time"] = torch.full((B, L), float(self.const_remaining_time.item()), device=dev)

        return out, None


@torch.no_grad()
def estimate_from_train(train_loader, use_median: bool = True):
    import numpy as np
    log = train_loader.dataset.log
    num_names = list(log.targets.numerical)  # z.B. ["time_to_next_event", "remaining_time"]

    ix_nt = num_names.index("time_to_next_event") if "time_to_next_event" in num_names else None
    ix_rt = num_names.index("remaining_time") if "remaining_time" in num_names else None

    act_ids, nt_vals, rt_vals = [], [], []
    for x_cat, x_num, y_cat, y_num in train_loader:
        m = (x_cat[..., 0] != 0)
        if y_cat.numel():
            act_ids.append(y_cat[..., 0][m].view(-1).cpu())
        if ix_nt is not None:
            nt_vals.append(y_num[..., ix_nt][m].view(-1).cpu())
        if ix_rt is not None:
            rt_vals.append(y_num[..., ix_rt][m].view(-1).cpu())

    majority = None
    if act_ids:
        y = torch.cat(act_ids)
        majority = int(torch.mode(y).values.item())

    agg = np.median if use_median else np.mean
    const_nt = float(agg(torch.cat(nt_vals).numpy())) if nt_vals else None
    const_rt = float(agg(torch.cat(rt_vals).numpy())) if rt_vals else None
    return majority, const_nt, const_rt