from __future__ import annotations
import math, torch
import torch.nn as nn
from collections import defaultdict, Counter

class NgramPredictor(nn.Module):
    """
    Engine-kompatibles Plug-in:
    - ACT: Logits (B,L,C) aus n-gram Häufigkeiten mit Backoff (L→1→unigram→global majority)
    - NT/RT: (B,L) mit Kontext-Mittelwerten (Backoff bis globaler Mittelwert)
    Erwartete Inputs wie in der Engine: forward(x_cat, x_num, attention_mask) -> (out, None)
    """
    def __init__(
        self,
        n: int,
        n_classes_activity: int | None,
        act_tables: dict[int, dict[tuple, Counter]] | None,
        nt_tables: dict[int, dict[tuple, tuple[float,int]]] | None,
        rt_tables: dict[int, dict[tuple, tuple[float,int]]] | None,
        global_act_majority: int | None,
        global_nt_mean: float | None,
        global_rt_mean: float | None,
        padding_idx: int = 0,
        act_feature_index: int = 0,   # x_cat[..., act_feature_index] ist die Aktivität
    ):
        super().__init__()
        self.n = int(n)
        self.padding_idx = padding_idx
        self.act_feature_index = act_feature_index

        # Metadaten
        self.n_classes_activity = n_classes_activity

        # Python-Maps (bewusst KEINE Tensor-Buffers, da sparsam und read-only)
        self.act_tables = act_tables or {}
        self.nt_tables  = nt_tables  or {}
        self.rt_tables  = rt_tables  or {}
        self.global_act_majority = global_act_majority
        self.global_nt_mean = global_nt_mean
        self.global_rt_mean = global_rt_mean

        self.eval()

    @torch.no_grad()
    def _act_logits_from_context(self, ctx_tuple: tuple) -> torch.Tensor | None:
        """Gibt Logits (C,) aus Kontext zurück oder None, wenn nicht gefunden."""
        if self.n_classes_activity is None:
            return None
        # Backoff: L = len(ctx) .. 1
        for L in range(len(ctx_tuple), 0, -1):
            tbl = self.act_tables.get(L)
            if not tbl: 
                continue
            cnts = tbl.get(ctx_tuple[-L:])
            if not cnts: 
                continue
            total = max(1, sum(cnts.values()))
            # Log-Wahrscheinlichkeiten als Logits
            logits = torch.full((self.n_classes_activity,), -1e9)
            for cls, c in cnts.items():
                p = c / total
                logits[cls] = math.log(max(p, 1e-12))
            return logits
        # Unigram/Global-Backoff
        if self.global_act_majority is not None:
            logits = torch.full((self.n_classes_activity,), -1e9)
            logits[self.global_act_majority] = 0.0
            return logits
        return None

    @torch.no_grad()
    def _num_from_context(self, ctx_tuple: tuple, tables, global_mean: float | None) -> float | None:
        for L in range(min(len(ctx_tuple), self.n-1), 0, -1):
            tbl = tables.get(L)
            if not tbl: 
                continue
            sc = tbl.get(ctx_tuple[-L:])
            if sc:
                s, c = sc
                if c > 0:
                    return s / c
        return global_mean

    @torch.no_grad()
    def forward(self, x_cat, x_num, attention_mask):
        B, L = attention_mask.shape
        device = attention_mask.device
        out = {}

        # Aktivitäten-Sequenz (IDs)
        acts = x_cat[..., self.act_feature_index]  # (B,L)

        # ACT: Logits (B,L,C)
        if self.n_classes_activity is not None:
            logits = torch.full((B, L, self.n_classes_activity), -1e9, device=device)
            for b in range(B):
                seq = acts[b].tolist()
                for t in range(L):
                    if attention_mask[b, t] == 0:
                        continue
                    # Kontext: letzte n-1 Aktivitäten vor Position t
                    # Achtung: y_cat an t ist "next activity" für Position t
                    start = max(0, t - (self.n - 1))
                    prefix = [a for a in seq[start:t] if a != 0]
                    if not prefix:
                        ctx = ()
                    else:
                        ctx = tuple(prefix)
                    lg = self._act_logits_from_context(ctx)
                    if lg is not None:
                        logits[b, t] = lg.to(device)
            out["activity"] = logits

        # NT: (B,L)
        if self.nt_tables is not None and len(self.nt_tables):
            nt_pred = torch.empty((B, L), device=device)
            for b in range(B):
                seq = acts[b].tolist()
                for t in range(L):
                    if attention_mask[b, t] == 0:
                        nt_pred[b, t] = 0.0
                        continue
                    start = max(0, t - (self.n - 1))
                    prefix = [a for a in seq[start:t] if a != 0]
                    ctx = tuple(prefix)
                    val = self._num_from_context(ctx, self.nt_tables, self.global_nt_mean)
                    nt_pred[b, t] = float(val if val is not None else 0.0)
            out["next_time"] = nt_pred

        # RT: (B,L)
        if self.rt_tables is not None and len(self.rt_tables):
            rt_pred = torch.empty((B, L), device=device)
            for b in range(B):
                seq = acts[b].tolist()
                for t in range(L):
                    if attention_mask[b, t] == 0:
                        rt_pred[b, t] = 0.0
                        continue
                    start = max(0, t - (self.n - 1))
                    prefix = [a for a in seq[start:t] if a != 0]
                    ctx = tuple(prefix)
                    val = self._num_from_context(ctx, self.rt_tables, self.global_rt_mean)
                    rt_pred[b, t] = float(val if val is not None else 0.0)
            out["remaining_time"] = rt_pred

        return out, None


@torch.no_grad()
def build_ngram_tables_from_train(
    train_loader,
    n: int,
    need_act: bool,
    need_nt: bool,
    need_rt: bool,
    act_feature_index: int = 0,
):
    """
    Baut n-gram Tabellen aus TRAIN:
      - ACT: levels[L][ctx] -> Counter(class_id)
      - NT/RT: levels[L][ctx] -> (sum, count)
    Maskierung wie in der Engine (x_cat[...,0]!=0). 
    """
    act_levels = {L: defaultdict(Counter) for L in range(1, n)} if need_act else None
    nt_levels  = {L: defaultdict(lambda: (0.0, 0)) for L in range(1, n)} if need_nt else None
    rt_levels  = {L: defaultdict(lambda: (0.0, 0)) for L in range(1, n)} if need_rt else None

    # globale Backoffs
    global_act_counter = Counter()
    nt_sum = nt_cnt = 0.0, 0
    rt_sum = rt_cnt = 0.0, 0

    for x_cat, x_num, y_cat, y_num in train_loader:
        mask = (x_cat[..., 0] != 0)  # gleiche Definition wie Engine
        acts = x_cat[..., act_feature_index]  # (B,L)

        B, L = acts.shape
        for b in range(B):
            for t in range(L):
                if not mask[b, t]:
                    continue
                # Kontext und Targets an Position t
                start = max(0, t - (n - 1))
                prefix = [int(a) for a in acts[b, start:t] if a != 0]
                ctx = tuple(prefix)

                if need_act:
                    cls_id = int(y_cat[b, t, 0])  # 1. kateg. Target = activity
                    # Levels updatet man für ALLE L<=len(ctx)
                    for Lctx in range(1, min(len(ctx), n - 1) + 1):
                        subctx = ctx[-Lctx:]
                        ctr = act_levels[Lctx][subctx]
                        ctr[cls_id] += 1
                    global_act_counter.update([cls_id])

                if need_nt:
                    val = float(y_num[b, t, 0])
                    for Lctx in range(1, min(len(ctx), n - 1) + 1):
                        subctx = ctx[-Lctx:]
                        s, c = nt_levels[Lctx][subctx]
                        nt_levels[Lctx][subctx] = (s + val, c + 1)
                    nt_sum, nt_cnt = (nt_sum[0] + val, nt_sum[1] + 1) if isinstance(nt_sum, tuple) else (nt_sum + val, nt_cnt + 1)

                if need_rt:
                    # wenn NT und RT beide aktiv sind, liegt RT im Index 1
                    val = float(y_num[b, t, -1])
                    for Lctx in range(1, min(len(ctx), n - 1) + 1):
                        subctx = ctx[-Lctx:]
                        s, c = rt_levels[Lctx][subctx]
                        rt_levels[Lctx][subctx] = (s + val, c + 1)
                    rt_sum, rt_cnt = (rt_sum[0] + val, rt_sum[1] + 1) if isinstance(rt_sum, tuple) else (rt_sum + val, rt_cnt + 1)

    global_act_majority = max(global_act_counter.items(), key=lambda kv: kv[1])[0] if need_act and global_act_counter else None
    global_nt_mean = (nt_sum[0] / max(1, nt_sum[1])) if (need_nt and isinstance(nt_sum, tuple) and nt_sum[1] > 0) else None
    global_rt_mean = (rt_sum[0] / max(1, rt_sum[1])) if (need_rt and isinstance(rt_sum, tuple) and rt_sum[1] > 0) else None

    return act_levels, nt_levels, rt_levels, global_act_majority, global_nt_mean, global_rt_mean