"""Multi-objective knowledge distillation: frozen teacher -> (spiking) student.

L = L_task + lam_kd*L_kd + alpha*L_traj + gamma*L_attn + eta*L_sparse
Everything is computed on RUL normalised by y_scale (=125), so each term is O(0.01-1) and the
weights actually mean what they say (in cycles^2 the old MSE terms swamped everything).

  L_task   SmoothL1(student, truth)                       supervised signal
  L_kd     MSE(student, teacher)                          output (response) distillation, Hinton 2015
  L_traj   1 - cos( W v_student , h_teacher ), per step   hint / feature distillation, FitNets 2014
  L_attn   KL(teacher_attn || student_attn)               attention transfer, Zagoruyko 2017
  L_sparse relu(rate - target)^2                          energy regulariser (fewer spikes = fewer ACs)

Non-spiking students (small LSTM/GRU) have no internals, so only task + kd apply to them.
"""
import copy
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rul.metrics import all_metrics
from rul.train import pick_device, predict


class DistillationLoss(nn.Module):
    def __init__(self, teacher_dim=128, student_dim=48, lambda_kd=0.5, alpha_traj=0.2,
                 gamma_attn=0.1, eta_sparse=1.0, target_spike_rate=0.15):
        super().__init__()
        self.w = dict(kd=lambda_kd, traj=alpha_traj, attn=gamma_attn, sparse=eta_sparse)
        self.target = target_spike_rate
        self.task_loss = nn.SmoothL1Loss(beta=0.1)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, s_rul, y, t_rul=None, s_v=None, t_h=None, s_attn=None, t_attn=None, rate=None):
        z = s_rul.new_zeros(())
        l_task = self.task_loss(s_rul, y)
        l_kd = F.mse_loss(s_rul, t_rul) if (self.w["kd"] and t_rul is not None) else z
        l_traj = z
        if self.w["traj"] and s_v is not None and t_h is not None:
            l_traj = (1 - F.cosine_similarity(self.proj(s_v), t_h, dim=-1)).mean()
        l_attn = z
        if self.w["attn"] and s_attn is not None and t_attn is not None:
            l_attn = F.kl_div(torch.log(s_attn + 1e-8), t_attn + 1e-8, reduction="batchmean")
        l_sparse = z
        if self.w["sparse"] and rate is not None:
            l_sparse = F.relu(rate - self.target) ** 2
        total = (l_task + self.w["kd"] * l_kd + self.w["traj"] * l_traj
                 + self.w["attn"] * l_attn + self.w["sparse"] * l_sparse)
        return total, {"task": l_task.item(), "kd": float(l_kd.detach()), "traj": float(l_traj.detach()),
                       "attn": float(l_attn.detach()), "sparse": float(l_sparse.detach())}


def fit_distillation(student, teacher, data: Dict, epochs=40, lr=2e-3, batch=256, patience=10,
                     cap=125, lambda_kd=0.5, alpha_traj=0.2, gamma_attn=0.1, eta_sparse=1.0,
                     target_spike_rate=0.15, y_scale=125.0, device: Optional[str] = None,
                     verbose=True, select="all") -> Dict:
    """teacher=None -> pure supervised training of the student (the 'no distillation' ablation)."""
    device = device or pick_device()
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
    student.to(device)
    spiking = hasattr(student, "forward_with_internals")
    crit = DistillationLoss(
        teacher_dim=getattr(teacher, "hidden_dim", 128) if teacher is not None else 128,
        student_dim=getattr(student, "hidden", 48), lambda_kd=lambda_kd if teacher else 0.0,
        alpha_traj=alpha_traj if (teacher and spiking) else 0.0,
        gamma_attn=gamma_attn if (teacher and spiking) else 0.0,
        eta_sparse=eta_sparse if spiking else 0.0, target_spike_rate=target_spike_rate).to(device)
    opt = torch.optim.Adam(list(student.parameters()) + list(crit.proj.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.as_tensor(data["X_train"]), torch.as_tensor(data["y_train"])),
                        batch_size=batch, shuffle=True)

    best, best_state, bad = float("inf"), None, 0
    hist = {"loss": [], "val_rmse": [], "spike_rate": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        student.train()
        tot, rate_sum, nb = 0.0, 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), (yb / y_scale).to(device)
            t = (None,) * 3
            if teacher is not None:
                with torch.no_grad():
                    t = teacher.forward_with_internals(xb)
            if spiking:
                s_rul, s_v, _, s_attn, rate = student.forward_with_internals(xb)
                loss, _ = crit(s_rul, yb, t[0], s_v, t[1], s_attn, t[2], rate)
                rate_sum += rate.item()
            else:
                loss, _ = crit(student(xb), yb, t[0])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        if select == "tl":
            val_rmse = all_metrics(data["y_val_tl"], predict(student, data["X_val_tl"], device, y_scale=y_scale).clip(0, cap))["RMSE"]
        else:
            val_rmse = all_metrics(data["y_val"], predict(student, data["X_val"], device, y_scale=y_scale).clip(0, cap))["RMSE"]
        hist["loss"].append(tot / nb); hist["val_rmse"].append(val_rmse); hist["spike_rate"].append(rate_sum / nb)
        if val_rmse < best:
            best, best_state, bad = val_rmse, copy.deepcopy(student.state_dict()), 0
        else:
            bad += 1
        if verbose:
            print(f"epoch {epoch:3d} loss {tot/nb:.4f} rate {rate_sum/nb:5.1%} val {val_rmse:6.2f}{'*' if bad == 0 else ''}")
        if bad >= patience:
            break
    student.load_state_dict(best_state)

    out = {"val": all_metrics(data["y_val"], predict(student, data["X_val"], device, y_scale=y_scale).clip(0, cap)),
           "test": all_metrics(data["y_test"], predict(student, data["X_test"], device, y_scale=y_scale).clip(0, cap)),
           "history": hist, "epochs_run": epoch, "train_seconds": time.time() - t0}
    if "X_val_tl" in data:
        out["val_tl"] = all_metrics(data["y_val_tl"], predict(student, data["X_val_tl"], device, y_scale=y_scale).clip(0, cap))
    if "X_val_pseudo" in data:
        out["val_pseudo"] = all_metrics(data["y_val_pseudo"], predict(student, data["X_val_pseudo"], device, y_scale=y_scale).clip(0, cap))
    if spiking:  # firing statistics on the TEST windows, per layer
        student.eval()
        with torch.no_grad():
            student.forward_with_internals(torch.as_tensor(data["X_test"], device=device))
        out["rate1"], out["rate2"] = student.rate1.item(), student.rate2.item()
        out["test_spike_rate"] = (out["rate1"] + out["rate2"]) / 2
    return out
