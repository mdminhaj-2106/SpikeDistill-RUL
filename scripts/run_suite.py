"""Resumable experiment suite. One JSON per job in results/suite/. Usage:
    python scripts/run_suite.py phase1|phase2 [--workers 8] [--only substring]
Phase 1 trains baselines + teachers (saves teacher checkpoints); phase 2 distils them.
Every job: its own prepare(seed) -> engine split, scalers, selector all fit on that seed's training engines only.
"""
import os
for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # must precede numpy/torch import, else every worker spawns its own BLAS thread pool
import argparse, json, sys, time, traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "suite"
CK = OUT / "ckpt"
SUBS = ["FD001", "FD002", "FD003", "FD004"]
SEEDS = [0, 1, 2]
MULTI = {"FD002", "FD004"}                     # 6 operating regimes
MAMBA = {"mamba", "bimamba", "bimambateacher"}  # slow python scan -> fewer epochs
TEACHERS = {"teacher", "transformerteacher", "bimambateacher"}
LR = {"deepnn": 1e-3}


def J(**k):
    k["id"] = "__".join(str(k[x]) for x in ("kind", "subset", "model", "tag", "seed") if x in k and k[x] != "")
    return k


def phase1():
    jobs = []
    for sub in SUBS:
        for seed in SEEDS:
            for m in ["widenn", "deepnn", "rnn", "gru", "lstm", "bilstm", "cnnlstm", "tcn", "transformer",
                      "teacher", "transformerteacher"]:
                jobs.append(J(kind="base", subset=sub, model=m, tag="", seed=seed, epochs=30, patience=7))
            for m in sorted(MAMBA):
                if sub == "FD001":  # ponytail: Mamba python scan is ~10x slower, FD001 only; add other subsets with a CUDA scan
                    jobs.append(J(kind="base", subset=sub, model=m, tag="", seed=seed, epochs=12, patience=5))
            # feature-engineering ablation on the LSTM (the "fair" reference model)
            jobs.append(J(kind="base", subset=sub, model="lstm", tag="ema0.3", seed=seed, epochs=30, patience=7, smooth=0.3))
            if sub in MULTI:
                jobs.append(J(kind="base", subset=sub, model="lstm", tag="noregime", seed=seed, epochs=30, patience=7, regime=False))
                jobs.append(J(kind="base", subset=sub, model="lstm", tag="ema0.3_noregime", seed=seed, epochs=30, patience=7, smooth=0.3, regime=False))
    return jobs


def D(sub, seed, teacher, student, tag="", skw=None, **loss):
    return J(kind="distill", subset=sub, model=f"{teacher or 'none'}->{student}", tag=tag, seed=seed, teacher=teacher,
             student=student, skw=skw or {}, loss=loss, epochs=30, patience=7)


def phase2():
    jobs = []
    for seed in SEEDS:
        for sub in SUBS:          # headline: ANN teacher -> students, and the no-teacher controls
            for st in ["spike", "smalllstm", "smallgru"]:
                jobs.append(D(sub, seed, "teacher", st))
                jobs.append(D(sub, seed, None, st))
        s = "FD001"
        for t in ["transformerteacher", "bimambateacher"]:     # teacher x student grid
            for st in ["spike", "smalllstm", "smallgru"]:
                jobs.append(D(s, seed, t, st))
        # loss-term ablation (spiking student, attention-BiLSTM teacher)
        for tag, kw in {"task+kd": dict(alpha_traj=0, gamma_attn=0, eta_sparse=0),
                        "task+kd+traj": dict(gamma_attn=0, eta_sparse=0),
                        "task+kd+attn": dict(alpha_traj=0, eta_sparse=0),
                        "task+kd+sparse": dict(alpha_traj=0, gamma_attn=0),
                        "full-kd": dict(lambda_kd=0), "full-traj": dict(alpha_traj=0),
                        "full-attn": dict(gamma_attn=0), "full-sparse": dict(eta_sparse=0)}.items():
            jobs.append(D(s, seed, "teacher", "spike", tag=tag, **kw))
        for r in [0.03, 0.06, 0.10, 0.25, 0.40]:             # sparsity target sweep
            jobs.append(D(s, seed, "teacher", "spike", tag=f"rate{r}", target_spike_rate=r))
        for tag, skw in {"atan": dict(surrogate="atan"), "triangle": dict(surrogate="triangle"),
                         "fixed_beta": dict(learn_beta=False), "fixed_vth": dict(learn_vth=False),
                         "feedforward": dict(recurrent=False), "h16": dict(hidden=16), "h32": dict(hidden=32),
                         "h96": dict(hidden=96)}.items():
            jobs.append(D(s, seed, "teacher", "spike", tag=tag, skw=skw))
    return jobs


def phase3():
    """Follow-ups: (a) longer training for models that hit the 30-epoch cap, (b) cheap mean-pool readout."""
    jobs = []
    for seed in SEEDS:
        for sub in ["FD001", "FD004"]:
            for m in ["widenn", "deepnn", "rnn", "gru", "lstm", "tcn", "teacher"]:
                jobs.append(J(kind="base", subset=sub, model=m, tag="long", seed=seed, epochs=100, patience=15))
            for t in [None, "teacher"]:
                for st in ["spike", "smallgru"]:
                    j = D(sub, seed, t, st, tag="long"); j.update(epochs=100, patience=15, id=j["id"])
                    jobs.append(j)
        for m in ["mamba", "bimamba"]:   # ponytail: FD001 only, python scan is slow
            jobs.append(J(kind="base", subset="FD001", model=m, tag="long", seed=seed, epochs=40, patience=8))
        for sub in SUBS:
            for t in [None, "teacher"]:
                jobs.append(D(sub, seed, t, "spike", tag="meanpool", skw=dict(readout="mean"), gamma_attn=0))
    return jobs


def phase4():
    """Beat-the-baseline study: window length (time steps), hybrid architectures, Mamba on every subset. MPS for conv/attention/scan."""
    MPS = {"tcn", "tcnattn", "teacher", "mamba", "bimamba", "bimambateacher"}
    dil = {15: (1, 2, 4), 30: (1, 2, 4, 8), 50: (1, 2, 4, 8, 16), 70: (1, 2, 4, 8, 16, 32)}  # receptive field >= window
    jobs = []
    for seed in SEEDS:
        for sub in ["FD001", "FD004"]:                      # (a) window / time-step study
            for w in (15, 50, 70):
                for m in ["lstm", "gru", "tcn", "transformer", "teacher"]:
                    kw = dict(dilations=dil[w]) if m == "tcn" else dict(window=w) if m == "transformer" else {}
                    jobs.append(J(kind="base", subset=sub, model=m, tag=f"w{w}", seed=seed, epochs=30, patience=7, window=w,
                                  mkw=kw, device="mps" if m in MPS else "cpu"))
                j = D(sub, seed, None, "spike", tag=f"w{w}"); j["window"] = w; jobs.append(j)
        for sub in SUBS:                                    # (b) hybrid architectures
            for w in (30, 50):
                for m in ["tcnattn", "cnnbigruattn", "convtransformer"]:
                    kw = dict(dilations=dil[w]) if m == "tcnattn" else {}
                    jobs.append(J(kind="base", subset=sub, model=m, tag="" if w == 30 else f"w{w}", seed=seed, epochs=30, patience=7,
                                  window=w, mkw=kw, device="mps" if m in MPS else "cpu"))
        for sub in ["FD002", "FD003", "FD004"]:             # (c) Mamba everywhere, now affordable on MPS
            for m in ["mamba", "bimamba", "bimambateacher"]:
                jobs.append(J(kind="base", subset=sub, model=m, tag="long", seed=seed, epochs=40, patience=8, device="mps"))
    return jobs


EXTRA = [3, 4, 5, 6, 7]  # five more seeds for the headline comparison (8 in total)


def wstar():
    """Window length chosen on validation data by scripts/choose_window.py (30 if not run yet)."""
    f = ROOT / "results" / "window_choice.json"
    return int(json.loads(f.read_text())["window"]) if f.exists() else 30


def ckname(model, sub, seed, W, ck_tag=None):
    if ck_tag:                                   # protocol v2 teachers: independent of the legacy window-tagged files
        return f"{model}_{sub}_s{seed}_{ck_tag}.pt"
    return f"{model}_{sub}_s{seed}.pt" if W == 30 else f"{model}_{sub}_s{seed}_w{W}.pt"


def phase5a():
    """Teachers for the extra seeds (checkpoints are needed before phase5b). MPS."""
    W = wstar()
    if W == 30:   # seeds 0-2 already exist from phase 1
        return [J(kind="base", subset=s, model="teacher", tag="", seed=sd, epochs=30, patience=7, device="mps") for sd in EXTRA for s in SUBS]
    return [J(kind="base", subset=s, model="teacher", tag=f"w{W}t", seed=sd, epochs=30, patience=7, window=W, save_ckpt=True, device="mps")
            for sd in range(8) for s in SUBS]


def phase5b():
    """Teacher vs no-teacher for the spiking student and the small GRU on the extra seeds."""
    jobs = []
    W = wstar(); htag = "h" if W == 30 else f"h{W}"
    for sd in range(8):                       # seeds 0-7; tag "h" = headline run with stored predictions
        for s in SUBS:
            for st in ("spike", "smallgru", "smalllstm"):
                for t in (None, "teacher"):
                    j = D(s, sd, t, st, tag=htag); j["window"] = W; jobs.append(j)
    return jobs


def phase4c():
    """The remaining plain baselines at the chosen 50-cycle window so every model is compared like for like."""
    jobs = []
    for seed in SEEDS:
        for sub in SUBS:
            for m in ["widenn", "deepnn", "rnn", "bilstm", "cnnlstm"]:
                kw = dict(window=50) if m in ("widenn", "deepnn") else {}
                jobs.append(J(kind="base", subset=sub, model=m, tag="w50", seed=seed, epochs=30, patience=7, window=50, mkw=kw, device="cpu",
                              **({"lr_note": "deepnn uses lr 1e-3"} if m == "deepnn" else {})))
    return jobs


def phase4b():
    """Matched 50-cycle baselines on FD002/FD003 so the hybrids can be compared like for like."""
    MPS = {"tcn", "teacher"}
    jobs = []
    for seed in SEEDS:
        for sub in ["FD002", "FD003"]:
            for m in ["lstm", "gru", "tcn", "transformer", "teacher"]:
                kw = dict(dilations=(1, 2, 4, 8, 16)) if m == "tcn" else dict(window=50) if m == "transformer" else {}
                jobs.append(J(kind="base", subset=sub, model=m, tag="w50", seed=seed, epochs=30, patience=7, window=50, mkw=kw, device="mps" if m in MPS else "cpu"))
    return jobs


# ===================== protocol v2: test-like validation, window sweep, full benchmark, pooled training =====================
V2 = dict(epochs=40, patience=8, select="tl")
SWEEP_W = [20, 30, 40, 50, 60, 80, 100]
SWEEP_MODELS = ["gru", "lstm", "tcn", "teacher"]
V2_MODELS = ["widenn", "deepnn", "rnn", "gru", "lstm", "bilstm", "cnnlstm", "tcn", "transformer", "teacher", "transformerteacher",
             "tcnattn", "cnnbigruattn", "convtransformer", "mamba", "mamba2", "bimamba", "bimambateacher"]
V2_SEEDS = list(range(8))
DIL = (1, 2, 4, 8, 16, 32)                           # receptive field 127 >= every window we try


def w2():
    """Window chosen by scripts/choose_window_v2.py (test-like validation, one-standard-error rule)."""
    f = ROOT / "results" / "window_choice_v2.json"
    return int(json.loads(f.read_text())["window"]) if f.exists() else 50


def _mkw(m):
    return dict(dilations=DIL) if m in ("tcn", "tcnattn") else {}


def sweep():
    jobs = []
    for W in SWEEP_W:
        for sub in SUBS:
            for m in SWEEP_MODELS:
                for seed in range(3):
                    jobs.append(J(kind="base", subset=sub, model=m, tag=f"sw{W}", seed=seed, window=W, mkw=_mkw(m), **V2))
    return jobs


def v2base():
    W = w2(); jobs = []
    for sub in SUBS:
        for m in V2_MODELS:
            for seed in (range(5) if m in ("mamba", "mamba2", "bimamba", "bimambateacher") else V2_SEEDS):
                j = J(kind="base", subset=sub, model=m, tag="v2", seed=seed, window=W, mkw=_mkw(m), **V2)
                if m in TEACHERS:
                    j["save_ckpt"] = True; j["ck_tag"] = "v2"
                jobs.append(j)
    return jobs


def v2distill():
    """Headline: spiking / small-GRU / small-LSTM students, with and without the teacher, 8 seeds, all four subsets, window w2()."""
    W = w2(); jobs = []
    for sd in V2_SEEDS:
        for s in SUBS:
            for st in ("spike", "smallgru", "smalllstm"):
                for t in (None, "teacher"):
                    j = D(s, sd, t, st, tag="v2h"); j.update(window=W, ck_tag="v2", **V2); jobs.append(j)
    return jobs


def v2abl():
    """Design ablations under protocol v2 (window w2(), early stop on test-like validation). FD001 (single regime) and FD004 (six regimes, where the teacher matters).
    The default configuration and the no-teacher control are the v2h runs (seeds 0-4 are reused), not repeated here."""
    W = w2(); jobs = []
    loss = {"task+kd": dict(alpha_traj=0, gamma_attn=0, eta_sparse=0), "task+kd+traj": dict(gamma_attn=0, eta_sparse=0),
            "task+kd+attn": dict(alpha_traj=0, eta_sparse=0), "task+kd+sparse": dict(alpha_traj=0, gamma_attn=0),
            "full-kd": dict(lambda_kd=0), "full-traj": dict(alpha_traj=0), "full-attn": dict(gamma_attn=0), "full-sparse": dict(eta_sparse=0)}
    arch = {"atan": dict(surrogate="atan"), "triangle": dict(surrogate="triangle"), "fixed_beta": dict(learn_beta=False), "fixed_vth": dict(learn_vth=False),
            "feedforward": dict(recurrent=False), "h16": dict(hidden=16), "h32": dict(hidden=32), "h96": dict(hidden=96)}
    for sd in range(5):
        for s in ("FD001", "FD004"):
            for tag, kw in loss.items():
                j = D(s, sd, "teacher", "spike", tag="a2_" + tag, **kw); j.update(window=W, ck_tag="v2", **V2); jobs.append(j)
            for r in (0.03, 0.06, 0.10, 0.25, 0.40):
                j = D(s, sd, "teacher", "spike", tag=f"a2_rate{r}", target_spike_rate=r); j.update(window=W, ck_tag="v2", **V2); jobs.append(j)
            for tag, skw in arch.items():
                j = D(s, sd, "teacher", "spike", tag="a2_" + tag, skw=skw); j.update(window=W, ck_tag="v2", **V2); jobs.append(j)
            j = D(s, sd, "teacher", "spike", tag="a2_meanpool", skw=dict(readout="mean"), gamma_attn=0); j.update(window=W, ck_tag="v2", **V2); jobs.append(j)
    return jobs


WINDOWED = {"transformer", "transformerteacher"}      # learned positional embedding of length = window
POOL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]   # informative in every subset
POOL_MODELS = ["gru", "lstm", "tcn", "transformer", "teacher", "mamba2"]


def pooled():
    """One model trained on the windows of ALL four subsets together (common 14 sensors), scored on each subset's test engines."""
    W = w2(); jobs = []
    for m in POOL_MODELS:
        for seed in range(5):
            jobs.append(J(kind="pooled", subset="ALL", model=m, tag="v2", seed=seed, window=W, mkw=_mkw(m), **V2))
    return jobs


def run_pooled(job):
    import inspect
    import numpy as np, torch
    from rul.data import prepare
    from rul.energy import count_macs, energy_pj
    from rul.metrics import all_metrics
    from rul.models import count_params, create_model
    from rul.train import fit
    seed, W = job["seed"], job["window"]
    torch.manual_seed(seed); np.random.seed(seed)
    parts = {s: prepare(s, seed=seed, window=W, condition_aware=s in MULTI, sensors=POOL_SENSORS) for s in SUBS}
    cat = lambda k: np.concatenate([parts[s][k] for s in SUBS])
    d = {k: cat(k) for k in ("X_train", "y_train", "X_val", "y_val", "X_val_tl", "y_val_tl", "X_test", "y_test")}
    mkw = dict(job.get("mkw", {}))
    m = create_model(job["model"], len(POOL_SENSORS), **mkw)
    if "window" not in mkw and (job["model"] in WINDOWED or "window" in inspect.signature(type(m).__init__).parameters):
        m = create_model(job["model"], len(POOL_SENSORS), window=W, **mkw)
    r = fit(m, d, epochs=job["epochs"], patience=job["patience"], lr=LR.get(job["model"], 1e-3), device=job.get("device", "cpu"),
            verbose=False, y_scale=125.0, select=job.get("select", "tl"))
    p = np.asarray(r["test_pred"], float); y = d["y_test"].astype(float); off = 0; by = {}
    for s in SUBS:
        n = len(parts[s]["y_test"]); ps, ys = p[off:off + n], y[off:off + n]; off += n
        by[s] = {**all_metrics(ys, ps), "RMSE_c125": float(np.sqrt(np.mean((ps - np.minimum(ys, 125)) ** 2))), "n": n}
    macs = count_macs(m, W, len(POOL_SENSORS))
    rec = {k: job[k] for k in job if k not in ("mkw",)}
    rec.update(params=count_params(m), macs=macs, energy_pj=energy_pj(macs), val_tl=r["val_tl"], epochs_run=r["epochs_run"], train_seconds=r["train_seconds"],
               test_by_subset=by, n_fit=len(d["X_train"]), test_pred=r["test_pred"], y_test=d["y_test"].tolist(), test_offsets={s: len(parts[s]["y_test"]) for s in SUBS})
    return rec


def run(job):
    out = OUT / f"{job['id'].replace('>', '').replace('/', '_')}.json"
    if out.exists():
        return job["id"], "skip"
    try:
        import numpy as np, torch
        torch.set_num_threads(1)
        from rul.data import prepare
        from rul.distill import fit_distillation
        from rul.energy import count_macs, energy_pj
        from rul.models import count_params, create_model
        from rul.train import fit

        if job["kind"] == "pooled":
            rec = run_pooled(job)
            OUT.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec))
            return job["id"], "ok pooled " + " ".join(f"{k}:{v['RMSE_c125']:.1f}" for k, v in rec["test_by_subset"].items())
        seed, sub = job["seed"], job["subset"]
        torch.manual_seed(seed); np.random.seed(seed)
        regime = job.get("regime", True) and sub in MULTI
        W = job.get("window", 30)
        dev = job.get("device", "cpu")
        if dev == "cpu":
            torch.set_num_threads(1)
        d = prepare(sub, seed=seed, window=W, condition_aware=regime, smooth_alpha=job.get("smooth"))
        n = len(d["sensors"])
        rec = {k: job[k] for k in job if k not in ("loss", "skw", "mkw")}
        rec.update(n_sensors=n, sensors=d["sensors"], n_fit=len(d["X_train"]))
        if job["kind"] == "base":
            import inspect
            mkw = dict(job.get("mkw", {}))
            m = create_model(job["model"], n, **mkw)
            if "window" not in mkw and (job["model"] in WINDOWED or "window" in inspect.signature(type(m).__init__).parameters):
                m = create_model(job["model"], n, window=W, **mkw)            # window-dependent layers (flatten / positional embedding)
            r = fit(m, d, epochs=job["epochs"], patience=job["patience"], lr=LR.get(job["model"], 1e-3),
                    device=dev, verbose=False, y_scale=125.0, select=job.get("select", "all"))
            macs = count_macs(m, W, n)
            rec.update(params=count_params(m), macs=macs, acs=0, energy_pj=energy_pj(macs), **{k: r[k] for k in (
                "val", "test", "train_rmse_history", "val_rmse_history", "epochs_run", "train_seconds")},
                       test_pred=r["test_pred"], y_test=d["y_test"].tolist())
            if "val_pseudo" in r:
                rec["val_pseudo"] = r["val_pseudo"]
            if "val_tl" in r:
                rec["val_tl"] = r["val_tl"]; rec["val_tl_rmse_history"] = r["val_tl_rmse_history"]
            if job["model"] in TEACHERS and (job["tag"] == "" or job.get("save_ckpt")):
                CK.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in m.state_dict().items()}, CK / ckname(job["model"], sub, seed, W, job.get("ck_tag")))
        else:
            teacher = None
            if job["teacher"]:
                teacher = create_model(job["teacher"], n)
                teacher.load_state_dict(torch.load(CK / ckname(job["teacher"], sub, seed, W, job.get("ck_tag")), map_location="cpu"))
            sname = {"spike": "student", "smalllstm": "smalllstm", "smallgru": "smallgru"}[job["student"]]
            st = create_model(sname, n, **job["skw"])
            r = fit_distillation(st, teacher, d, epochs=job["epochs"], patience=job["patience"], device=dev,
                                 verbose=False, select=job.get("select", "all"), **job["loss"])
            if job["student"] == "spike":
                o = st.ops(W, r["rate1"], r["rate2"]); macs, acs = o["macs"], o["acs"]
            else:
                macs, acs = count_macs(st, W, n), 0
            from rul.train import predict
            rec.update(params=count_params(st), macs=macs, acs=acs, energy_pj=energy_pj(macs, acs),
                       **{k: v for k, v in r.items() if k != "history"}, history=r["history"],
                       test_pred=predict(st, d["X_test"], dev, y_scale=125.0).clip(0, 125).tolist(),
                       y_test=d["y_test"].tolist())
        OUT.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec))
        return job["id"], f"ok test RMSE {rec['test']['RMSE']:.2f}"
    except Exception:
        return job["id"], "FAIL " + traceback.format_exc()[-600:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase1", "phase2", "phase3", "phase4", "phase4b", "phase4c", "phase5a", "phase5b", "sweep", "v2base", "v2distill", "pooled", "v2abl"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--exclude", default="", help="skip jobs whose id contains this substring")
    ap.add_argument("--device", default="", help="run only jobs assigned to this device (cpu|mps)")
    a = ap.parse_args()
    jobs = [j for j in {"phase1": phase1, "phase2": phase2, "phase3": phase3, "phase4": phase4, "phase4b": phase4b, "phase4c": phase4c, "phase5a": phase5a, "phase5b": phase5b, "sweep": sweep, "v2base": v2base, "v2distill": v2distill, "pooled": pooled, "v2abl": v2abl}[a.phase]() if a.only in j["id"] and not (a.exclude and a.exclude in j["id"]) and (not a.device or j.get("device", "cpu") == a.device)]
    # slowest first so the pool drains evenly
    jobs.sort(key=lambda j: not (j.get("model") in MAMBA or j.get("model") == "mamba2" or j.get("teacher") == "bimambateacher"))
    print(len(jobs), "jobs", flush=True)
    t0 = time.time()
    with Pool(a.workers) as p:
        for i, (jid, msg) in enumerate(p.imap_unordered(run, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time()-t0:6.0f}s] {jid}: {msg}", flush=True)
