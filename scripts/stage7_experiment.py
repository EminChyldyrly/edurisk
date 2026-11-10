"""Этап 7: справедливость по origin и gender на S0 (нет смещения измерения) и S3 (есть).
Запуск: python scripts/stage7_experiment.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from edurisk.costs import bayes_threshold
from edurisk.dataset import build_dataset, split_students, take
from edurisk.fairness import apply_thresholds, disparity_ci, equalize_fpr_thresholds, group_report, rates
from edurisk.models import BoostModel, Calibrated
from edurisk.world import simulate_world

SEEDS, N, WEEK, K = range(1, 6), 2000, 8, 3.0
THR = bayes_threshold(K)
L = [f"Справедливость модели риска (бустинг + калибровка), неделя {WEEK}, порог {THR:.2f}; {len(SEEDS)} мира по {N} студентов.", ""]
for sc in ("S0", "S3"):
    ys, ps, gs, sids, gen, vy, vp, vg = [], [], [], [], [], [], [], []
    for s in SEEDS:
        w = simulate_world(N, seed=s, scenario=sc)
        d = build_dataset(w, weeks=(WEEK,))
        d["origin"] = w.students.origin.reindex(d.student_id).to_numpy()
        d["gender"] = w.students.gender.reindex(d.student_id).to_numpy()
        parts = split_students(d.student_id.unique(), seed=s)
        tr, va, te = (take(d, parts[k]) for k in ("train", "val", "test"))
        m = Calibrated(BoostModel(s).fit(tr, tr.y.to_numpy())).fit_calibration(va, va.y.to_numpy())
        ys.append(te.y.to_numpy()); ps.append(m.predict_proba(te)); gs.append(te.origin.to_numpy())
        gen.append(te.gender.to_numpy()); sids.append(te.student_id.to_numpy() + 10_000 * s)
        vy.append(va.y.to_numpy()); vp.append(m.predict_proba(va)); vg.append(va.origin.to_numpy())
    y, p, g, gd, sid = map(np.concatenate, (ys, ps, gs, gen, sids))
    vy, vp, vg = map(np.concatenate, (vy, vp, vg))
    L.append(f"=== Сценарий {sc} ===")
    rep = group_report(y, p, g, THR)
    L.append("По происхождению (origin):\n" + rep.round(3).to_string())
    r = disparity_ci(y, p, g, sid, THR, ref="local", other="other_region", n_boot=300, seed=1)
    L.append("   разность other_region − local: " + "; ".join(f"{k} {v['diff']:+.3f} [{v['lo']:+.3f}; {v['hi']:+.3f}]" for k, v in r.items()))
    repg = group_report(y, p, gd, THR)
    L.append("По полу (gender):\n" + repg.round(3).to_string())
    r = disparity_ci(y, p, gd, sid, THR, ref="m", other="f", n_boot=300, seed=1)
    L.append("   разность f − m: " + "; ".join(f"{k} {v['diff']:+.3f} [{v['lo']:+.3f}; {v['hi']:+.3f}]" for k, v in r.items()))
    if sc == "S3":
        tpr_all, fpr_all = rates(vy, vp >= THR)
        thr = equalize_fpr_thresholds(vy, vp, vg, target_fpr=fpr_all)
        flag = apply_thresholds(p, g, thr)
        rep2 = group_report(y, flag.astype(float), g, 0.5)
        L.append(f"Мера: пороги по origin, выравнивающие FPR на валидации (цель FPR {fpr_all:.3f}); найденные пороги: " + ", ".join(f"{k} {v:.3f}" for k, v in thr.items()))
        L.append(rep2[["n", "flag_rate", "tpr", "fpr"]].round(3).to_string())
        r = disparity_ci(y, flag.astype(float), g, sid, 0.5, ref="local", other="other_region", n_boot=300, seed=1)
        L.append("   разность other_region − local после меры: " + "; ".join(f"{k} {v['diff']:+.3f} [{v['lo']:+.3f}; {v['hi']:+.3f}]" for k, v in r.items()))
        tot = rates(y, flag)
        base = rates(y, p >= THR)
        L.append(f"   общая цена меры: TPR {base[0]:.3f} -> {tot[0]:.3f}, FPR {base[1]:.3f} -> {tot[1]:.3f}")
    L.append("")
txt = "\n".join(L)
print(txt)
open(pathlib.Path(__file__).resolve().parent.parent / "docs" / "stage7_results.txt", "w", encoding="utf-8").write(txt + "\n")
