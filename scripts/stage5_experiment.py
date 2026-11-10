"""Этап 5: T2 (оценка 2-5) и T3 (допуск), 5 миров по 2000 студентов, недели 4/8/12.
Запуск: python scripts/stage5_experiment.py  (около 8 минут)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from edurisk.costs import ze_weights
from edurisk.dataset import add_targets, build_dataset, split_students, take
from edurisk.experiment5 import (ALPHA, ASYM, GAMMA, T2_MODELS, conformal_eval, fit_t2, make_t2, run_seed5)
from edurisk.metrics import holm, paired_bootstrap_loss
from edurisk.ordinal import GRADES, cost_matrix, decide_argmax, decide_min_cost
from edurisk.world import simulate_world

SEEDS, N, WEEKS = range(1, 6), 2000, (4, 8, 12)
runs = {}
for s in SEEDS:
    runs[s], mze = run_seed5(s, N, WEEKS)
    print("seed", s, "готов", flush=True)

ms = lambda v: f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}"
L = [f"Этап 5. 5 миров по {N} студентов; оценки: асимметрия завышения {ASYM}, ЗЕТ-вес γ={GAMMA}; конформный уровень {1-ALPHA:.0%}.", ""]
for t in WEEKS:
    L += [f"=== Неделя {t}: T2, оценка 2-5 (дифзачёты, экзамены, курсовые) ===",
          f"{'модель':12s} {'NLL':>12s} {'QWK':>12s} {'MAE':>12s} {'macroF1':>12s} {'recall 2':>12s} {'AUC(=2)':>12s} {'стоим.argmax':>13s} {'стоим.min':>12s} {'recall 2 min':>13s}"]
    for m in T2_MODELS:
        g = lambda k: [runs[s][t]["t2"]["metrics"][m][k] for s in SEEDS]
        L.append(f"{m:12s} {ms(g('nll')):>12s} {ms(g('argmax_qwk')):>12s} {ms(g('argmax_mae')):>12s} {ms(g('argmax_macro_f1')):>12s} "
                 f"{ms(g('argmax_recall_2')):>12s} {ms(g('auc_2')):>12s} {ms(g('cost_argmax')):>13s} {ms(g('cost_mincost')):>12s} {ms(g('recall_2_mincost')):>13s}")
    L.append("")
    # парные сравнения на объединённом тесте
    te = [runs[s][t]["t2"]["test"] for s in SEEDS]
    y = np.concatenate([x.grade.to_numpy() for x in te])
    grp = np.concatenate([x.student_id.to_numpy() + 10_000 * s for s, x in zip(SEEDS, te)])
    w = ze_weights(np.concatenate([x.ze.to_numpy() for x in te]), GAMMA, mze)
    M = cost_matrix(ASYM)
    P = {m: np.vstack([runs[s][t]["t2"]["models"][m].predict_proba(runs[s][t]["t2"]["test"]) for s in SEEDS]) for m in T2_MODELS}
    iy = np.searchsorted(GRADES, y)
    nll = lambda m: -np.log(np.clip(P[m][np.arange(len(y)), iy], 1e-6, 1))
    cost = lambda m, dec=decide_min_cost: w * M[iy, np.searchsorted(GRADES, dec(P[m], M) if dec is decide_min_cost else dec(P[m]))]
    cmp = [("fh_boost", "multi_boost", "каскад против многоклассовой (бустинг)"),
           ("fh_logit", "multi_logit", "каскад против многоклассовой (logit)"),
           ("fh_boost_ze", "fh_boost", "ЗЕТ-обучение против обычного (каскад)"),
           ("multi_boost", "multi_logit", "бустинг против logit (многоклассовые)")]
    rows = [(a, b, d, paired_bootstrap_loss(nll(a), nll(b), grp), paired_bootstrap_loss(cost(a), cost(b), grp)) for a, b, d in cmp]
    pn, pc = holm([r[3]["p"] for r in rows]), holm([r[4]["p"] for r in rows])
    L.append("Парные сравнения (A−B; отрицательная разность = A лучше; 95% ИИ по студентам; p с поправкой Холма):")
    for (a, b, d, rn, rc), hn, hc in zip(rows, pn, pc):
        L.append(f"  {d:42s} NLL {rn['diff']:+.4f} [{rn['lo']:+.4f}; {rn['hi']:+.4f}] p={hn:.3f} | стоимость {rc['diff']:+.4f} [{rc['lo']:+.4f}; {rc['hi']:+.4f}] p={hc:.3f}")
    rc = paired_bootstrap_loss(cost("fh_boost", decide_min_cost), cost("fh_boost", lambda P_: decide_argmax(P_)), grp)
    L.append(f"  решение по минимуму ожидаемой стоимости против argmax (fh_boost): стоимость {rc['diff']:+.4f} [{rc['lo']:+.4f}; {rc['hi']:+.4f}]")
    L.append("")
    L.append("Конформные наборы (fh_boost), доля покрытия и средний размер набора, покрытие по классам 2/3/4/5:")
    for kind in ("marginal", "classwise"):
        cov = [runs[s][t]["t2"]["conformal"][kind]["coverage"] for s in SEEDS]
        size = [runs[s][t]["t2"]["conformal"][kind]["size"] for s in SEEDS]
        byc = np.mean([runs[s][t]["t2"]["conformal"][kind]["coverage_by_class"] for s in SEEDS], axis=0)
        L.append(f"  {kind:10s} покрытие {ms(cov)}, размер {ms(size)}, по классам " + " / ".join(f"{c:.2f}" for c in byc))
    L.append("")
    L.append(f"=== Неделя {t}: T3, недопуск к аттестации (доля {np.mean([1 - runs[s][t]['t3']['test'].admitted.mean() for s in SEEDS]):.3f}) ===")
    for m in ("logit", "boost+cal"):
        L.append(f"  {m:10s} " + "  ".join(f"{k} {ms([runs[s][t]['t3']['metrics'][m][k] for s in SEEDS])}" for k in ("roc_auc", "pr_auc", "brier", "ece")))
    L.append("")

# сдвиг когорты: порог конформного набора с когорты 0, проверка на когорте 1 (другие студенты, seed+100)
L.append("=== Сдвиг когорты (S2), неделя 8: конформные наборы, откалиброванные на когорте 0 ===")
cov0, cov1, c2_0, c2_1, sz0, sz1 = [], [], [], [], [], []
for s in SEEDS:
    w0 = simulate_world(N, seed=s, scenario="S2", cohort_year=0)
    w1 = simulate_world(N, seed=s + 100, scenario="S2", cohort_year=1)
    mz = float(w0.curriculum.ze.mean())
    d0 = make_t2(add_targets(build_dataset(w0, weeks=(8,)), w0))
    d1 = make_t2(add_targets(build_dataset(w1, weeks=(8,)), w1))
    parts = split_students(d0.student_id.unique(), seed=s)
    tr, va, te = (take(d0, parts[k]) for k in ("train", "val", "test"))
    models = fit_t2(tr, va, mz, seed=s)
    a = conformal_eval(models["fh_boost"], va, te)["classwise"]
    b = conformal_eval(models["fh_boost"], va, d1)["classwise"]
    cov0.append(a["coverage"]); cov1.append(b["coverage"]); c2_0.append(a["coverage_by_class"][0]); c2_1.append(b["coverage_by_class"][0])
    sz0.append(a["size"]); sz1.append(b["size"])
L.append(f"  та же когорта: покрытие {ms(cov0)} (класс 2: {ms(c2_0)}), размер {ms(sz0)}")
L.append(f"  новая когорта: покрытие {ms(cov1)} (класс 2: {ms(c2_1)}), размер {ms(sz1)}")
txt = "\n".join(L)
print(txt)
open(pathlib.Path(__file__).resolve().parent.parent / "docs" / "stage5_results.txt", "w", encoding="utf-8").write(txt + "\n")
