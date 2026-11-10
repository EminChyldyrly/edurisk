"""Этап 4: сравнение моделей риска на 5 мирах (seed 1..5), недели 4 и 8. Запуск: python scripts/stage4_experiment.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.metrics import roc_auc_score

from edurisk.costs import bayes_threshold, mean_cost, row_cost, ze_weights
from edurisk.experiment import GAMMA, K, MODELS, run_seed
from edurisk.metrics import holm, paired_bootstrap, paired_bootstrap_loss

SEEDS, N, WEEKS = range(1, 6), 2000, (4, 8)
runs = {}
for s in SEEDS:
    runs[s], mze = run_seed(s, N, WEEKS)
    print("seed", s, "готов", flush=True)

L = ["Этап 4: модели риска провала (T1). 5 миров по %d студентов, разбиение 60/20/20 по студентам." % N,
     f"Стоимость: k={K} (пропуск провала = {K} ложных тревог), ЗЕТ-вес γ={GAMMA}, порог 1/(1+k); среднее по seed ± sd.", ""]
for t in WEEKS:
    L.append(f"=== Неделя {t} ===")
    L.append(f"{'модель':14s} {'ROC-AUC':>14s} {'PR-AUC':>14s} {'Brier':>14s} {'ECE':>14s} {'стоимость':>14s} {'W-recall@10%':>14s}")
    for m in MODELS:
        cells = []
        for key in ("roc_auc", "pr_auc", "brier", "ece", "cost", "wrecall_top10"):
            v = np.array([runs[s][t]["metrics"][m][key] for s in SEEDS])
            cells.append(f"{v.mean():.3f}±{v.std(ddof=1):.3f}")
        L.append(f"{m:14s} " + " ".join(f"{c:>14s}" for c in cells))
    for b in ("_baseline_cost_none", "_baseline_cost_all"):
        v = np.array([runs[s][t]["metrics"][b]["cost"] for s in SEEDS])
        L.append(f"{'никого не помечать' if b.endswith('none') else 'помечать всех':28s} стоимость {v.mean():.3f}")
    L.append("")

    # парные сравнения на объединённой тестовой выборке (группа = студент внутри мира)
    y = np.concatenate([runs[s][t]["test"].y.to_numpy() for s in SEEDS])
    g = np.concatenate([runs[s][t]["test"].student_id.to_numpy() + 10_000 * s for s in SEEDS])
    ze = np.concatenate([runs[s][t]["test"].ze.to_numpy() for s in SEEDS])
    w = ze_weights(ze, GAMMA, mze)
    P = {m: np.concatenate([runs[s][t]["models"][m].predict_proba(runs[s][t]["test"]) for s in SEEDS]) for m in MODELS}
    pairs = [("logit", "rule"), ("boost+cal", "logit"), ("boost_cs+cal", "boost+cal"), ("stack", "logit"), ("stack", "boost+cal")]
    thr = bayes_threshold(K)
    ca, cc = [], []
    for a, b in pairs:
        ca.append(paired_bootstrap(roc_auc_score, y, P[a], P[b], g, n_boot=300, seed=1))
        cc.append(paired_bootstrap_loss(row_cost(y, P[a] >= thr, w, K), row_cost(y, P[b] >= thr, w, K), g))
    pa, pc = holm([r["p"] for r in ca]), holm([r["p"] for r in cc])
    L.append(f"Парные сравнения (разность A−B, 95% ИИ бутстрэпом по студентам, p с поправкой Холма в каждой группе из {len(pairs)}):")
    for (a, b), ra, rc, ha, hc in zip(pairs, ca, cc, pa, pc):
        L.append(f"  {a:13s} − {b:11s} | ROC-AUC {ra['diff']:+.4f} [{ra['lo']:+.4f}; {ra['hi']:+.4f}] p={ha:.3f}"
                 f" | стоимость {rc['diff']:+.4f} [{rc['lo']:+.4f}; {rc['hi']:+.4f}] p={hc:.3f}")
    L.append("")

    # чувствительность к k и γ (модели обучены при k=3, γ=2; меняется только оценка и порог решения)
    L.append("Чувствительность стоимости (строки: модель; столбцы: k, γ):")
    combos = [(kk, gg) for kk in (1, 3, 5) for gg in (0, 1, 2)]
    L.append(f"{'':14s}" + "".join(f"{f'k{kk} g{gg}':>9s}" for kk, gg in combos))
    for m in ("rule", "logit", "boost+cal", "stack"):
        row = [mean_cost(y, P[m] >= bayes_threshold(kk), ze_weights(ze, gg, mze), kk) for kk, gg in combos]
        L.append(f"{m:14s}" + "".join(f"{v:9.3f}" for v in row))
    L.append("")
txt = "\n".join(L)
print(txt)
open(pathlib.Path(__file__).resolve().parent.parent / "docs" / "stage4_results.txt", "w", encoding="utf-8").write(txt + "\n")
