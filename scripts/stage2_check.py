"""Этап 2, проверка пригодности: признаки этапа 1 по каждой форме контроля и по сценариям.
Обучение на одном seed, проверка на другом (студенты не пересекаются). Запуск: python scripts/stage2_check.py"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from edurisk.features import FEATURES, build_features
from edurisk.world import simulate_world

WEEKS = (4, 8, 12)


def data(scenario, seed, form, cohort_year=0):
    w = simulate_world(3000, seed=seed, scenario=scenario, cohort_year=cohort_year)
    cs = np.flatnonzero((w.curriculum.form == form).to_numpy())
    out = {}
    for t in WEEKS:
        X = [build_features(w.course_view(c), t) for c in cs]
        y = [w.course_view(c).outcome for c in cs]
        import pandas as pd
        out[t] = (pd.concat(X), np.concatenate(y))
    return out


def fit_eval(train, test, t):
    Xa, ya = train[t]
    keep = [c for c in FEATURES if Xa[c].notna().any()]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=0).fit(Xa[keep], ya)
    p = m.predict_proba(test[t][0][keep])[:, 1]
    yt = test[t][1]
    return roc_auc_score(yt, p), average_precision_score(yt, p), yt.mean(), p.mean()


lines = ["Этап 2: бустинг на признаках этапа 1; обучение seed=1, тест seed=2", ""]
lines.append("По формам контроля, сценарий S0 (ROC-AUC / PR-AUC при доле провалов):")
for form in ("zachet", "difzachet", "exam", "coursework"):
    tr, te = data("S0", 1, form), data("S0", 2, form)
    row = []
    for t in WEEKS:
        a, pr, base, _ = fit_eval(tr, te, t)
        row.append(f"нед.{t}: {a:.3f}/{pr:.3f} (база {base:.2f})")
    lines.append(f"  {form:10s} " + " | ".join(row))
lines.append("")
lines.append("Сдвиг когорты S2, экзамены (обучение на когорте 0, тест на когорте 1; ROC-AUC, средний прогноз vs факт):")
tr = data("S2", 1, "exam", 0)
for name, yr in (("та же когорта", 0), ("новая когорта", 1)):
    te = data("S2", 2, "exam", yr)
    for t in WEEKS:
        a, pr, base, pm = fit_eval(tr, te, t)
        lines.append(f"  {name:14s} нед.{t:2d}: AUC {a:.3f}, прогноз {pm:.3f}, факт {base:.3f}")
txt = "\n".join(lines)
print(txt)
open("docs/stage2_results.txt", "w", encoding="utf-8").write(txt + "\n")
