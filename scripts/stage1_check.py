"""Эксперимент этапа 1: что дают производные признаки по сравнению с сырыми недельными значениями.

Обучение на когорте seed=1, проверка на независимой когорте seed=2 (по одной паре когорт,
без доверительных интервалов: разницы порядка 0.01 AUC считать значимыми нельзя).
Это проверка работоспособности конвейера на симуляторе, а не оценка качества на реальных студентах.
Запуск из корня репозитория:  python scripts/stage1_check.py
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from edurisk.features import STATIC, build_features
from edurisk.simulate import simulate


def raw_weekly(c, week):
    """Только то, что записано в системе: статика, логины по неделям, баллы контрольных."""
    o = c.observed(week)
    d = o.students[STATIC].copy()
    for t in range(week):
        d[f"logins_w{t + 1}"] = o.logins[:, t]
    for t in np.flatnonzero(o.cp_scheduled):
        d[f"score_w{t + 1}"] = o.cp_score[:, t]
    return d


def engineered(c, week):
    return build_features(c, week)


def both(c, week):
    return pd.concat([raw_weekly(c, week), build_features(c, week).drop(columns=STATIC)], axis=1)


SETS = [("сырые по неделям", raw_weekly), ("производные признаки", engineered), ("сырые + производные", both)]
N = 4000
train, test = simulate(N, seed=1), simulate(N, seed=2)
print(f"доля провалов: train {train.outcome.mean():.3f}, test {test.outcome.mean():.3f}")
print("ROC-AUC / PR-AUC (бустинг, независимая тестовая когорта)")
print(f"{'неделя':>6} | " + " | ".join(f"{name:>22}" for name, _ in SETS))
for week in (2, 4, 6, 8, 10, 12, 14, 16):
    cells = []
    for _, make in SETS:
        Xtr, Xte = make(train, week), make(test, week)
        cols = [c for c in Xtr if Xtr[c].notna().any()]       # на ранних неделях часть признаков ещё не определена
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0)
        m.fit(Xtr[cols], train.outcome)
        p = m.predict_proba(Xte[cols])[:, 1]
        cells.append(f"{roc_auc_score(test.outcome, p):.3f} / {average_precision_score(test.outcome, p):.3f}")
    print(f"{week:>6} | " + " | ".join(f"{c:>22}" for c in cells))
