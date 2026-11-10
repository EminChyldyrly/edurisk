"""Этап 6, уточнение проверки объяснений: заложенная причина «потеря интереса» учитывается только тогда, когда
она уже проявилась в данных к неделе 8 (начало спада не позже недели 6). Запуск: python scripts/stage6_explain_refined.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from edurisk.dataset import build_dataset, split_students, take
from edurisk.explain import GROUPS, group_shapley
from edurisk.models import BoostModel
from edurisk.world import simulate_world

L = []
w = simulate_world(2000, seed=7)
d = build_dataset(w, weeks=(8,))
parts = split_students(d.student_id.unique(), seed=7)
tr, te = take(d, parts["train"]), take(d, parts["test"])
model = BoostModel(0).fit(tr, tr.y.to_numpy())
bg = tr.sample(100, random_state=0)
ab, dis, onset = w.truth["student"]["ability"], w.truth["disengaging"], w.truth["onset"]
sid, sem = te.student_id.to_numpy(), te.semester.to_numpy()
fail = (te.y == 1).to_numpy()
cats = {
    "потеря интереса, уже видна (начало ≤ нед. 6)": fail & dis[sid, sem] & (onset[sid, sem] <= 6) & (ab[sid] > -0.5),
    "потеря интереса, ещё не видна (начало ≥ нед. 9)": fail & dis[sid, sem] & (onset[sid, sem] >= 9) & (ab[sid] > -0.5),
    "низкие способности, интерес не терялся": fail & ~dis[sid, sem] & (ab[sid] < -1.0)}
rng = np.random.default_rng(0)
gi = {g: i for i, g in enumerate(GROUPS)}
L.append("Проверка объяснений на симуляторе (S0, неделя 8, бустинг): у провалившихся студентов известна заложенная причина.")
attr = {}
for k, m in cats.items():
    ix = rng.choice(np.flatnonzero(m), size=min(120, m.sum()), replace=False)
    attr[k] = np.array([[group_shapley(model.predict_proba, te.iloc[[i]], bg)[g] for g in GROUPS] for i in ix])
    pos = np.clip(attr[k], 0, None)
    sh = pos / np.maximum(pos.sum(1, keepdims=True), 1e-9)
    L.append(f"- {k}: в тесте {m.sum()}, объяснено {len(ix)}; средний вклад " + ", ".join(f"{g} {attr[k][:, gi[g]].mean():+.3f}" for g in GROUPS)
             + f"; доля активности в положительных вкладах {sh[:, gi['activity']].mean():.2f}, контрольных {sh[:, gi['assessments']].mean():.2f}")

def share(A, g):
    pos = np.clip(A, 0, None)
    return pos[:, gi[g]] / np.maximum(pos.sum(1), 1e-9)

def boot(a, b):
    bs = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for _ in range(1000)]
    return a.mean() - b.mean(), np.quantile(bs, .025), np.quantile(bs, .975)

v, h, l = (attr[k] for k in cats)
for name, A, B in (("видная потеря интереса − низкие способности", v, l), ("ещё не видная потеря интереса − низкие способности", h, l)):
    for g in ("activity", "assessments"):
        d_, lo, hi = boot(share(A, g), share(B, g))
        L.append(f"  доля группы «{g}»: {name}: {d_:+.2f} [{lo:+.2f}; {hi:+.2f}]")
txt = "\n".join(L)
print(txt)
open(pathlib.Path(__file__).resolve().parent.parent / "docs" / "stage6_explain_refined.txt", "w", encoding="utf-8").write(txt + "\n")
