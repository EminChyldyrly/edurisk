"""Этап 6: (A) абляция признаков, (B) проверка объяснений на симуляторе. Запуск: python scripts/stage6_experiment.py"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from edurisk.dataset import build_dataset, split_students, take
from edurisk.explain import GROUPS, group_importance, group_shapley
from edurisk.features import DERIVED, RAW
from edurisk.metrics import paired_bootstrap
from edurisk.models import BoostModel, LogitModel
from edurisk.world import simulate_world

CONTEXT = GROUPS["background"] + GROUPS["history"] + GROUPS["course"]
SETS = {"контекст": CONTEXT, "контекст+сырые": CONTEXT + RAW, "контекст+производные": CONTEXT + DERIVED,
        "всё": CONTEXT + RAW + DERIVED}
SEEDS, N, WEEKS, SCEN = range(1, 5), 1200, (4, 8), ("S0", "S1", "S4")
L = []
t0 = time.time()

# ---------- A. абляция
auc = {}      # (scen, week, set, model) -> [auc по seed]
pooled = {}   # (scen, week) -> (y, {set: p}, groups) для парного бутстрэпа (бустинг)
for sc in SCEN:
    for s in SEEDS:
        w = simulate_world(N, seed=s, scenario=sc)
        d = build_dataset(w, weeks=WEEKS)
        for t in WEEKS:
            dt = d[d.week == t]
            parts = split_students(dt.student_id.unique(), seed=s)
            tr, te = take(dt, parts["train"]), take(dt, parts["test"])
            for name, cols in SETS.items():
                for mname, mk in (("logit", lambda c: LogitModel(features=c)), ("boost", lambda c: BoostModel(s, features=c))):
                    p = mk(cols).fit(tr, tr.y.to_numpy()).predict_proba(te)
                    auc.setdefault((sc, t, name, mname), []).append(roc_auc_score(te.y, p))
                    if mname == "boost":
                        e = pooled.setdefault((sc, t), {"y": [], "g": [], "p": {k: [] for k in SETS}})
                        e["p"][name].append(p)
                        if name == "контекст":
                            e["y"].append(te.y.to_numpy())
                            e["g"].append(te.student_id.to_numpy() + 100_000 * s + 1_000_000 * 0)
    print("A:", sc, "готов", round(time.time() - t0), "с", flush=True)

ms = lambda v: f"{np.mean(v):.3f}±{np.std(v, ddof=1):.3f}"
L.append(f"А. Абляция признаков (ROC-AUC, {len(SEEDS)} мира по {N} студентов; обучение 60% студентов, тест 20%).")
for sc in SCEN:
    for t in WEEKS:
        L.append(f"-- сценарий {sc}, неделя {t}")
        for mname in ("logit", "boost"):
            L.append(f"   {mname:6s} " + "  ".join(f"{n}: {ms(auc[(sc, t, n, mname)])}" for n in SETS))
        e = pooled[(sc, t)]
        y, g = np.concatenate(e["y"]), np.concatenate(e["g"])
        P = {k: np.concatenate(v) for k, v in e["p"].items()}
        for a, b in (("контекст+сырые", "контекст"), ("всё", "контекст+сырые"), ("контекст+производные", "контекст+сырые")):
            r = paired_bootstrap(roc_auc_score, y, P[a], P[b], g, n_boot=200, seed=1)
            L.append(f"   бустинг: {a} − {b}: {r['diff']:+.4f} [{r['lo']:+.4f}; {r['hi']:+.4f}]")
L.append("")

# ---------- B. проверка объяснений на симуляторе (S0, неделя 8)
w = simulate_world(2000, seed=7)
d = build_dataset(w, weeks=(8,))
parts = split_students(d.student_id.unique(), seed=7)
tr, te = take(d, parts["train"]), take(d, parts["test"])
model = BoostModel(0).fit(tr, tr.y.to_numpy())
bg = tr.sample(100, random_state=0)
ab = w.truth["student"]["ability"]
dis = w.truth["disengaging"]
sid, sem = te.student_id.to_numpy(), te.semester.to_numpy()
cause_dis = (te.y == 1).to_numpy() & dis[sid, sem] & (ab[sid] > -0.5)
cause_abl = (te.y == 1).to_numpy() & ~dis[sid, sem] & (ab[sid] < -1.0)
rng = np.random.default_rng(0)
pick = lambda m: rng.choice(np.flatnonzero(m), size=min(120, m.sum()), replace=False)
rows = {"потеря интереса": pick(cause_dis), "низкие способности": pick(cause_abl)}
beh, cap = ["activity", "assessments"], ["history", "background"]
attr = {}
for k, ix in rows.items():
    attr[k] = np.array([[group_shapley(model.predict_proba, te.iloc[[i]], bg)[g] for g in GROUPS] for i in ix])
L.append("Б. Проверка объяснений: у отобранных провалившихся студентов известна заложенная причина (S0, неделя 8).")
L.append(f"   в тесте: потеря интереса при нормальных способностях — {cause_dis.sum()}, низкие способности без потери интереса — {cause_abl.sum()}; объяснено по {len(rows['потеря интереса'])} и {len(rows['низкие способности'])}.")
gi = {g: i for i, g in enumerate(GROUPS)}
for k, A in attr.items():
    L.append(f"   {k:20s} средний вклад (в вероятности): " + ", ".join(f"{g} {A[:, gi[g]].mean():+.3f}" for g in GROUPS))
def share(A):  # доля «поведенческих» групп в сумме положительных вкладов
    pos = np.clip(A, 0, None)
    return pos[:, [gi[g] for g in beh]].sum(1) / np.maximum(pos.sum(1), 1e-9)
sa, sb = share(attr["потеря интереса"]), share(attr["низкие способности"])
bs = [np.mean(rng.choice(sa, len(sa))) - np.mean(rng.choice(sb, len(sb))) for _ in range(1000)]
L.append(f"   доля поведенческих групп (активность+контрольные) в положительных вкладах: потеря интереса {sa.mean():.2f}, низкие способности {sb.mean():.2f}, разность {sa.mean()-sb.mean():+.2f} [{np.quantile(bs,.025):+.2f}; {np.quantile(bs,.975):+.2f}]")

# проверка на здравый смысл: модель, обученная на перемешанных метках
rnd = BoostModel(0).fit(tr, rng.permutation(tr.y.to_numpy()))
ix = rows["потеря интереса"][:60]
A_true = attr["потеря интереса"][:60]
A_rnd = np.array([[group_shapley(rnd.predict_proba, te.iloc[[i]], bg)[g] for g in GROUPS] for i in ix])
L.append(f"   проверка перемешанными метками: корреляция вкладов настоящей и «случайной» модели {np.corrcoef(A_true.ravel(), A_rnd.ravel())[0,1]:+.2f}; "
         f"средняя абсолютная величина вкладов {np.abs(A_true).mean():.3f} против {np.abs(A_rnd).mean():.3f}")

# стабильность: пересборка модели на бутстрэп-выборках студентов, групповая перестановочная важность
ids = tr.student_id.unique()
imps = []
for b in range(8):
    pick_ids = set(np.random.default_rng(100 + b).choice(ids, len(ids)).tolist())
    sub = tr[tr.student_id.isin(pick_ids)]
    m_b = BoostModel(b).fit(sub, sub.y.to_numpy())
    imp = group_importance(m_b.predict_proba, te, te.y.to_numpy(), roc_auc_score, seed=b)
    imps.append([imp[g] for g in GROUPS])
imps = np.array(imps)
rho = [spearmanr(imps[i], imps[j])[0] for i in range(len(imps)) for j in range(i + 1, len(imps))]
L.append("   групповая важность (падение AUC при перемешивании группы), среднее по 8 пересборкам: " + ", ".join(f"{g} {imps[:, gi[g]].mean():.3f}±{imps[:, gi[g]].std(ddof=1):.3f}" for g in GROUPS))
L.append(f"   стабильность порядка групп: средняя ранговая корреляция Спирмена между пересборками {np.mean(rho):.2f} (мин. {np.min(rho):.2f})")
txt = "\n".join(L)
print(txt)
open(pathlib.Path(__file__).resolve().parent.parent / "docs" / "stage6_results.txt", "w", encoding="utf-8").write(txt + "\n")
