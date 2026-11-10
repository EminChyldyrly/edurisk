"""Проверка самих тестов: вносим в код типичные ошибки и смотрим, что тесты падают.
Запуск из корня репозитория: python scripts/mutate.py"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MUTATIONS = [
    # (файл, что заменить, на что, описание)
    ("edurisk/simulate.py", "self.logins[:, :week]", "self.logins[:, :week + 1]", "утечка: срез захватывает следующую неделю"),
    ("edurisk/simulate.py", "self.cp_scheduled[:week]", "self.cp_scheduled[:week + 1]", "утечка в расписании контрольных"),
    ("edurisk/simulate.py", "self.cp_score[:, :week]", "self.cp_score[:, :week + 1]", "утечка баллов"),
    ("edurisk/simulate.py", "return sched & ~passed", "return sched & ~passed & ~np.isnan(self.cp_score)", "пропуск контрольной не считается провалом"),
    ("edurisk/simulate.py", ">= self.pass_score", "> self.pass_score", "граница проходного балла"),
    ("edurisk/simulate.py", "hazard[:, :3] = 0.0", "hazard[:, :2] = 0.0", "уход возможен на 3-й неделе"),
    ("edurisk/simulate.py", "outcome = dropped | (perf", "outcome = (perf", "ушедшие не считаются провалившими"),
    ("edurisk/features.py", "cur = np.where(failed[:, col], cur + 1, 0)", "cur = np.where(failed[:, col], cur + 1, cur)", "серия не обнуляется после успеха"),
    ("edurisk/features.py", "best = np.maximum(best, cur)", "best = cur", "максимальная серия = текущая"),
    ("edurisk/features.py", "L[:, -4:].sum(1)", "L[:, -3:].sum(1)", "окно 4 недель стало 3"),
    ("edurisk/features.py", "EWMA_ALPHA * L[:, t] + (1 - EWMA_ALPHA) * ew", "(1 - EWMA_ALPHA) * L[:, t] + EWMA_ALPHA * ew", "перепутаны веса EWMA"),
    ("edurisk/features.py", "k - 1 - np.argmax(has[:, ::-1], axis=1)", "k - np.argmax(has[:, ::-1], axis=1)", "сдвиг на единицу в «недель с входа»"),
    ("edurisk/features.py", "return (y * x).sum(1) / (x ** 2).sum()", "return (y * x).sum(1) / (x ** 2).sum() * -1", "знак тренда"),
    ("edurisk/features.py", "S[i, idx[-1]] - S[i, idx[-2]]", "S[i, idx[-2]] - S[i, idx[-1]]", "знак изменения балла"),
    ("edurisk/features.py", "out[\"cp_failed_count\"] / n_cp if n_cp else 0.0", "out[\"cp_failed_count\"] / max(k, 1)", "доля провалов от числа недель, а не контрольных"),
    ("edurisk/features.py", "out[\"logins_total\"] = L.sum(1)", "out[\"logins_total\"] = L[:, 1:].sum(1)", "потеряна первая неделя"),
    ("edurisk/features.py", "mean4 = out[\"logins_last4\"].mean()", "mean4 = out[\"logins_last4\"].median()", "относительная активность по медиане"),
    ("edurisk/world.py", "adm = (share >= cfg.admit_share)", "adm = (share > cfg.admit_share)", "граница допуска: строго больше"),
    ("edurisk/world.py", "gfin = np.where(adm, g_first, 2)", "gfin = g_first", "недопуск не превращается в двойку"),
    ("edurisk/world.py", "ok = np.where(is_graded, ok_graded, ok_zachet)", "ok = np.where(is_graded, ok_graded, adm)", "зачёт выдаётся только по допуску"),
    ("edurisk/world.py", "need = start_alive[:, None] & ~ok & ~dropped_here[:, None]", "need = start_alive[:, None] & ~ok", "пересдача и у отчисленных"),
    ("edurisk/world.py", "enrolled[:, cs] = alive[:, None]", "enrolled[:, cs] = True", "отчисленные продолжают учиться"),
    ("edurisk/world.py", "act = alive[:, None]", "act = start_alive[:, None]", "активность после ухода"),
    ("edurisk/world.py", "pre[:, k] = perf_all[:, p]", "pre[:, k] = 0.0", "пререквизит не влияет"),
    ("edurisk/world.py", "- 0.35 * diff[None, :]", "+ 0.35 * diff[None, :]", "знак сложности курса"),
    ("edurisk/world.py", "f = f - cfg.feedback_jump * hit2.any(1) - 0.03 * failed.sum(1)", "f = f - cfg.feedback_linear * failed.sum(1)", "S1 без порогового скачка"),
    ("edurisk/world.py", " * login_factor[:, None]", "", "S3 без смещения измерения"),
    ("edurisk/world.py", "perf_shift = cfg.s2_perf_shift", "perf_shift = 0.0", "S2 без сдвига итога"),
    ("edurisk/world.py", "logins[lost_l] = np.nan", "pass", "S4 не теряет логины"),
    ("edurisk/world.py", "(enr & ~self.passed[:, idx]).sum(1)", "(~self.passed[:, idx]).sum(1)", "в историю попали не учившиеся"),
    ("edurisk/world.py", "cur.semester.to_numpy() < semester)", "cur.semester.to_numpy() <= semester)", "утечка: история включает текущий семестр"),
    ("edurisk/world.py", "outcome=~self.passed[m, c]", "outcome=self.passed[m, c]", "перевёрнута метка"),
    ("edurisk/curriculum.py", "& (cur.form != \"coursework\").to_numpy()", "", "курсовая может быть пререквизитом"),
    ("edurisk/store.py", "WHERE course_id = ? AND week <= ?", "WHERE course_id = ? AND week <= ? + 1", "SQL-срез захватывает следующую неделю"),
    ("edurisk/store.py", "int(w.dropped[s, c]), int(w.admitted[s, c])", "int(w.admitted[s, c]), int(w.dropped[s, c])", "перепутаны dropped и admitted"),
    ("edurisk/store.py", "(ids[s], int(cur.index[c]), t + 1, _n(w.logins[s, c, t])", "(ids[s], int(cur.index[c]), t, _n(w.logins[s, c, t])", "неделя в БД с нуля"),
    ("edurisk/store.py", "None if r.prereq < 0 else int(r.prereq)", "int(r.prereq)", "пререквизит -1 пишется как курс"),
    ("edurisk/store.py", "logins[r, c, t] = act.logins.astype(float)", "logins[r, c, t] = act.minutes.astype(float)", "перепутаны логины и минуты при чтении"),
    ("edurisk/store.py", "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY", "BIGINT PRIMARY KEY", "журнал без автонумерации в PostgreSQL"),
    ("edurisk/oulad.py", "v[\"week\"] = v.date // 7", "v[\"week\"] = v.date // 7 + 1", "OULAD: неделя клика сдвинута"),
    ("edurisk/oulad.py", "(s.date_submitted <= s.date)", "(s.date_submitted <= s.date + 10)", "OULAD: опоздавшие засчитываются"),
    ("edurisk/oulad.py", "(s.is_banked == 0)", "(s.is_banked >= 0)", "OULAD: перенесённые баллы засчитываются"),
    ("edurisk/oulad.py", "admitted[r, c] = ~wd", "admitted[r, c] = True", "OULAD: отчисленные допущены"),
    ("edurisk/oulad.py", "days = v.groupby([\"r\", \"c\", \"week\"]).date.nunique()", "days = v.groupby([\"r\", \"c\", \"week\"]).date.count()", "OULAD: дни активности считаются по строкам"),
    ("edurisk/oulad.py", "asm.assessment_type.isin([\"TMA\", \"CMA\"])", "asm.assessment_type.isin([\"TMA\", \"CMA\", \"Exam\"])", "OULAD: экзамен стал контрольной точкой"),
    ("edurisk/costs.py", "return c_fa / (c_fa + k)", "return k / (c_fa + k)", "порог решения перевёрнут"),
    ("edurisk/costs.py", "return (ze / m) ** gamma", "return (ze / m) ** (gamma + 1)", "степень ЗЕТ-веса"),
    ("edurisk/costs.py", "return float(np.mean(w * (k * (y & ~flag) + c_fa * (~y & flag))))", "return float(np.mean(w * (c_fa * (y & ~flag) + k * (~y & flag))))", "стоимости ошибок перепутаны"),
    ("edurisk/costs.py", "np.argsort(-(p * w * k), kind=\"stable\")", "np.argsort(-(p * k), kind=\"stable\")", "приоритет без ЗЕТ-веса"),
    ("edurisk/costs.py", "np.argsort(-(p * w * k), kind=\"stable\")", "np.argsort((p * w * k), kind=\"stable\")", "приоритет по возрастанию"),
    ("edurisk/metrics.py", "run = max(run, (m - rank) * p[i])", "run = (m - rank) * p[i]", "Холм без монотонности"),
    ("edurisk/metrics.py", "tot += m.sum() * abs(y[m].mean() - p[m].mean())", "tot += abs(y[m].mean() - p[m].mean())", "ECE без весов бинов"),
    ("edurisk/metrics.py", "idx = np.minimum((p * bins).astype(int), bins - 1)", "idx = (p * bins).astype(int) % bins", "ECE: прогноз 1.0 в нулевой бин"),
    ("edurisk/dataset.py", "\"val\": set(ids[a:b].tolist()), \"test\": set(ids[b:].tolist())", "\"val\": set(ids[a - 5:b].tolist()), \"test\": set(ids[b:].tolist())", "студенты пересекаются между train и val"),
    ("edurisk/dataset.py", "h = hist[sem].loc[X.index, HISTORY]", "h = hist[sem + 1 if sem + 1 in hist else sem].loc[X.index, HISTORY]", "история захватывает текущий семестр"),
    ("edurisk/dataset.py", "X[\"y\"] = coh.outcome.astype(int)", "X[\"y\"] = (~coh.outcome).astype(int)", "метка перевёрнута"),
    ("edurisk/models.py", "return (X[\"cp_max_fail_streak\"].to_numpy() >= 2) | (", "return (X[\"cp_max_fail_streak\"].to_numpy() >= 3) | (", "правило: серия 3 вместо 2"),
    ("edurisk/models.py", "return self.iso_.predict(self.base.predict_proba(X))", "return self.base.predict_proba(X)", "калибровка не применяется"),
    ("edurisk/experiment.py", "Calibrated(boost_cs).fit_calibration(val, yva)", "Calibrated(boost_cs).fit_calibration(train, ytr)", "калибровка на обучающей выборке"),
    ("edurisk/experiment.py", "cs = wz * np.where(ytr == 1, k, 1.0)", "cs = wz", "обучение без веса класса"),
    ("edurisk/ordinal.py", "d = GRADES[None, :] - GRADES[:, None]", "d = GRADES[:, None] - GRADES[None, :]", "направление асимметрии стоимости", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "return GRADES[np.argmin(P @ M, axis=1)]", "return GRADES[np.argmax(P @ M, axis=1)]", "решение по максимуму стоимости", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "C = np.minimum.accumulate(", "C = np.maximum.accumulate(", "монотонность каскада наоборот", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "return np.column_stack([1 - C[:, 0], C[:, 0] - C[:, 1], C[:, 1] - C[:, 2], C[:, 2]])", "return np.column_stack([1 - C[:, 0], C[:, 0] - C[:, 1], C[:, 1] - C[:, 2], 1 - C[:, 2]])", "вероятности классов не суммируются в 1", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "CUTS = (3, 4, 5)", "CUTS = (3, 4, 4)", "порог «= 5» заменён", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "out[:, np.searchsorted(GRADES, self.m.classes_)] = P", "out[:, :P.shape[1]] = P", "классы многоклассовой модели без выравнивания", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "np.mean(pred[y_grade == 2] == 2)", "np.mean(pred[y_grade == 3] == 3)", "recall класса 2 считается по тройкам", "tests.test_ordinal"),
    ("edurisk/ordinal.py", "np.asarray(w, float) * M[iy, ip]", "np.asarray(w, float) * M[ip, iy]", "матрица стоимости транспонирована", "tests.test_ordinal"),
    ("edurisk/conformal.py", "k = int(np.ceil((n + 1) * (1 - alpha)))", "k = int(np.ceil(n * (1 - alpha)))", "конформный квантиль без поправки n+1", "tests.test_ordinal"),
    ("edurisk/conformal.py", "return (1.0 - P) <= np.broadcast_to(q, (P.shape[1],))[None, :]", "return (1.0 - P) < np.broadcast_to(q, (P.shape[1],))[None, :]", "граница набора строгая", "tests.test_ordinal"),
    ("edurisk/conformal.py", "s = 1.0 - P_cal[np.arange(len(y_idx)), y_idx]", "s = P_cal[np.arange(len(y_idx)), y_idx]", "нонконформность без инверсии", "tests.test_ordinal"),
    ("edurisk/conformal.py", "return np.array([_q(s[y_idx == c], alpha) for c in range(P_cal.shape[1])])", "return np.array([_q(s, alpha) for c in range(P_cal.shape[1])])", "пороги по классам берутся по всем данным", "tests.test_ordinal"),
    ("edurisk/dataset.py", "out[\"admitted\"] = out[\"admitted\"].astype(int)", "out[\"admitted\"] = 1 - out[\"admitted\"].astype(int)", "допуск перевёрнут", "tests.test_ordinal"),
    ("edurisk/experiment5.py", "ytr, yva = 1 - train.admitted.to_numpy(), 1 - val.admitted.to_numpy()", "ytr, yva = train.admitted.to_numpy(), val.admitted.to_numpy()", "T3: метка допуска вместо недопуска", "tests.test_ordinal"),
    ("edurisk/experiment5.py", "cf.calibrate(Pv, yv, alpha, cw)", "cf.calibrate(Pt, yt, alpha, cw)", "конформный порог по тестовой выборке", "tests.test_ordinal"),
    ("edurisk/experiment5.py", "LogitModel()).fit(train, g).fit_calibration(val, gv)", "LogitModel()).fit(train, g).fit_calibration(train, g)", "калибровка каскада на обучении", "tests.test_ordinal"),
    ("edurisk/experiment5.py", "return d[d.grade.notna()].assign(", "return d[d.grade.notna() | True].assign(", "в T2 попали зачёты", "tests.test_ordinal"),
    ("edurisk/explain.py", "w = 1.0 / (n * comb(n - 1, k))", "w = 1.0 / n", "веса Шепли без биномиального коэффициента", "tests.test_explain"),
    ("edurisk/explain.py", "swap = [c for g in names if g not in S for c in groups[g]]", "swap = [c for g in names if g in S for c in groups[g]]", "подстановка фона в обратную сторону", "tests.test_explain"),
    ("edurisk/explain.py", "out.iloc[:, j] += C[:, n_base + k]", "out.iloc[:, j] -= C[:, n_base + k]", "индикатор пропуска вычитается из вклада", "tests.test_explain"),
    ("edurisk/explain.py", "r[c] = background[c].median()", "r[c] = background[c].max()", "что-если подставляет максимум", "tests.test_explain"),
    ("edurisk/explain.py", "out[g] = float(base - metric(y, predict(Xp)))", "out[g] = float(metric(y, predict(Xp)) - base)", "знак групповой важности", "tests.test_explain"),
    ("edurisk/explain.py", "if v <= 0.01:", "if v >= 0.01:", "в текст попадают факторы понижения", "tests.test_explain"),
    ("edurisk/explain.py", "ranked = sorted(((v, g) for g, v in contrib.items() if not g.startswith(\"_\")), reverse=True)", "ranked = sorted(((v, g) for g, v in contrib.items() if not g.startswith(\"_\")), reverse=False)", "факторы в тексте по возрастанию", "tests.test_explain"),
    ("edurisk/fairness.py", "tpr = float(flag[y].mean()) if y.any() else float(\"nan\")", "tpr = float(flag[~y].mean()) if y.any() else float(\"nan\")", "TPR считается по отрицательным", "tests.test_fairness"),
    ("edurisk/fairness.py", "k = int(np.floor((1 - target_fpr) * len(neg)))", "k = int(np.ceil((1 - target_fpr) * len(neg)))", "порог выравнивания FPR: округление вверх", "tests.test_fairness"),
    ("edurisk/fairness.py", "out[g] = float(np.nextafter(neg[k], 1.0))", "out[g] = float(neg[k])", "порог не строго выше отрицательного", "tests.test_fairness"),
    ("edurisk/privacy.py", "\"teacher\": {\"view_course_risk\", \"view_student_card\", \"mark_disagree\"}", "\"teacher\": {\"view_course_risk\", \"view_student_card\", \"mark_disagree\", \"view_audit\"}", "преподаватель видит журнал", "tests.test_fairness"),
    ("edurisk/privacy.py", "return action in ROLES.get(role, set())", "return role in ROLES", "любое действие разрешено известной роли", "tests.test_fairness"),
    ("edurisk/privacy.py", "return int(df.groupby(quasi, dropna=False).size().min())", "return int(df.groupby(quasi, dropna=False).size().max())", "k-анонимность по максимуму", "tests.test_fairness"),
    ("edurisk/privacy.py", "out = df.drop(columns=[c for c in drop if c in df.columns]).copy()", "out = df.copy()", "чувствительные столбцы не удаляются", "tests.test_fairness"),
    ("edurisk/service.py", "if d[\"exp\"] < (now or time.time()):", "if d[\"exp\"] > (now or time.time()):", "срок токена проверяется наоборот", "tests.test_app"),
    ("edurisk/service.py", "if not hmac.compare_digest(sig, good):", "if False:", "подпись токена не проверяется", "tests.test_app"),
    ("edurisk/service.py", "LEVELS = ((0.5, \"высокая\"), (0.25, \"повышенная\"), (0.0, \"обычная\"))", "LEVELS = ((0.6, \"высокая\"), (0.25, \"повышенная\"), (0.0, \"обычная\"))", "порог высокой зоны", "tests.test_app"),
    ("edurisk/service.py", "d[\"priority\"] = d.p * d.w * k", "d[\"priority\"] = d.p * k", "приоритет без ЗЕТ-веса", "tests.test_app"),
    ("edurisk/service.py", "sort_values(\"priority\", ascending=False)", "sort_values(\"priority\", ascending=True)", "список риска по возрастанию", "tests.test_app"),
    ("edurisk/service.py", "\"comment\": comment[:500]", "\"comment\": comment[:5000]", "комментарий не ограничен 500 символами", "tests.test_app"),
    ("edurisk/app.py", "return course in svc.allowed_courses(user)", "return True", "преподаватель видит чужие дисциплины", "tests.test_app"),
    ("edurisk/app.py", "if not hmac.compare_digest(form.get(\"csrf\", [\"\"])[0], csrf_of(tok, svc.secret)):", "if False:", "CSRF-токен не проверяется", "tests.test_app"),
    ("edurisk/app.py", "            svc.log_view(user, course, pseudo)\n", "            pass\n", "просмотр карточки не пишется в журнал", "tests.test_app"),
    ("edurisk/app.py", "Environment(loader=DictLoader(_T), autoescape=True)", "Environment(loader=DictLoader(_T), autoescape=False)", "HTML не экранируется", "tests.test_app"),
    ("edurisk/app.py", "(\"Cache-Control\", \"no-store\")", "(\"Cache-Control\", \"public\")", "карточки кэшируются", "tests.test_app"),
    ("edurisk/app.py", "if not can(role, \"view_audit\"):", "if not can(role, \"view_aggregates\"):", "журнал доступен роли с агрегатами", "tests.test_app"),
]


def main() -> int:
    base = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, capture_output=True)
    if base.returncode != 0:
        print("Исходные тесты не проходят:\n", base.stderr.decode()[-800:])
        return 2
    survived = []
    for rel, old, new, what, *only in MUTATIONS:
        targets = only[0] if only else None
        if "--targeted" in sys.argv and not targets:
            continue
        path = ROOT / rel
        src = path.read_text(encoding="utf-8")
        if src.count(old) != 1:
            print(f"ПРОПУЩЕНО (фрагмент найден {src.count(old)} раз): {what}")
            survived.append(what + " [не применена]")
            continue
        try:
            path.write_text(src.replace(old, new), encoding="utf-8")
            cmd = [sys.executable, "-m", "unittest"] + ([targets] if targets else ["discover", "-s", "tests"])
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True)
        finally:
            path.write_text(src, encoding="utf-8")
        if r.returncode == 0:
            survived.append(what)
        print(("ВЫЖИЛА   " if r.returncode == 0 else "поймана   ") + what)
    total = sum(1 for m in MUTATIONS if len(m) > 4) if "--targeted" in sys.argv else len(MUTATIONS)
    print(f"\nвсего {total}, выжило {len(survived)}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
