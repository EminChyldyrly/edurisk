"""Веб-интерфейс преподавателя на WSGI (стандартная библиотека + jinja2, без JavaScript и сборки).
Графики — встроенный SVG. Маршруты:
  GET  /health
  GET  /login?token=...                   демо-вход: кладёт токен в cookie и ведёт на первую страницу
  GET  /api/courses/<c>/risk.json        список риска (JSON)
  GET  /courses/<c>                       список риска дисциплины
  GET  /courses/<c>/students/<псевдоним>  карточка студента (просмотр пишется в журнал)
  POST /courses/<c>/students/<псевдоним>/disagree   «не согласен с оценкой модели»
  GET  /audit                             журнал (только admin)
Доступ: подписанный токен в заголовке Authorization: Bearer ... или cookie `token`; проверка роли и дисциплины.
Это прототип: в продакшене вход через внешнего провайдера (OAuth2) и ASGI-фреймворк (FastAPI); логика в service.py
от веб-слоя не зависит.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from http.cookies import SimpleCookie
from urllib.parse import parse_qs

from jinja2 import DictLoader, Environment

from .explain import GROUP_RU
from .privacy import can
from .service import RiskService, read_token

_T = {
"base": """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d2330;background:#fafbfc}
table{border-collapse:collapse;width:100%}th,td{padding:6px 10px;border-bottom:1px solid #dde2ea;text-align:left}
.tag{padding:2px 8px;border-radius:10px;font-size:13px}.l-высокая{background:#fde2e0}.l-повышенная{background:#fdf0d3}.l-обычная{background:#e3f2e6}
.note{color:#5b6475;font-size:13px}.card{background:#fff;border:1px solid #dde2ea;border-radius:8px;padding:14px 18px;margin:14px 0}
</style><body><h1>{{ title }}</h1>{% block body %}{% endblock %}
<p class="note">Оценка модели — подсказка для внимания, а не вывод о студенте. Решение остаётся за преподавателем.</p></body></html>""",
"course": """{% extends "base" %}{% block body %}
<p class="note">Неделя {{ week }}. Студенты упорядочены по приоритету (вероятность × З.Е.-вес × стоимость пропуска). Показаны псевдонимы.</p>
<table><tr><th>Псевдоним</th><th>Вероятность</th><th>Зона внимания</th><th>Подряд не сдано КТ</th><th>Нед. без входа</th></tr>
{% for r in rows %}<tr><td><a href="/courses/{{ course }}/students/{{ r.pseudo }}">{{ r.pseudo }}</a></td>
<td>{{ "%.0f%%"|format([r.p*100, 99]|min) }}</td><td><span class="tag l-{{ r.level }}">{{ r.level }}</span></td>
<td>{{ r.cp_max_fail_streak|int }}</td><td>{{ r.weeks_since_login|int }}</td></tr>{% endfor %}</table>{% endblock %}""",
"card": """{% extends "base" %}{% block body %}
<p><a href="/courses/{{ c.course }}">← к списку</a> · {{ c.course_name }} · студент {{ c.pseudo }}</p>
<div class="card"><b>{{ "%.0f%%"|format([c.p*100, 99]|min) }}</b> <span class="tag l-{{ c.level }}">{{ c.level }}</span><p>{{ c.text }}</p></div>
<div class="card"><b>Входы в LMS по неделям</b><br>{{ logins_svg|safe }}</div>
<div class="card"><b>Контрольные точки</b> (линия — проходной балл {{ c.pass_score|int }})<br>{{ cp_svg|safe }}</div>
<div class="card"><b>Вклад факторов в прогноз</b> (доли вероятности; справа — повышает, слева — понижает)<br>{{ contrib_svg|safe }}
<p class="note">Что если: при обычной активности {{ "%.0f%%"|format(c.what_if.activity*100) }}, при обычных результатах контрольных {{ "%.0f%%"|format(c.what_if.assessments*100) }}. Это сценарий о модели, а не обещание эффекта вмешательства.</p></div>
<form method="post" action="/courses/{{ c.course }}/students/{{ c.pseudo }}/disagree" class="card">
<b>Не согласен с оценкой модели</b><br><input type="hidden" name="csrf" value="{{ csrf }}">
<textarea name="comment" rows="3" style="width:100%" maxlength="500" placeholder="Что модель не учитывает?"></textarea><br><button>Отправить</button></form>{% endblock %}""",
"audit": """{% extends "base" %}{% block body %}<table><tr><th>№</th><th>Время (UTC)</th><th>Кто</th><th>Действие</th><th>Студент</th><th>Детали</th></tr>
{% for r in rows %}<tr>{% for x in r %}<td>{{ x if x is not none else "" }}</td>{% endfor %}</tr>{% endfor %}</table>{% endblock %}""",
"msg": """{% extends "base" %}{% block body %}<p>{{ text }}</p>{% endblock %}""",
}
_env = Environment(loader=DictLoader(_T), autoescape=True)


def _bars(values, labels=None, w=640, h=90, color="#4a78c2") -> str:
    n = max(len(values), 1)
    m = max(max(values, default=0), 1)
    bw = w / n
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 18}" role="img" aria-label="столбчатая диаграмма">']
    for i, v in enumerate(values):
        bh = h * v / m
        s.append(f'<rect x="{i * bw + 2:.1f}" y="{h - bh:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" fill="{color}"><title>неделя {i + 1}: {v:g}</title></rect>')
        if n <= 20:
            s.append(f'<text x="{i * bw + bw / 2:.1f}" y="{h + 14}" font-size="10" text-anchor="middle" fill="#5b6475">{i + 1}</text>')
    s.append("</svg>")
    return "".join(s)


def _cp_svg(cp, pass_score, w=640, h=100) -> str:
    n = max(len(cp), 1)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h + 18}" role="img" aria-label="контрольные точки">']
    ypass = h - h * pass_score / 100
    s.append(f'<line x1="0" x2="{w}" y1="{ypass:.1f}" y2="{ypass:.1f}" stroke="#c2574a" stroke-dasharray="4 3"/>')
    for i, (week, sc) in enumerate(cp):
        x = (i + 0.5) * w / n
        if sc is None:
            s.append(f'<text x="{x:.1f}" y="{h / 2:.0f}" text-anchor="middle" fill="#c2574a" font-size="12">нет балла</text>')
        else:
            y = h - h * sc / 100
            col = "#4a9c5d" if sc >= pass_score else "#c2574a"
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{col}"><title>неделя {week}: {sc:.0f}</title></circle>')
        s.append(f'<text x="{x:.1f}" y="{h + 14}" font-size="10" text-anchor="middle" fill="#5b6475">нед. {week}</text>')
    s.append("</svg>")
    return "".join(s)


def _contrib_svg(contrib, w=640) -> str:
    items = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    m = max((abs(v) for _, v in items), default=0) or 1
    row, mid = 24, w * 0.55
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{row * len(items) + 4}" role="img" aria-label="вклады факторов">',
         f'<line x1="{mid}" x2="{mid}" y1="0" y2="{row * len(items)}" stroke="#9aa3b2"/>']
    for i, (g, v) in enumerate(items):
        ln = (w * 0.4) * abs(v) / m
        x = mid if v >= 0 else mid - ln
        col = "#c2574a" if v >= 0 else "#4a9c5d"
        s.append(f'<text x="4" y="{i * row + 16}" font-size="13">{GROUP_RU[g]}</text>')
        s.append(f'<rect x="{x:.1f}" y="{i * row + 4}" width="{max(ln, 1):.1f}" height="16" fill="{col}"><title>{v:+.3f}</title></rect>')
    s.append("</svg>")
    return "".join(s)


def csrf_of(token: str, secret: bytes) -> str:
    return hmac.new(secret, ("csrf:" + token).encode(), hashlib.sha256).hexdigest()[:32]


def make_app(svc: RiskService):
    routes = [
        ("GET", re.compile(r"^/health$"), "health"),
        ("GET", re.compile(r"^/login$"), "login"),
        ("GET", re.compile(r"^/api/courses/(\d+)/risk\.json$"), "api_risk"),
        ("GET", re.compile(r"^/courses/(\d+)$"), "course"),
        ("GET", re.compile(r"^/courses/(\d+)/students/([0-9a-f]{12})$"), "card"),
        ("POST", re.compile(r"^/courses/(\d+)/students/([0-9a-f]{12})/disagree$"), "disagree"),
        ("GET", re.compile(r"^/audit$"), "audit"),
    ]

    def render(name, status="200 OK", **ctx):
        body = _env.get_template(name).render(week=svc.week, **ctx).encode()
        return status, [("Content-Type", "text/html; charset=utf-8"), ("X-Content-Type-Options", "nosniff"),
                        ("Cache-Control", "no-store")], body

    def msg(status, text):
        return render("msg", status, title=status, text=text)

    def token_of(environ):
        h = environ.get("HTTP_AUTHORIZATION", "")
        if h.startswith("Bearer "):
            return h[7:]
        c = SimpleCookie(environ.get("HTTP_COOKIE", ""))
        return c["token"].value if "token" in c else ""

    def app(environ, start_response):
        method, path = environ["REQUEST_METHOD"], environ["PATH_INFO"]
        hit = next(((n, m) for meth, rx, n in routes if meth == method and (m := rx.match(path))), None)
        if hit is None:
            known = any(rx.match(path) for _, rx, _ in routes)
            status, hdr, body = msg("405 Method Not Allowed" if known else "404 Not Found", "Нет такой страницы.")
        else:
            name, m = hit
            status, hdr, body = _dispatch(name, m, environ)
        start_response(status, hdr)
        return [body]

    def _dispatch(name, m, environ):
        if name == "health":
            return "200 OK", [("Content-Type", "text/plain")], b"ok"
        if name == "login":     # демо-вход по ссылке: токен из адреса кладётся в cookie (в продакшене вместо этого OAuth2)
            t = parse_qs(environ.get("QUERY_STRING", "")).get("token", [""])[0]
            who = read_token(t, svc.secret)
            if who is None:
                return msg("401 Unauthorized", "Ссылка входа недействительна или устарела.")
            first = "/audit" if who[1] == "admin" else (f"/courses/{svc.allowed_courses(who[0])[0]}" if svc.allowed_courses(who[0]) else "/courses/0")
            return "302 Found", [("Location", first), ("Content-Type", "text/plain"),
                                 ("Set-Cookie", f"token={t}; Path=/; HttpOnly; SameSite=Strict")], b""
        tok = token_of(environ)
        who = read_token(tok, svc.secret)
        if who is None:
            return msg("401 Unauthorized", "Нужен действующий вход.")
        user, role = who
        course = int(m.group(1)) if m.groups() else None

        def course_ok(action):
            if not can(role, action):
                return False
            if role == "teacher":
                return course in svc.allowed_courses(user)
            return True

        if name == "audit":
            if not can(role, "view_audit"):
                return msg("403 Forbidden", "Недостаточно прав.")
            return render("audit", title="Журнал действий", rows=svc.audit())
        if course not in range(svc.world.n_courses):
            return msg("404 Not Found", "Нет такой дисциплины.")
        if name in ("api_risk", "course"):
            if not course_ok("view_course_risk"):
                return msg("403 Forbidden", "Недостаточно прав.")
            q = parse_qs(environ.get("QUERY_STRING", ""))
            try:
                top = int(q["top"][0]) if "top" in q else None
            except ValueError:
                return msg("400 Bad Request", "top должен быть числом.")
            df = svc.risk_list(course, top)
            if name == "api_risk":
                return "200 OK", [("Content-Type", "application/json; charset=utf-8")], json.dumps(
                    df.round(4).to_dict("records"), ensure_ascii=False).encode()
            return render("course", title=svc.course_name(course), course=course, rows=df.itertuples())
        pseudo = m.group(2)
        if not course_ok("view_student_card" if name == "card" else "mark_disagree"):
            return msg("403 Forbidden", "Недостаточно прав.")
        if name == "card":
            try:
                c = svc.card(course, pseudo)
            except KeyError:
                return msg("404 Not Found", "Студент не найден.")
            svc.log_view(user, course, pseudo)
            return render("card", title="Карточка студента", c=c, csrf=csrf_of(tok, svc.secret),
                          logins_svg=_bars(c["logins"]), cp_svg=_cp_svg(c["cp"], c["pass_score"]),
                          contrib_svg=_contrib_svg(c["contrib"]))
        # disagree
        try:
            n = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            n = 0
        form = parse_qs(environ["wsgi.input"].read(min(n, 4096)).decode("utf-8", "replace"))
        if not hmac.compare_digest(form.get("csrf", [""])[0], csrf_of(tok, svc.secret)):
            return msg("403 Forbidden", "Неверный токен формы.")
        try:
            svc.disagree(user, course, pseudo, form.get("comment", [""])[0])
        except KeyError:
            return msg("404 Not Found", "Студент не найден.")
        return msg("200 OK", "Отметка записана в журнал. Спасибо.")

    return app
