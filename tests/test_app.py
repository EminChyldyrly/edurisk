import io
import json
import unittest
from urllib.parse import urlencode

import numpy as np

from edurisk.app import csrf_of, make_app
from edurisk.service import build_demo_service, level_of, make_token, read_token

SECRET = b"s3cret"


class TokenTests(unittest.TestCase):
    def test_roundtrip_tamper_expiry(self):
        t = make_token("u1", "teacher", SECRET, ttl=100, now=1000)
        self.assertEqual(read_token(t, SECRET, now=1050), ("u1", "teacher"))
        self.assertIsNone(read_token(t, SECRET, now=1101))                     # истёк
        self.assertIsNone(read_token(t, b"other", now=1050))                   # чужой секрет
        payload, sig = t.rsplit(".", 1)
        forged = make_token("u1", "admin", SECRET, ttl=100, now=1000).split(".")[0] + "." + sig
        self.assertIsNone(read_token(forged, SECRET, now=1050))                # роль подменена
        for bad in ("", "abc", "a.b", "....", None):
            self.assertIsNone(read_token(bad, SECRET) if bad is not None else None)

    def test_levels(self):
        self.assertEqual(level_of(0.0), "обычная")
        self.assertEqual(level_of(0.2499), "обычная")
        self.assertEqual(level_of(0.25), "повышенная")
        self.assertEqual(level_of(0.5), "высокая")
        self.assertEqual(level_of(1.0), "высокая")


class ServiceAndAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = build_demo_service(seed=3, n_train=1200, n_serve=300)
        cls.app = staticmethod(make_app(cls.svc))
        s = cls.svc.secret
        cls.tok = {u: make_token(u, r, s) for u, r in (("teacher1", "teacher"), ("teacher2", "teacher"),
                                                     ("m1", "methodist"), ("a1", "admin"))}

    def call(self, method, path, token=None, body=None, query=""):
        env = {"REQUEST_METHOD": method, "PATH_INFO": path, "QUERY_STRING": query,
               "wsgi.input": io.BytesIO(body or b""), "CONTENT_LENGTH": str(len(body or b""))}
        if token:
            env["HTTP_AUTHORIZATION"] = "Bearer " + token
        out = {}
        data = self.app(env, lambda st, hd: out.update(status=st, headers=dict(hd)))
        return out["status"], out["headers"], b"".join(data).decode("utf-8")

    # --- сервис
    def test_risk_list_sorted_and_priority_formula(self):
        df = self.svc.risk_list(0)
        self.assertTrue(df.priority.is_monotonic_decreasing)
        self.assertEqual(df.pseudo.nunique(), len(df))
        d = self.svc.data
        r = d[d.course == 0].sort_values("priority", ascending=False).iloc[0]
        self.assertAlmostEqual(r.priority, r.p * r.w * 3.0)
        self.assertEqual(len(df), self.svc.world.enrolled[:, 0].sum())
        self.assertEqual(len(self.svc.risk_list(0, top=5)), 5)

    def test_card_efficiency_and_content(self):
        pseudo = self.svc.risk_list(0).pseudo.iloc[0]
        c = self.svc.card(0, pseudo)
        self.assertEqual(len(c["logins"]), self.svc.week)
        self.assertEqual(set(c["contrib"]), {"activity", "assessments", "history", "background", "course"})
        self.assertIn("зона внимания", c["text"])
        self.assertTrue(0 <= c["p"] <= 1)
        with self.assertRaises(KeyError):
            self.svc.card(0, "0" * 12)
        # студент, не учившийся на дисциплине
        not_enr = np.flatnonzero(~self.svc.world.enrolled[:, 0])
        if len(not_enr):
            from edurisk.privacy import pseudonym
            with self.assertRaises(KeyError):
                self.svc.card(0, pseudonym(int(not_enr[0]), self.svc.secret))

    # --- веб-слой
    def test_auth_and_roles(self):
        self.assertEqual(self.call("GET", "/health")[0], "200 OK")
        self.assertEqual(self.call("GET", "/courses/0")[0], "401 Unauthorized")
        self.assertEqual(self.call("GET", "/courses/0", token="garbage")[0], "401 Unauthorized")
        self.assertEqual(self.call("GET", "/courses/0", self.tok["teacher1"])[0], "200 OK")
        self.assertEqual(self.call("GET", "/courses/0", self.tok["teacher2"])[0], "403 Forbidden")   # чужая дисциплина
        self.assertEqual(self.call("GET", "/courses/0", self.tok["m1"])[0], "200 OK")
        self.assertEqual(self.call("GET", "/courses/0", self.tok["a1"])[0], "403 Forbidden")
        self.assertEqual(self.call("GET", "/audit", self.tok["teacher1"])[0], "403 Forbidden")
        self.assertEqual(self.call("GET", "/audit", self.tok["m1"])[0], "403 Forbidden")
        self.assertEqual(self.call("GET", "/audit", self.tok["a1"])[0], "200 OK")
        pseudo = self.svc.risk_list(0).pseudo.iloc[0]
        self.assertEqual(self.call("GET", f"/courses/0/students/{pseudo}", self.tok["m1"])[0], "403 Forbidden")
        self.assertEqual(self.call("GET", f"/courses/0/students/{pseudo}", self.tok["a1"])[0], "403 Forbidden")

    def test_routes_and_validation(self):
        t = self.tok["teacher1"]
        self.assertEqual(self.call("GET", "/nope", t)[0], "404 Not Found")
        self.assertEqual(self.call("POST", "/courses/0", t)[0], "405 Method Not Allowed")
        self.assertEqual(self.call("GET", "/courses/999", t)[0], "404 Not Found")
        self.assertEqual(self.call("GET", "/api/courses/0/risk.json", t, query="top=abc")[0], "400 Bad Request")
        st, hd, body = self.call("GET", "/api/courses/0/risk.json", t, query="top=3")
        rows = json.loads(body)
        self.assertEqual((st, len(rows)), ("200 OK", 3))
        self.assertGreaterEqual(rows[0]["priority"], rows[2]["priority"])
        self.assertNotIn("student_id", rows[0])                        # наружу только псевдонимы
        self.assertEqual(self.call("GET", "/courses/0/students/" + "0" * 12, t)[0], "404 Not Found")
        st, hd, _ = self.call("GET", "/courses/0", t)
        self.assertEqual(hd["X-Content-Type-Options"], "nosniff")
        self.assertEqual(hd["Cache-Control"], "no-store")

    def test_card_page_logs_view_and_is_neutral(self):
        t = self.tok["teacher1"]
        pseudo = self.svc.risk_list(0).pseudo.iloc[0]
        before = len(self.svc.audit())
        st, _, html = self.call("GET", f"/courses/0/students/{pseudo}", t)
        self.assertEqual(st, "200 OK")
        for part in ("<svg", "зона внимания", "Не согласен с оценкой модели", "решение остаётся за преподавателем", pseudo):
            self.assertIn(part, html)
        self.assertNotIn("двоечник", html.lower())
        rows = self.svc.audit()
        self.assertEqual(len(rows), before + 1)
        self.assertEqual((rows[0][2], rows[0][3]), ("teacher1", "view_card"))

    def test_disagree_flow_csrf_and_escaping(self):
        t = self.tok["teacher1"]
        pseudo = self.svc.risk_list(0).pseudo.iloc[1]
        path = f"/courses/0/students/{pseudo}/disagree"
        n0 = len(self.svc.audit())
        bad = urlencode({"csrf": "x", "comment": "нет"}).encode()
        self.assertEqual(self.call("POST", path, t, bad)[0], "403 Forbidden")
        self.assertEqual(len(self.svc.audit()), n0)                    # без токена формы ничего не записано
        evil = "<script>alert(1)</script> болел"
        ok = urlencode({"csrf": csrf_of(t, self.svc.secret), "comment": evil}).encode()
        self.assertEqual(self.call("POST", path, t, ok)[0], "200 OK")
        rows = self.svc.audit()
        self.assertEqual(len(rows), n0 + 1)
        self.assertEqual(rows[0][3], "disagree")
        self.assertIn("болел", json.loads(rows[0][5])["comment"])
        _, _, html = self.call("GET", "/audit", self.tok["a1"])
        self.assertNotIn("<script>alert", html)                         # экранируется
        self.assertIn("&lt;script&gt;", html)
        # чужой преподаватель не может оставить отметку; csrf привязан к токену
        t2 = self.tok["teacher2"]
        ok2 = urlencode({"csrf": csrf_of(t2, self.svc.secret), "comment": "x"}).encode()
        self.assertEqual(self.call("POST", path, t2, ok2)[0], "403 Forbidden")
        self.assertEqual(self.call("POST", path, t, urlencode({"csrf": csrf_of(t2, self.svc.secret)}).encode())[0], "403 Forbidden")

    def test_demo_login_sets_cookie_and_cookie_authenticates(self):
        st, hd, _ = self.call("GET", "/login", query="token=" + self.tok["teacher1"])
        self.assertEqual(st, "302 Found")
        self.assertEqual(hd["Location"], "/courses/0")
        self.assertIn("HttpOnly", hd["Set-Cookie"])
        self.assertIn("SameSite=Strict", hd["Set-Cookie"])
        cookie = hd["Set-Cookie"].split(";")[0]
        env = {"REQUEST_METHOD": "GET", "PATH_INFO": "/courses/0", "QUERY_STRING": "", "HTTP_COOKIE": cookie,
               "wsgi.input": io.BytesIO(b""), "CONTENT_LENGTH": "0"}
        out = {}
        self.app(env, lambda st_, hd_: out.update(status=st_))
        self.assertEqual(out["status"], "200 OK")
        self.assertEqual(self.call("GET", "/login", query="token=bad")[0], "401 Unauthorized")
        self.assertEqual(self.call("GET", "/login")[0], "401 Unauthorized")
        self.assertEqual(self.call("GET", "/login", query="token=" + self.tok["a1"])[1]["Location"], "/audit")
        self.assertEqual(self.call("GET", "/login", query="token=" + self.tok["teacher2"])[1]["Location"], "/courses/3")

    def test_comment_is_truncated(self):
        t = self.tok["teacher1"]
        pseudo = self.svc.risk_list(0).pseudo.iloc[2]
        body = urlencode({"csrf": csrf_of(t, self.svc.secret), "comment": "я" * 3000}).encode()
        self.call("POST", f"/courses/0/students/{pseudo}/disagree", t, body)
        self.assertLessEqual(len(json.loads(self.svc.audit()[0][5])["comment"]), 500)


if __name__ == "__main__":
    unittest.main()
