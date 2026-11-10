"""Демо-сервер: python scripts/run_demo.py  -> http://127.0.0.1:8000 (токены печатаются в консоль)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wsgiref.simple_server import make_server
from edurisk.app import make_app
from edurisk.service import build_demo_service, make_token

svc = build_demo_service()
print("Готово. Откройте в браузере ссылку нужной роли (ссылка действует сутки):")
for u, r in (("teacher1", "преподаватель (дисциплины 0-2)"), ("teacher2", "другой преподаватель (3-4)"), ("methodist1", "методист: только списки"), ("admin1", "администратор: журнал")):
    role = {"teacher1": "teacher", "teacher2": "teacher", "methodist1": "methodist", "admin1": "admin"}[u]
    print(f"  {r}:\n    http://127.0.0.1:8000/login?token={make_token(u, role, svc.secret, ttl=86400)}")
print("Остановить сервер: Ctrl+C")
make_server("127.0.0.1", 8000, make_app(svc)).serve_forever()
