"""python scripts/load_oulad.py <папка с CSV OULAD> [БД.sqlite]  — загрузка, сводка, запись в БД."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from edurisk.oulad import load_oulad, summary
from edurisk.store import connect, init_db, save_world

w = load_oulad(sys.argv[1])
print(summary(w))
if len(sys.argv) > 2:
    conn = connect(sys.argv[2])
    init_db(conn)
    save_world(conn, w)
    print("записано в", sys.argv[2])
