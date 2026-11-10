"""Печатает DDL: python scripts/make_schema.py [sqlite|pg]"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from edurisk.store import schema_sql
print(schema_sql(sys.argv[1] if len(sys.argv) > 1 else "sqlite"), end="")
