"""Минимизация и псевдонимизация данных, проверка k-анонимности, политика доступа по ролям.
Это технические меры; юридическую часть (основание обработки, срок хранения, согласия) нужно согласовать
с ответственным за персональные данные вуза."""
from __future__ import annotations

import hashlib
import hmac

import pandas as pd

ROLES = {
    "teacher": {"view_course_risk", "view_student_card", "mark_disagree"},
    "methodist": {"view_aggregates", "view_course_risk"},
    "admin": {"view_audit", "view_aggregates"},
}


def can(role: str, action: str) -> bool:
    """Разрешено ли действие роли. Неизвестная роль или действие — запрет."""
    return action in ROLES.get(role, set())


def pseudonym(student_id: int, secret: bytes, length: int = 12) -> str:
    """Стабильный псевдоним: HMAC-SHA256 от идентификатора с секретом хранилища (без секрета не восстановить)."""
    if not secret:
        raise ValueError("нужен непустой секрет")
    return hmac.new(secret, str(int(student_id)).encode(), hashlib.sha256).hexdigest()[:length]


def pseudonymize_frame(df: pd.DataFrame, secret: bytes, id_col: str = "student_id", drop=("gender", "origin")) -> pd.DataFrame:
    """Заменяет идентификатор псевдонимом и убирает чувствительные столбцы, не нужные для работы."""
    out = df.drop(columns=[c for c in drop if c in df.columns]).copy()
    out[id_col] = [pseudonym(i, secret) for i in out[id_col]]
    return out


def k_anonymity(df: pd.DataFrame, quasi: list) -> int:
    """Минимальный размер группы по квази-идентификаторам (чем больше, тем труднее переидентификация)."""
    if df.empty:
        return 0
    return int(df.groupby(quasi, dropna=False).size().min())
