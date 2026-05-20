from __future__ import annotations

from .defense import use_shield
from .weapons import use_sword


EQUIPMENT_HANDLERS = {
    "shield": use_shield,
    "sword": use_sword,
}
