from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


EVENT_DETAIL_ALIASES: dict[str, tuple[str, ...]] = {}


def normalize_event_records(
    events: list[str],
    event_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    details_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in event_details:
        detail_type = str(detail.get("type", "detail"))
        details_by_type[detail_type].append(detail)

    records: list[dict[str, Any]] = []
    for event_name in events:
        detail = _pop_matching_detail(event_name, details_by_type)
        if detail is None:
            records.append({"name": event_name})
            continue
        payload = {key: value for key, value in detail.items() if key != "type"}
        records.append({"name": event_name, **payload})

    for detail_type, leftovers in details_by_type.items():
        for detail in leftovers:
            payload = {key: value for key, value in detail.items() if key != "type"}
            records.append({"name": detail_type, **payload})
    return records


def event_records_to_counts(event_records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(record.get("name", "detail")) for record in event_records))


def event_counts_to_flags(event_counts: dict[str, int]) -> dict[str, bool]:
    return {event_name: count > 0 for event_name, count in event_counts.items()}


def merge_event_counts(*counts_sets: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for counts in counts_sets:
        for key, value in counts.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def merge_event_records(*record_sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for records in record_sets:
        merged.extend(records)
    return merged


def build_event_counts(events: list[str]) -> dict[str, int]:
    return event_records_to_counts([{"name": event_name} for event_name in events])


def build_event_flags(events: list[str]) -> dict[str, bool]:
    return event_counts_to_flags(build_event_counts(events))


def build_event_records(
    events: list[str],
    event_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return normalize_event_records(events, event_details)


def _pop_matching_detail(
    event_name: str,
    details_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    names = (event_name, *EVENT_DETAIL_ALIASES.get(event_name, ()))
    for name in names:
        details = details_by_type.get(name)
        if details:
            detail = details.pop(0)
            if not details:
                details_by_type.pop(name, None)
            return detail
    return None
