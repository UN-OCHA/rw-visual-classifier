"""ReliefWeb API client for report metadata pagination."""

from __future__ import annotations

import time
from typing import Any, Iterator

import httpx

from rw_visual_classifier.config import (
    API_PAGE_LIMIT,
    FORMAT_ID_INFOGRAPHIC,
    FORMAT_ID_MAP,
    RELIEFWEB_API_APPNAME,
    RELIEFWEB_API_URL,
)

REPORT_FIELDS = [
    "id",
    "format.id",
    "format.name",
    "source.id",
    "source.name",
    "primary_country.id",
    "primary_country.name",
    "date.created",
    "file.preview.url-large",
    "file.uuid",
]


class ReliefWebError(RuntimeError):
    """Raised when the ReliefWeb API returns an error payload."""


class ReliefWebClient:
    """Thin POST client for /v2/reports with pagination."""

    def __init__(
        self,
        base_url: str | None = None,
        appname: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or RELIEFWEB_API_URL).rstrip("/")
        self.appname = appname or RELIEFWEB_API_APPNAME
        if not self.appname:
            raise ValueError(
                "RELIEFWEB_API_APPNAME is required (set it in .env)"
            )
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ReliefWebClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _reports_url(self) -> str:
        return f"{self.base_url}/reports?appname={self.appname}"

    def search_reports(
        self,
        *,
        filter_body: dict[str, Any],
        offset: int = 0,
        limit: int = API_PAGE_LIMIT,
        fields: list[str] | None = None,
        sort: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "filter": filter_body,
            "fields": {"include": fields or REPORT_FIELDS},
            "sort": sort or ["id:asc"],
            "limit": limit,
            "offset": offset,
        }
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self._client.post(self._reports_url(), json=payload)
                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                if "error" in data:
                    raise ReliefWebError(str(data["error"]))
                return data
            except (httpx.HTTPError, ReliefWebError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise ReliefWebError(f"ReliefWeb request failed: {last_error}")

    def iter_reports(
        self,
        filter_body: dict[str, Any],
        *,
        page_limit: int = API_PAGE_LIMIT,
        max_items: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        offset = 0
        fetched = 0
        while True:
            page = self.search_reports(
                filter_body=filter_body,
                offset=offset,
                limit=page_limit,
            )
            items = page.get("data") or []
            if not items:
                break
            for item in items:
                yield item
                fetched += 1
                if max_items is not None and fetched >= max_items:
                    return
            offset += len(items)
            total = int(page.get("totalCount") or 0)
            if offset >= total:
                break


def map_infographic_filter() -> dict[str, Any]:
    return {
        "operator": "AND",
        "conditions": [
            {
                "field": "format.id",
                "value": [FORMAT_ID_MAP, FORMAT_ID_INFOGRAPHIC],
            },
            {"field": "file.preview.url"},
        ],
    }


def format_with_preview_filter(format_id: int) -> dict[str, Any]:
    return {
        "operator": "AND",
        "conditions": [
            {"field": "format.id", "value": format_id},
            {"field": "file.preview.url"},
        ],
    }


def other_filter() -> dict[str, Any]:
    return {
        "operator": "AND",
        "conditions": [
            {
                "field": "format.id",
                "value": [FORMAT_ID_MAP, FORMAT_ID_INFOGRAPHIC],
                "negate": True,
            },
            {"field": "file.preview.url"},
        ],
    }


def label_from_format_ids(format_ids: list[int]) -> str:
    ids = set(format_ids)
    if FORMAT_ID_MAP in ids:
        return "Map"
    if FORMAT_ID_INFOGRAPHIC in ids:
        return "Infographic"
    return "Other"


def extract_preview(fields: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (url_large, file_uuid) for the first file with a large preview."""
    files = fields.get("file") or []
    if isinstance(files, dict):
        files = [files]
    for file_obj in files:
        preview = (file_obj or {}).get("preview") or {}
        url = preview.get("url-large") or preview.get("url")
        if url:
            return url, file_obj.get("uuid")
    return None, None


def _first_ref(
    value: Any,
    *,
    id_key: str = "id",
    name_key: str = "name",
) -> tuple[str, str]:
    """Take the first taxonomy entry from a list/dict field."""
    if value is None:
        return "", ""
    entries = value
    if isinstance(value, dict):
        entries = [value]
    if not entries:
        return "", ""
    first = entries[0] or {}
    ref_id = first.get(id_key, "")
    ref_name = first.get(name_key, "")
    return ("" if ref_id is None else str(ref_id), str(ref_name or ""))


def _date_created(fields: dict[str, Any]) -> str:
    date_obj = fields.get("date")
    if isinstance(date_obj, dict):
        return str(date_obj.get("created") or "")
    return ""


def parse_report_item(item: dict[str, Any]) -> dict[str, Any] | None:
    fields = item.get("fields") or {}
    report_id = fields.get("id") or item.get("id")
    if report_id is None:
        return None
    formats = fields.get("format") or []
    if isinstance(formats, dict):
        formats = [formats]
    format_ids = [int(f["id"]) for f in formats if "id" in f]
    format_names = [str(f.get("name", "")) for f in formats]
    url, file_uuid = extract_preview(fields)
    if not url:
        return None
    label = label_from_format_ids(format_ids)
    source_id, source_name = _first_ref(fields.get("source"))
    # primary_country may appear as object or via country with primary flag
    country_id, country_name = _first_ref(fields.get("primary_country"))
    if not country_id:
        countries = fields.get("country") or []
        if isinstance(countries, dict):
            countries = [countries]
        primary = next(
            (c for c in countries if (c or {}).get("primary")),
            countries[0] if countries else None,
        )
        if primary:
            country_id = str(primary.get("id") or "")
            country_name = str(primary.get("name") or "")
    return {
        "report_id": int(report_id),
        "label": label,
        "format_ids": ",".join(str(i) for i in format_ids),
        "format_names": "|".join(format_names),
        "source_id": source_id,
        "source_name": source_name,
        "country_id": country_id,
        "country_name": country_name,
        "date_created": _date_created(fields),
        "preview_url": url,
        "file_uuid": file_uuid or "",
    }
