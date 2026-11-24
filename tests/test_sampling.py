"""Unit tests for stratified catalog sampling (no network)."""

import pandas as pd

from rw_visual_classifier.dataset_build import (
    sample_from_catalog,
    strata_key_for_row,
    stratified_sample,
)


def _catalog_rows() -> pd.DataFrame:
    rows = []
    # Maps: 3 sources × 2 years × 4 docs
    rid = 1
    for source in ("10", "20", "30"):
        for year in ("2018", "2022"):
            for _ in range(4):
                rows.append(
                    {
                        "report_id": rid,
                        "label": "Map",
                        "format_ids": "12",
                        "format_names": "Map",
                        "source_id": source,
                        "source_name": f"Src{source}",
                        "country_id": "1",
                        "country_name": "Test",
                        "date_created": f"{year}-01-15T00:00:00+00:00",
                        "preview_url": f"https://example.com/{rid}.png",
                        "file_uuid": f"u{rid}",
                    }
                )
                rid += 1
    # Infographics: 2 sources × 1 year × 6
    for source in ("10", "40"):
        for _ in range(6):
            rows.append(
                {
                    "report_id": rid,
                    "label": "Infographic",
                    "format_ids": "12570",
                    "format_names": "Infographic",
                    "source_id": source,
                    "source_name": f"Src{source}",
                    "country_id": "1",
                    "country_name": "Test",
                    "date_created": "2020-06-01T00:00:00+00:00",
                    "preview_url": f"https://example.com/{rid}.png",
                    "file_uuid": f"u{rid}",
                }
            )
            rid += 1
    # Other: two formats × two sources × 5
    for fmt in ("Situation Report", "Appeal"):
        for source in ("10", "50"):
            for _ in range(5):
                rows.append(
                    {
                        "report_id": rid,
                        "label": "Other",
                        "format_ids": "10",
                        "format_names": fmt,
                        "source_id": source,
                        "source_name": f"Src{source}",
                        "country_id": "2",
                        "country_name": "Elsewhere",
                        "date_created": "2019-03-01T00:00:00+00:00",
                        "preview_url": f"https://example.com/{rid}.png",
                        "file_uuid": f"u{rid}",
                    }
                )
                rid += 1
    return pd.DataFrame(rows)


def test_strata_key_map_uses_source_and_year():
    row = pd.Series(
        {
            "label": "Map",
            "source_id": "99",
            "date_created": "2021-05-01T00:00:00+00:00",
            "format_names": "Map",
        }
    )
    assert strata_key_for_row(row) == "99|2021"


def test_strata_key_other_uses_format_and_source():
    row = pd.Series(
        {
            "label": "Other",
            "source_id": "7",
            "format_names": "Appeal",
            "date_created": "2021-05-01T00:00:00+00:00",
        }
    )
    assert strata_key_for_row(row) == "Appeal|7"


def test_stratified_sample_exact_count_and_seed_stability():
    catalog = _catalog_rows()
    maps = catalog[catalog["label"] == "Map"]
    a = stratified_sample(maps, 12, seed=42)
    b = stratified_sample(maps, 12, seed=42)
    c = stratified_sample(maps, 12, seed=99)
    assert len(a) == 12
    assert a["report_id"].tolist() == b["report_id"].tolist()
    assert a["report_id"].tolist() != c["report_id"].tolist()
    # Multiple sources represented
    assert a["source_id"].nunique() >= 2


def test_sample_from_catalog_limit_per_class():
    catalog = _catalog_rows()
    manifest = sample_from_catalog(catalog, limit_per_class=10, seed=42)
    counts = manifest["label"].value_counts().to_dict()
    assert counts["Map"] == 10
    assert counts["Infographic"] == 10
    assert counts["Other"] == 10
    assert set(manifest["split"]) <= {"train", "val", "test"}
    assert "rel_path" in manifest.columns
