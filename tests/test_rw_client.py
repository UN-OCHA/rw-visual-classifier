"""Unit tests for metadata parsing (no network)."""

import pickle

from PIL import Image

from rw_visual_classifier.rw_client import label_from_format_ids, parse_report_item
from rw_visual_classifier.transforms import (
    Letterbox,
    build_train_transform,
    letterbox,
)


def test_label_from_format_ids():
    assert label_from_format_ids([12]) == "Map"
    assert label_from_format_ids([12570]) == "Infographic"
    assert label_from_format_ids([10]) == "Other"
    assert label_from_format_ids([12, 10]) == "Map"


def test_parse_report_item_extracts_preview():
    item = {
        "id": "123",
        "fields": {
            "id": 123,
            "format": [{"id": 12, "name": "Map"}],
            "source": [{"id": 55, "name": "OCHA"}],
            "primary_country": {"id": 114, "name": "Syria"},
            "date": {"created": "2022-01-02T03:04:05+00:00"},
            "file": [
                {
                    "uuid": "abc",
                    "preview": {
                        "url-large": "https://example.com/large.png",
                        "url": "https://example.com/preview.png",
                    },
                }
            ],
        },
    }
    row = parse_report_item(item)
    assert row is not None
    assert row["report_id"] == 123
    assert row["label"] == "Map"
    assert row["preview_url"] == "https://example.com/large.png"
    assert row["file_uuid"] == "abc"
    assert row["source_id"] == "55"
    assert row["source_name"] == "OCHA"
    assert row["country_id"] == "114"
    assert row["country_name"] == "Syria"
    assert row["date_created"] == "2022-01-02T03:04:05+00:00"


def test_parse_report_item_skips_missing_preview():
    item = {
        "fields": {
            "id": 1,
            "format": [{"id": 10, "name": "Situation Report"}],
            "file": [{"uuid": "x"}],
        }
    }
    assert parse_report_item(item) is None


def test_letterbox_square():
    img = Image.new("RGB", (400, 200), color=(255, 0, 0))
    out = letterbox(img, size=224)
    assert out.size == (224, 224)


def test_transforms_are_picklable():
    """DataLoader workers on macOS use spawn and must pickle transforms."""
    transform = build_train_transform()
    restored = pickle.loads(pickle.dumps(transform))
    img = Image.new("RGB", (100, 50), color=(0, 128, 255))
    tensor = restored(img)
    assert tuple(tensor.shape) == (3, 224, 224)
    assert pickle.loads(pickle.dumps(Letterbox(size=128)))(img).size == (128, 128)
