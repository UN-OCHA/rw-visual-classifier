"""Build catalog → stratified sample → download preview images."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
from tqdm import tqdm

from rw_visual_classifier.config import (
    CATALOG_PATH,
    DATA_DIR,
    DEFAULT_LIMIT_PER_CLASS,
    DOWNLOAD_CONCURRENCY,
    DOWNLOAD_RETRIES,
    IMAGES_DIR,
    MANIFEST_PATH,
    OTHER_SAMPLE_SIZE,
    SPLIT_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from rw_visual_classifier.rw_client import (
    ReliefWebClient,
    map_infographic_filter,
    other_filter,
    parse_report_item,
)

CATALOG_COLUMNS = [
    "report_id",
    "label",
    "format_ids",
    "format_names",
    "source_id",
    "source_name",
    "country_id",
    "country_name",
    "date_created",
    "preview_url",
    "file_uuid",
]


def _image_path(label: str, report_id: int) -> Path:
    return IMAGES_DIR / label / f"{report_id}.jpg"


def _assign_splits(report_ids: list[int], seed: int = SPLIT_SEED) -> dict[int, str]:
    ids = sorted(report_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    splits: dict[int, str] = {}
    for i, rid in enumerate(ids):
        if i < n_train:
            splits[rid] = "train"
        elif i < n_train + n_val:
            splits[rid] = "val"
        else:
            splits[rid] = "test"
    if n > 0 and not any(s == "test" for s in splits.values()) and TEST_RATIO > 0:
        splits[ids[-1]] = "test"
    return splits


def _year_from_date(value: object) -> str:
    text = str(value or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return "unknown"


def strata_key_for_row(row: pd.Series) -> str:
    label = str(row["label"])
    source_id = str(row.get("source_id") or "unknown")
    if label in ("Map", "Infographic"):
        year = _year_from_date(row.get("date_created"))
        return f"{source_id}|{year}"
    format_names = str(row.get("format_names") or "unknown")
    return f"{format_names}|{source_id}"


def stratified_sample(
    frame: pd.DataFrame,
    n: int,
    *,
    seed: int = SPLIT_SEED,
    strata_col: str = "_strata",
) -> pd.DataFrame:
    """Sample ``n`` rows with roughly even quota across strata.

    Allocates ceil(n / n_strata) per stratum (capped by group size), then
    randomly trims or tops up to exactly ``n``.
    """
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= n:
        return frame.copy()

    work = frame.copy()
    if strata_col not in work.columns:
        work[strata_col] = work.apply(strata_key_for_row, axis=1)

    rng = random.Random(seed)
    groups = {
        key: group.index.tolist()
        for key, group in work.groupby(strata_col, sort=False)
    }
    strata_keys = sorted(groups.keys())
    n_strata = len(strata_keys)
    base_quota = int(math.ceil(n / n_strata)) if n_strata else n

    chosen: list[int] = []
    leftovers: list[int] = []
    for key in strata_keys:
        idxs = list(groups[key])
        rng.shuffle(idxs)
        take = min(base_quota, len(idxs))
        chosen.extend(idxs[:take])
        leftovers.extend(idxs[take:])

    if len(chosen) > n:
        rng.shuffle(chosen)
        chosen = chosen[:n]
    elif len(chosen) < n:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: n - len(chosen)])

    sampled = work.loc[chosen].drop(columns=[strata_col], errors="ignore")
    return sampled.reset_index(drop=True)


def build_catalog(*, refresh: bool = False, path: Path = CATALOG_PATH) -> pd.DataFrame:
    """Fetch full preview metadata for Map/Infographic/Other into catalog.csv."""
    if path.exists() and not refresh:
        print(f"Using existing catalog: {path}")
        return load_catalog(path)

    rows: list[dict] = []
    with ReliefWebClient() as client:
        for item in tqdm(
            client.iter_reports(map_infographic_filter()),
            desc="Catalog Map/Infographic",
            unit="rep",
        ):
            row = parse_report_item(item)
            if row is not None:
                rows.append(row)

        for item in tqdm(
            client.iter_reports(other_filter()),
            desc="Catalog Other",
            unit="rep",
        ):
            row = parse_report_item(item)
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError("No reports collected from ReliefWeb")

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["report_id"], keep="first")
        .sort_values("report_id")
        .reset_index(drop=True)
    )
    # Ensure stable column order
    for col in CATALOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[CATALOG_COLUMNS]
    save_catalog(df, path)
    print(f"Wrote catalog: {path} ({len(df)} rows)")
    print(df["label"].value_counts().to_string())
    return df


def save_catalog(df: pd.DataFrame, path: Path = CATALOG_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    return pd.read_csv(path)


def sample_from_catalog(
    catalog: pd.DataFrame,
    *,
    limit_per_class: int | None = DEFAULT_LIMIT_PER_CLASS,
    other_sample_size: int = OTHER_SAMPLE_SIZE,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Stratified sample from catalog into a train/val/test manifest frame."""
    if catalog.empty:
        raise RuntimeError("Catalog is empty")

    label_seed_offset = {"Map": 1, "Infographic": 2, "Other": 3}
    parts: list[pd.DataFrame] = []
    for label in ("Map", "Infographic", "Other"):
        subset = catalog[catalog["label"] == label]
        if subset.empty:
            continue
        label_seed = seed + label_seed_offset[label]
        if limit_per_class and limit_per_class > 0:
            target = min(limit_per_class, len(subset))
            parts.append(stratified_sample(subset, target, seed=label_seed))
        elif label == "Other":
            target = min(other_sample_size, len(subset))
            parts.append(stratified_sample(subset, target, seed=label_seed))
        else:
            # Unlimited Map / Infographic: keep all
            parts.append(subset.copy())

    if not parts:
        raise RuntimeError("No rows selected from catalog")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["report_id"], keep="first")
    splits = _assign_splits(df["report_id"].tolist(), seed=seed)
    df["split"] = df["report_id"].map(splits)
    df["rel_path"] = df.apply(
        lambda r: str(Path(r["label"]) / f"{r['report_id']}.jpg"),
        axis=1,
    )
    return df.sort_values("report_id").reset_index(drop=True)


def _download_one(url: str, dest: Path) -> tuple[bool, str]:
    """Download one image. Uses a per-call client (thread-safe)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True, "cached"
    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err = ""
    for attempt in range(DOWNLOAD_RETRIES):
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(2**attempt)
                        last_err = f"HTTP {response.status_code}"
                        continue
                    response.raise_for_status()
                    with open(tmp, "wb") as handle:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            handle.write(chunk)
            tmp.replace(dest)
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 — surface as per-file status
            last_err = str(exc)
            time.sleep(2**attempt)
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False, last_err


def download_images(
    df: pd.DataFrame,
    *,
    concurrency: int = DOWNLOAD_CONCURRENCY,
    force: bool = False,
) -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    work: list[tuple[int, str, Path]] = []
    for row in df.itertuples(index=False):
        dest = _image_path(row.label, int(row.report_id))
        if force and dest.exists():
            dest.unlink()
        work.append((int(row.report_id), row.preview_url, dest))

    status: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_download_one, url, dest): rid
            for rid, url, dest in work
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading images",
            unit="img",
        ):
            rid = futures[future]
            ok, msg = future.result()
            status[rid] = "ok" if ok else f"fail:{msg}"

    df = df.copy()
    df["download_status"] = df["report_id"].map(status)
    return df


def save_manifest(df: pd.DataFrame, path: Path = MANIFEST_PATH) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)


def filter_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose image file exists on disk."""
    ready_mask = []
    for row in df.itertuples(index=False):
        path = IMAGES_DIR / row.rel_path
        ready_mask.append(path.exists() and path.stat().st_size > 0)
    return df.loc[ready_mask].reset_index(drop=True)


def manifest_fingerprint(df: pd.DataFrame) -> str:
    payload = df[["report_id", "label", "split"]].to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build ReliefWeb visual classifier dataset "
        "(catalog → stratified sample → download)"
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=DEFAULT_LIMIT_PER_CLASS,
        help=(
            "Stratified cap per class (default 5000). "
            "Use 0 for all Map/Infographic + Other up to --other-sample-size"
        ),
    )
    parser.add_argument(
        "--other-sample-size",
        type=int,
        default=OTHER_SAMPLE_SIZE,
        help="Other cap when --limit-per-class 0 (default 75000)",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Refetch catalog from the ReliefWeb API",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Build/refresh catalog and stop",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Resample manifest from existing catalog (no API, no download)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download images for an existing manifest",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DOWNLOAD_CONCURRENCY,
        help="Parallel image downloads",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download images even if cached",
    )
    args = parser.parse_args(argv)

    limit = args.limit_per_class
    if limit is not None and limit < 0:
        raise SystemExit("--limit-per-class must be >= 0")

    if args.download_only:
        df = load_manifest()
    elif args.sample_only:
        catalog = load_catalog()
        print(f"Loaded catalog: {CATALOG_PATH} ({len(catalog)} rows)")
        df = sample_from_catalog(
            catalog,
            limit_per_class=limit,
            other_sample_size=args.other_sample_size,
        )
        save_manifest(df)
        print(f"Wrote manifest: {MANIFEST_PATH} ({len(df)} rows)")
        print(df["label"].value_counts().to_string())
        print(df["split"].value_counts().to_string())
        return
    else:
        catalog = build_catalog(refresh=args.refresh_catalog)
        if args.catalog_only:
            return
        df = sample_from_catalog(
            catalog,
            limit_per_class=limit,
            other_sample_size=args.other_sample_size,
        )
        save_manifest(df)
        print(f"Wrote manifest: {MANIFEST_PATH} ({len(df)} rows)")
        print(df["label"].value_counts().to_string())
        print(df["split"].value_counts().to_string())

    print("Downloading preview images…")
    df = download_images(
        df,
        concurrency=args.concurrency,
        force=args.force_download,
    )
    save_manifest(df)
    ok = (df["download_status"] == "ok").sum()
    print(f"Downloads ok: {ok}/{len(df)}")
    print(f"Manifest fingerprint: {manifest_fingerprint(df)}")


if __name__ == "__main__":
    main()
