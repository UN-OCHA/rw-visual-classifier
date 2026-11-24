"""Generate interactive HTML reviews of test-set predictions."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from rw_visual_classifier.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    IMAGES_DIR,
    ONNX_PATH,
    ROOT,
    SPLIT_SEED,
)
from rw_visual_classifier.data import load_split_frames
from rw_visual_classifier.predict import load_session, predict_image

REVIEW_DIR = ARTIFACTS_DIR / "review"


def _row_payload(row, result: dict) -> dict:
    rel_path = str(row.rel_path)
    true_label = str(row.label)
    pred_label = str(result["label"])
    rel_from_html = Path("../..") / "data" / "images" / rel_path
    conf = float(result["probabilities"].get(pred_label, 0.0))
    return {
        "report_id": int(row.report_id),
        "true_label": true_label,
        "pred_label": pred_label,
        "pair": f"{true_label}→{pred_label}",
        "confidence": conf,
        "probabilities": result["probabilities"],
        "image": rel_from_html.as_posix(),
        "rw_url": f"https://reliefweb.int/node/{int(row.report_id)}",
    }


def collect_predictions(
    model_path: Path = ONNX_PATH,
    split: str = "test",
) -> list[dict]:
    frames = load_split_frames()
    if split not in frames or frames[split].empty:
        raise RuntimeError(f"No rows for split={split}")
    frame = frames[split]
    session = load_session(model_path)

    rows: list[dict] = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="predict"):
        image_path = IMAGES_DIR / str(row.rel_path)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        result = predict_image(session, image)
        rows.append(_row_payload(row, result))
    return rows


def collect_failures(predictions: list[dict]) -> list[dict]:
    failures = [p for p in predictions if p["true_label"] != p["pred_label"]]
    failures.sort(key=lambda x: (x["pair"], x["report_id"]))
    return failures


def sample_agreements(
    predictions: list[dict],
    *,
    sample_size: int = 150,
    seed: int = SPLIT_SEED,
) -> list[dict]:
    """Stratified sample of model==RW rows (~equal per class).

    Within each class, take roughly half from above-median confidence and half
    from below (when enough rows exist).
    """
    agreements = [p for p in predictions if p["true_label"] == p["pred_label"]]
    if not agreements:
        return []

    per_class = max(1, sample_size // len(CLASS_NAMES))
    rng = random.Random(seed)
    sampled: list[dict] = []

    for label in CLASS_NAMES:
        pool = [p for p in agreements if p["true_label"] == label]
        if not pool:
            continue
        target = min(per_class, len(pool))
        confs = sorted(p["confidence"] for p in pool)
        median = confs[len(confs) // 2]
        high = [p for p in pool if p["confidence"] >= median]
        low = [p for p in pool if p["confidence"] < median]
        take_high = min(len(high), (target + 1) // 2)
        take_low = min(len(low), target - take_high)
        # Top up from the other bucket if one side is short
        if take_high + take_low < target:
            if len(high) > take_high:
                take_high = min(len(high), target - take_low)
            elif len(low) > take_low:
                take_low = min(len(low), target - take_high)
        chosen = rng.sample(high, take_high) + rng.sample(low, take_low)
        if len(chosen) < target:
            chosen_ids = {p["report_id"] for p in chosen}
            leftover = [p for p in pool if p["report_id"] not in chosen_ids]
            need = target - len(chosen)
            if leftover:
                chosen.extend(rng.sample(leftover, min(need, len(leftover))))
        sampled.extend(chosen)

    # If some classes were short, top up randomly from remaining agreements
    if len(sampled) < sample_size:
        chosen_ids = {p["report_id"] for p in sampled}
        leftover = [p for p in agreements if p["report_id"] not in chosen_ids]
        need = sample_size - len(sampled)
        if leftover:
            sampled.extend(rng.sample(leftover, min(need, len(leftover))))

    sampled.sort(key=lambda x: (x["true_label"], -x["confidence"], x["report_id"]))
    return sampled[:sample_size]


def _html_page(
    items: list[dict],
    *,
    mode: str,
    title: str,
    subtitle: str,
    storage_key: str,
    export_name: str,
    item_noun: str,
) -> str:
    payload = json.dumps(items, indent=None)
    class_names_json = json.dumps(list(CLASS_NAMES))
    filter_label = "Filter by class" if mode == "agreements" else "Filter"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4f1ec;
      --ink: #1c1a17;
      --muted: #5c574f;
      --card: #fffdf9;
      --line: #d9d2c6;
      --map: #1f6f5b;
      --info: #9a4d1c;
      --other: #3d4f6f;
      --accent: #2b2a28;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 0%, #efe6d8 0%, transparent 45%),
        radial-gradient(circle at 90% 10%, #e4ebe7 0%, transparent 40%),
        var(--bg);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(8px);
      background: rgba(244, 241, 236, 0.92);
      border-bottom: 1px solid var(--line);
      padding: 0.9rem 1.25rem 1rem;
    }}
    h1 {{
      margin: 0 0 0.35rem;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
    }}
    .meta {{ color: var(--muted); font-size: 0.92rem; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1rem;
      align-items: center;
      margin-top: 0.75rem;
    }}
    select, button {{
      font: inherit;
      border: 1px solid var(--line);
      background: white;
      border-radius: 8px;
      padding: 0.4rem 0.7rem;
      cursor: pointer;
    }}
    button.primary {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .counts {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.75rem;
    }}
    .chip {{
      background: white;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.25rem 0.7rem;
      font-size: 0.85rem;
    }}
    .chip strong {{ font-variant-numeric: tabular-nums; }}
    main {{
      padding: 1rem 1.25rem 3rem;
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .card.hidden {{ display: none; }}
    .thumb {{
      background: #ece7df;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #f7f4ef;
    }}
    .body {{ padding: 0.85rem 0.9rem 1rem; display: grid; gap: 0.55rem; }}
    .pair {{
      font-size: 0.8rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .labels {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      font-size: 0.9rem;
    }}
    .tag {{
      border-radius: 6px;
      padding: 0.15rem 0.45rem;
      border: 1px solid var(--line);
      background: #f7f4ef;
    }}
    .tag.true-Map {{ color: var(--map); border-color: #b7d5cb; }}
    .tag.true-Infographic {{ color: var(--info); border-color: #e5c4a8; }}
    .tag.true-Other {{ color: var(--other); border-color: #b8c3d6; }}
    .probs {{
      font-size: 0.8rem;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .choices {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.35rem;
    }}
    .choices label {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.45rem 0.3rem;
      text-align: center;
      font-size: 0.85rem;
      cursor: pointer;
      background: white;
      user-select: none;
    }}
    .choices input {{ display: none; }}
    .choices label:has(input:checked) {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="meta">{subtitle}</div>
    <div class="controls">
      <label>{filter_label}
        <select id="pairFilter"><option value="all">All</option></select>
      </label>
      <button type="button" id="clearBtn">Clear my labels</button>
      <button type="button" class="primary" id="exportBtn">Export JSON</button>
      <span class="meta" id="progress"></span>
    </div>
    <div class="counts" id="counts"></div>
  </header>
  <main id="grid"></main>
  <script>
    const ITEMS = {payload};
    const CLASSES = {class_names_json};
    const MODE = {json.dumps(mode)};
    const ITEM_NOUN = {json.dumps(item_noun)};
    const STORAGE_KEY = {json.dumps(storage_key)};
    const EXPORT_NAME = {json.dumps(export_name)};

    const grid = document.getElementById("grid");
    const pairFilter = document.getElementById("pairFilter");
    const countsEl = document.getElementById("counts");
    const progressEl = document.getElementById("progress");

    function loadHuman() {{
      try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}"); }}
      catch {{ return {{}}; }}
    }}
    function saveHuman(map) {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    }}

    let human = loadHuman();

    const filterKeys = MODE === "agreements"
      ? [...CLASSES]
      : [...new Set(ITEMS.map(f => f.pair))].sort();
    for (const key of filterKeys) {{
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = key;
      pairFilter.appendChild(opt);
    }}

    function cardFilterKey(item) {{
      return MODE === "agreements" ? item.true_label : item.pair;
    }}

    function renderCounts() {{
      const labeled = Object.keys(human).filter(id =>
        ITEMS.some(i => String(i.report_id) === id)
      ).length;
      const byHuman = Object.fromEntries(CLASSES.map(c => [c, 0]));
      const agreeTrue = {{ yes: 0, no: 0 }};
      const agreePred = {{ yes: 0, no: 0 }};
      for (const f of ITEMS) {{
        const h = human[String(f.report_id)];
        if (!h) continue;
        byHuman[h] = (byHuman[h] || 0) + 1;
        if (h === f.true_label) agreeTrue.yes++; else agreeTrue.no++;
        if (h === f.pred_label) agreePred.yes++; else agreePred.no++;
      }}
      const silent = MODE === "agreements"
        ? `<span class="chip">Silent errors (you ≠ RW/model): <strong>${{agreeTrue.no}}</strong></span>`
        : `<span class="chip">Agree w/ model: <strong>${{agreePred.yes}}</strong> / disagree <strong>${{agreePred.no}}</strong></span>`;
      countsEl.innerHTML = `
        <span class="chip">${{ITEM_NOUN}}: <strong>${{ITEMS.length}}</strong></span>
        <span class="chip">Labeled: <strong>${{labeled}}</strong></span>
        ${{CLASSES.map(c => `<span class="chip">${{c}}: <strong>${{byHuman[c] || 0}}</strong></span>`).join("")}}
        <span class="chip">Agree w/ RW/model label: <strong>${{agreeTrue.yes}}</strong> / disagree <strong>${{agreeTrue.no}}</strong></span>
        ${{silent}}
      `;
      progressEl.textContent = `${{labeled}} / ${{ITEMS.length}} reviewed`;
    }}

    function renderCards() {{
      const filter = pairFilter.value;
      grid.innerHTML = "";
      for (const f of ITEMS) {{
        const card = document.createElement("article");
        card.className = "card";
        const key = cardFilterKey(f);
        card.dataset.pair = key;
        if (filter !== "all" && key !== filter) card.classList.add("hidden");
        const probs = CLASSES.map(c => `${{c[0]}}:${{(f.probabilities[c] * 100).toFixed(1)}}%`).join(" · ");
        const selected = human[String(f.report_id)] || "";
        const conf = ((f.confidence || 0) * 100).toFixed(1);
        card.innerHTML = `
          <div class="thumb"><img src="${{f.image}}" alt="report ${{f.report_id}}" loading="lazy" /></div>
          <div class="body">
            <div class="pair">${{MODE === "agreements" ? `Agree · ${{f.true_label}} · conf ${{conf}}%` : f.pair}}</div>
            <div class="labels">
              <span class="tag true-${{f.true_label}}">RW: ${{f.true_label}}</span>
              <span class="tag true-${{f.pred_label}}">Model: ${{f.pred_label}}</span>
              <a class="tag" href="${{f.rw_url}}" target="_blank" rel="noopener">#${{f.report_id}}</a>
            </div>
            <div class="probs">${{probs}}</div>
            <div class="choices" data-id="${{f.report_id}}">
              ${{CLASSES.map(c => `
                <label>
                  <input type="radio" name="h-${{f.report_id}}" value="${{c}}" ${{selected === c ? "checked" : ""}} />
                  ${{c}}
                </label>`).join("")}}
            </div>
          </div>
        `;
        grid.appendChild(card);
      }}
      renderCounts();
    }}

    grid.addEventListener("change", (e) => {{
      const input = e.target;
      if (!(input instanceof HTMLInputElement) || input.type !== "radio") return;
      const id = input.name.replace(/^h-/, "");
      human[id] = input.value;
      saveHuman(human);
      renderCounts();
    }});

    pairFilter.addEventListener("change", () => {{
      const filter = pairFilter.value;
      for (const card of grid.querySelectorAll(".card")) {{
        card.classList.toggle("hidden", filter !== "all" && card.dataset.pair !== filter);
      }}
    }});

    document.getElementById("clearBtn").addEventListener("click", () => {{
      if (!confirm("Clear all your labels for this review?")) return;
      human = {{}};
      saveHuman(human);
      renderCards();
    }});

    document.getElementById("exportBtn").addEventListener("click", () => {{
      const rows = ITEMS.map(f => ({{
        ...f,
        human_label: human[String(f.report_id)] || null,
      }}));
      const blob = new Blob([JSON.stringify(rows, null, 2)], {{ type: "application/json" }});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = EXPORT_NAME;
      a.click();
      URL.revokeObjectURL(a.href);
    }});

    renderCards();
  </script>
</body>
</html>
"""


def _write_review(
    items: list[dict],
    *,
    mode: str,
    html_name: str,
    json_name: str,
) -> tuple[Path, Path]:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REVIEW_DIR / html_name
    json_path = REVIEW_DIR / json_name
    if mode == "failures":
        html = _html_page(
            items,
            mode=mode,
            title="RW visual classifier — failure review",
            subtitle=(
                "Test-set mismatches only. Pick what <em>you</em> think each preview is. "
                "Counts update live and persist in this browser (localStorage)."
            ),
            storage_key="rwvc-failure-review-v1",
            export_name="failure-review.json",
            item_noun="Failures",
        )
    else:
        html = _html_page(
            items,
            mode=mode,
            title="RW visual classifier — agreement audit",
            subtitle=(
                "Random stratified sample where <strong>model = RW label</strong>. "
                "Mark what you think it is to estimate silent errors. "
                "Counts persist in localStorage (separate from failure review)."
            ),
            storage_key="rwvc-agreement-review-v1",
            export_name="agreement-review.json",
            item_noun="Agreements",
        )
    json_path.write_text(json.dumps(items, indent=2))
    html_path.write_text(html, encoding="utf-8")
    return html_path, json_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build an HTML page to review test-set predictions"
    )
    parser.add_argument(
        "--mode",
        choices=("failures", "agreements"),
        default="failures",
        help="failures = mismatches; agreements = stratified sample of matches",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=150,
        help="For --mode agreements: total sample size (default 150, ~50/class)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ONNX_PATH,
        help="ONNX model path",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Manifest split to scan (default: test)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="RNG seed for agreement sampling",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start a local HTTP server so images load reliably",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if not args.model.exists():
        raise SystemExit(f"ONNX model not found: {args.model}")
    if args.mode == "agreements" and args.sample <= 0:
        raise SystemExit("--sample must be > 0")

    predictions = collect_predictions(model_path=args.model, split=args.split)

    if args.mode == "failures":
        items = collect_failures(predictions)
        html_path, json_path = _write_review(
            items,
            mode="failures",
            html_name="index.html",
            json_name="failures.json",
        )
        print(f"Failures: {len(items)}")
        pairs = Counter(f["pair"] for f in items)
        for pair, count in sorted(pairs.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {pair}: {count}")
    else:
        items = sample_agreements(
            predictions, sample_size=args.sample, seed=args.seed
        )
        html_path, json_path = _write_review(
            items,
            mode="agreements",
            html_name="agreements.html",
            json_name="agreements.json",
        )
        print(f"Agreement sample: {len(items)} (requested {args.sample})")
        by_label = Counter(i["true_label"] for i in items)
        for label in CLASS_NAMES:
            print(f"  {label}: {by_label.get(label, 0)}")
        # Confidence mix
        by_bucket: dict[str, list[float]] = defaultdict(list)
        for item in items:
            by_bucket[item["true_label"]].append(item["confidence"])
        for label in CLASS_NAMES:
            confs = by_bucket.get(label) or []
            if confs:
                print(
                    f"  {label} conf: min={min(confs):.3f} "
                    f"median={sorted(confs)[len(confs)//2]:.3f} max={max(confs):.3f}"
                )

    print(f"Wrote {html_path}")
    print(f"Wrote {json_path}")

    page = html_path.relative_to(ROOT).as_posix()
    if args.serve:
        import http.server
        import os
        import webbrowser

        os.chdir(ROOT)
        url = f"http://127.0.0.1:{args.port}/{page}"
        print(f"Serving {ROOT} at http://127.0.0.1:{args.port}/")
        print(f"Open {url}")
        webbrowser.open(url)
        handler = http.server.SimpleHTTPRequestHandler
        with http.server.HTTPServer(("127.0.0.1", args.port), handler) as httpd:
            httpd.serve_forever()
    else:
        print(
            "Open with a local server from the repo root, e.g.:\n"
            f"  python -m http.server {args.port}\n"
            f"  then visit http://127.0.0.1:{args.port}/{page}\n"
            "Or re-run with --serve"
        )


if __name__ == "__main__":
    main()
