from __future__ import annotations

from pathlib import Path

import argparse
import html
import json

import numpy as np
import pandas as pd


PLOTLY_CDN_DEFAULT = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive raw-data dashboard for a single case directory."
    )
    parser.add_argument("--case-dir", type=str, required=True, help="Case directory containing nodes.csv and edges.csv.")
    parser.add_argument("--output", type=str, required=True, help="Output HTML path.")
    parser.add_argument(
        "--display-sample-size",
        type=int,
        default=25000,
        help="Maximum number of nodes rendered in the 3D plot at once.",
    )
    parser.add_argument(
        "--distance-scatter-sample-size",
        type=int,
        default=5000,
        help="Maximum number of points shown in the distance-response scatter plot.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for display sampling.")
    parser.add_argument("--plotly-cdn", type=str, default=PLOTLY_CDN_DEFAULT, help="Plotly CDN URL.")
    parser.add_argument("--title", type=str, default=None, help="Optional dashboard title.")
    return parser.parse_args()


def _json_ready_array(values: np.ndarray, decimals: int = 6) -> list[float] | list[int]:
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.int64).tolist()
    rounded = np.round(values.astype(np.float64), decimals=decimals)
    return rounded.tolist()


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    array = values.astype(np.float64)
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def _coerce_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    if numeric.notna().all():
        return numeric

    cleaned = (
        text.str.replace(",", "", regex=False)
        .str.replace("i", "1", regex=False)
        .str.replace("I", "1", regex=False)
        .str.replace("l", "1", regex=False)
        .str.replace("O", "0", regex=False)
        .str.replace("o", "0", regex=False)
    )
    repaired = pd.to_numeric(cleaned, errors="coerce")
    return repaired


def _load_case(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = case_dir / "nodes.csv"
    edges_path = case_dir / "edges.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes.csv under {case_dir}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges.csv under {case_dir}")

    nodes_df = pd.read_csv(nodes_path, low_memory=False)
    edges_df = pd.read_csv(edges_path, usecols=["src", "dst", "dist"])
    return nodes_df, edges_df


def _prepare_metrics(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, float]]]:
    required_columns = ["x", "y", "z", "bc_mask", "RTA", "RMises"]
    missing = [column for column in required_columns if column not in nodes_df.columns]
    if missing:
        raise KeyError(f"Missing required node columns: {missing}")

    for column in nodes_df.columns:
        converted = _coerce_numeric_series(nodes_df[column])
        if converted.notna().any():
            nodes_df[column] = converted

    for column in required_columns:
        converted = _coerce_numeric_series(nodes_df[column])
        missing_count = int(converted.isna().sum())
        if missing_count:
            raise ValueError(f"Column {column} still has {missing_count} non-numeric rows after cleanup.")
        nodes_df[column] = converted

    if "node_id" in nodes_df.columns:
        node_ids = pd.to_numeric(nodes_df["node_id"], errors="raise").astype(np.int64).to_numpy()
    else:
        node_ids = np.arange(len(nodes_df), dtype=np.int64)
        nodes_df["node_id"] = node_ids

    id_to_index = {int(node_id): index for index, node_id in enumerate(node_ids)}
    src_ids = pd.to_numeric(edges_df["src"], errors="raise").astype(np.int64).to_numpy()
    dst_ids = pd.to_numeric(edges_df["dst"], errors="raise").astype(np.int64).to_numpy()
    try:
        src_index = np.array([id_to_index[int(value)] for value in src_ids], dtype=np.int64)
        dst_index = np.array([id_to_index[int(value)] for value in dst_ids], dtype=np.int64)
    except KeyError as exc:
        raise KeyError(f"Edge endpoint node id not found in nodes.csv: {exc}") from exc

    distances = pd.to_numeric(edges_df["dist"], errors="raise").astype(np.float64).to_numpy()
    distances = np.maximum(distances, 1e-12)

    degree = np.bincount(np.concatenate([src_index, dst_index]), minlength=len(nodes_df)).astype(np.int64)
    nodes_df["graph_degree"] = degree

    for metric_name, prefix in (("RMises", "rmises"), ("RTA", "rta")):
        metric_values = pd.to_numeric(nodes_df[metric_name], errors="raise").astype(np.float64).to_numpy()
        edge_gradient = np.abs(metric_values[src_index] - metric_values[dst_index]) / distances

        gradient_sum = np.zeros(len(nodes_df), dtype=np.float64)
        gradient_max = np.zeros(len(nodes_df), dtype=np.float64)
        np.add.at(gradient_sum, src_index, edge_gradient)
        np.add.at(gradient_sum, dst_index, edge_gradient)
        np.maximum.at(gradient_max, src_index, edge_gradient)
        np.maximum.at(gradient_max, dst_index, edge_gradient)

        gradient_mean = np.divide(
            gradient_sum,
            np.maximum(degree.astype(np.float64), 1.0),
            out=np.zeros_like(gradient_sum),
            where=degree > 0,
        )
        nodes_df[f"{prefix}_grad_mean"] = gradient_mean
        nodes_df[f"{prefix}_grad_max"] = gradient_max

    numeric_columns = [
        column
        for column in nodes_df.columns
        if column not in {"node_id", "x", "y", "z"}
        and pd.api.types.is_numeric_dtype(nodes_df[column])
    ]
    preferred_order = [
        "RMises",
        "RTA",
        "bc_mask",
        "rmises_grad_max",
        "rmises_grad_mean",
        "rta_grad_max",
        "rta_grad_mean",
        "graph_degree",
    ]
    remaining = [column for column in numeric_columns if column not in preferred_order]
    ordered_metrics = [column for column in preferred_order if column in numeric_columns] + remaining

    summaries = {metric: _metric_summary(pd.to_numeric(nodes_df[metric], errors="raise").to_numpy()) for metric in ordered_metrics}
    return nodes_df, ordered_metrics, summaries


def _build_payload(
    case_dir: Path,
    nodes_df: pd.DataFrame,
    metrics: list[str],
    metric_summaries: dict[str, dict[str, float]],
    display_sample_size: int,
    distance_scatter_sample_size: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    node_count = len(nodes_df)
    node_ids = pd.to_numeric(nodes_df["node_id"], errors="raise").astype(np.int64).to_numpy()
    bc_mask = pd.to_numeric(nodes_df["bc_mask"], errors="raise").astype(np.float64).to_numpy()
    constrained_indices = np.flatnonzero(bc_mask != 0.0).astype(np.int64)

    if display_sample_size <= 0:
        raise ValueError("--display-sample-size must be positive")
    if distance_scatter_sample_size <= 0:
        raise ValueError("--distance-scatter-sample-size must be positive")

    display_size = min(display_sample_size, node_count)
    distance_size = min(distance_scatter_sample_size, node_count)
    display_indices = np.sort(rng.choice(node_count, size=display_size, replace=False)).astype(np.int64)
    distance_sample_indices = np.sort(rng.choice(node_count, size=distance_size, replace=False)).astype(np.int64)

    rmises = pd.to_numeric(nodes_df["RMises"], errors="raise").astype(np.float64).to_numpy()
    graph_degree = pd.to_numeric(nodes_df["graph_degree"], errors="raise").astype(np.float64).to_numpy()
    rmises_grad_max = pd.to_numeric(nodes_df["rmises_grad_max"], errors="raise").astype(np.float64).to_numpy()
    graph_degree_p90 = float(np.percentile(graph_degree, 90))
    rmises_grad_max_p90 = float(np.percentile(rmises_grad_max, 90))

    payload = {
        "case_name": case_dir.name,
        "title": case_dir.name,
        "node_count": int(node_count),
        "constrained_count": int(np.count_nonzero(bc_mask != 0.0)),
        "display_indices": _json_ready_array(display_indices),
        "distance_sample_indices": _json_ready_array(distance_sample_indices),
        "constrained_indices": _json_ready_array(constrained_indices),
        "node_ids": _json_ready_array(node_ids),
        "coords": {
            "x": _json_ready_array(pd.to_numeric(nodes_df["x"], errors="raise").to_numpy()),
            "y": _json_ready_array(pd.to_numeric(nodes_df["y"], errors="raise").to_numpy()),
            "z": _json_ready_array(pd.to_numeric(nodes_df["z"], errors="raise").to_numpy()),
        },
        "metrics": {
            metric: _json_ready_array(pd.to_numeric(nodes_df[metric], errors="raise").to_numpy())
            for metric in metrics
        },
        "metric_order": metrics,
        "metric_summaries": metric_summaries,
        "default_metric": "RMises" if "RMises" in metrics else metrics[0],
        "distance_metric_default": "RMises" if "RMises" in metrics else metrics[0],
        "default_node_id": int(node_ids[int(np.argmax(rmises))]),
        "heuristic_thresholds": {
            "graph_degree_p90": graph_degree_p90,
            "rmises_grad_max_p90": rmises_grad_max_p90,
        },
        "case_meta": {
            "path": str(case_dir.resolve()),
        },
    }
    return payload


def _build_html_document(title: str, payload: dict[str, object], plotly_cdn: str) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="{html.escape(plotly_cdn, quote=True)}"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f1eb;
      --bg-soft: #faf8f4;
      --panel: rgba(255, 253, 248, 0.92);
      --panel-strong: rgba(255, 250, 242, 0.98);
      --border: rgba(109, 80, 45, 0.16);
      --text: #2b2319;
      --muted: #766753;
      --primary: #b85a38;
      --secondary: #246d73;
      --accent: #3a8f5a;
      --warning: #a65a24;
      --danger: #8f2e2e;
      --shadow: 0 24px 56px rgba(56, 38, 18, 0.1);
      --radius: 24px;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(247, 197, 140, 0.28), transparent 26%),
        radial-gradient(circle at top right, rgba(118, 182, 187, 0.18), transparent 24%),
        linear-gradient(180deg, #f9f6ef 0%, #efe7da 100%);
    }}
    .shell {{
      width: min(1560px, calc(100vw - 28px));
      margin: 24px auto 40px;
    }}
    .hero {{
      padding: 28px 30px 34px;
      border: 1px solid var(--border);
      border-radius: calc(var(--radius) + 6px);
      background:
        linear-gradient(135deg, rgba(255, 246, 231, 0.95), rgba(255, 251, 246, 0.88)),
        linear-gradient(90deg, rgba(184, 90, 56, 0.08), rgba(36, 109, 115, 0.08));
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      width: 320px;
      height: 320px;
      right: -80px;
      top: -120px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(184, 90, 56, 0.18) 0%, rgba(184, 90, 56, 0) 72%);
      pointer-events: none;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 248, 238, 0.92);
      color: var(--primary);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 18px 0 10px;
      font-size: clamp(30px, 4vw, 52px);
      line-height: 1.02;
      letter-spacing: -0.04em;
      max-width: 900px;
    }}
    .hero p {{
      margin: 0;
      max-width: 900px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 16px;
    }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 22px;
      position: relative;
      z-index: 1;
    }}
    .hero-metric {{
      padding: 16px 18px;
      border-radius: 18px;
      background: rgba(255, 252, 246, 0.92);
      border: 1px solid rgba(116, 86, 52, 0.12);
    }}
    .hero-metric strong {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 600;
    }}
    .hero-metric span {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .grid {{
      display: grid;
      gap: 20px;
      margin-top: 20px;
    }}
    .controls-grid {{
      grid-template-columns: 1.5fr 1fr;
      align-items: start;
    }}
    .main-grid {{
      grid-template-columns: 1.55fr 0.85fr;
      align-items: start;
    }}
    .bottom-grid {{
      grid-template-columns: 1fr 1fr;
      align-items: start;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
      overflow: hidden;
    }}
    .card-header {{
      padding: 18px 20px 0;
    }}
    .card h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .card p.helper {{
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    .card-body {{
      padding: 18px 20px 20px;
    }}
    .control-columns {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    .control-group {{
      display: grid;
      gap: 14px;
    }}
    label {{
      display: grid;
      gap: 7px;
      font-size: 14px;
      color: var(--muted);
      font-weight: 600;
    }}
    select,
    input[type="number"],
    input[type="text"] {{
      width: 100%;
      border: 1px solid rgba(106, 86, 62, 0.16);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.78);
      padding: 10px 12px;
      font: inherit;
      color: var(--text);
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--primary);
    }}
    .inline-value {{
      font-size: 13px;
      color: var(--muted);
      text-align: right;
      margin-top: -2px;
    }}
    .button-row {{
      display: flex;
      gap: 12px;
      align-items: end;
    }}
    button {{
      border: none;
      border-radius: 12px;
      padding: 11px 14px;
      background: linear-gradient(180deg, #c56740 0%, #b24e2c 100%);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(184, 90, 56, 0.26);
    }}
    button.secondary {{
      background: linear-gradient(180deg, #2b7c84 0%, #1f646b 100%);
      box-shadow: 0 10px 24px rgba(36, 109, 115, 0.22);
    }}
    .plot {{
      width: 100%;
      min-height: 520px;
    }}
    #distancePlot,
    #histogramPlot {{
      min-height: 420px;
    }}
    #slicePlot {{
      min-height: 560px;
    }}
    .stat-list {{
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .stat-list li {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 253, 248, 0.78);
      border: 1px solid rgba(116, 86, 52, 0.08);
    }}
    .stat-list li strong {{
      color: var(--muted);
      font-size: 13px;
    }}
    .stat-list li span {{
      font-weight: 700;
      text-align: right;
    }}
    .hint-box {{
      margin-top: 14px;
      padding: 14px 15px;
      border-radius: 16px;
      background: rgba(36, 109, 115, 0.08);
      color: var(--text);
      border: 1px solid rgba(36, 109, 115, 0.14);
      line-height: 1.6;
      font-size: 14px;
    }}
    .warn-box {{
      background: rgba(184, 90, 56, 0.08);
      border-color: rgba(184, 90, 56, 0.14);
    }}
    .neighbor-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 13px;
    }}
    .neighbor-table th,
    .neighbor-table td {{
      padding: 9px 8px;
      border-bottom: 1px solid rgba(116, 86, 52, 0.08);
      text-align: right;
    }}
    .neighbor-table th:first-child,
    .neighbor-table td:first-child {{
      text-align: left;
    }}
    .neighbor-table th {{
      color: var(--muted);
      font-weight: 700;
    }}
    .footer-note {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }}
    @media (max-width: 1180px) {{
      .controls-grid,
      .main-grid,
      .bottom-grid,
      .control-columns {{
        grid-template-columns: 1fr;
      }}
      .plot {{
        min-height: 460px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <span class="eyebrow">Raw Case Dashboard</span>
      <h1>{html.escape(title)}</h1>
      <p>
        这个页面专门用来理解原始节点数据的空间分布。你可以切换 RTA、RMises、边界条件和梯度指标，
        再结合切片、点选、距离-应力曲线和邻域统计，判断热点、衰减规律和变化临界带是否合理。
      </p>
      <div class="hero-metrics" id="heroMetrics"></div>
    </section>

    <section class="grid controls-grid">
      <article class="card">
        <div class="card-header">
          <h2>视图控制</h2>
          <p class="helper">切换指标、切片和平面过滤，用同一个界面快速判断整体分布和临界区域。</p>
        </div>
        <div class="card-body control-columns">
          <div class="control-group">
            <label>
              颜色指标
              <select id="metricSelect"></select>
            </label>
            <label>
              边界条件过滤
              <select id="bcFilter">
                <option value="all">全部节点</option>
                <option value="free">只看自由点 (bc_mask = 0)</option>
                <option value="fixed">只看非自由点 (bc_mask != 0)</option>
              </select>
            </label>
            <label>
              距离曲线指标
              <select id="distanceMetricSelect"></select>
            </label>
            <label>
              高值阈值百分位
              <input id="percentileRange" type="range" min="0" max="99" step="1" value="0" />
              <div class="inline-value"><span id="percentileValue">0%</span></div>
            </label>
            <label>
              边界候选百分位
              <input id="boundaryPercentileRange" type="range" min="50" max="99" step="1" value="90" />
              <div class="inline-value"><span id="boundaryPercentileValue">90%</span></div>
            </label>
          </div>
          <div class="control-group">
            <label>
              切片轴
              <select id="sliceAxisSelect">
                <option value="none">不切片</option>
                <option value="x">X</option>
                <option value="y">Y</option>
                <option value="z">Z</option>
              </select>
            </label>
            <label>
              切片中心
              <input id="sliceCenterRange" type="range" min="0" max="1" step="0.001" value="0.5" />
              <div class="inline-value"><span id="sliceCenterValue">--</span></div>
            </label>
            <label>
              切片厚度
              <input id="sliceThicknessRange" type="range" min="1" max="100" step="1" value="20" />
              <div class="inline-value"><span id="sliceThicknessValue">20%</span></div>
            </label>
            <label>
              点透明度
              <input id="opacityRange" type="range" min="0.2" max="1" step="0.05" value="0.82" />
              <div class="inline-value"><span id="opacityValue">0.82</span></div>
            </label>
          </div>
        </div>
      </article>

      <article class="card">
        <div class="card-header">
          <h2>点选分析</h2>
          <p class="helper">输入 node_id 或直接点击 3D 图中的点。右侧会刷新该点的局部统计、最近邻和风险提示。</p>
        </div>
        <div class="card-body">
          <div class="button-row">
            <label style="flex: 1;">
              node_id
              <input id="nodeIdInput" type="number" step="1" />
            </label>
            <button id="jumpNodeButton">定位节点</button>
            <button id="jumpHotspotButton" class="secondary">跳到最大 RMises</button>
          </div>
          <div class="footer-note">
            “连接处”在当前原始数据里没有显式标签，这里只用 <code>graph_degree</code> 和局部梯度给出“连接/几何敏感候选”提示，不把它当成确定结论。
          </div>
        </div>
      </article>
    </section>

    <section class="grid main-grid">
      <article class="card">
        <div class="card-header">
          <h2>3D 原始场</h2>
          <p class="helper">点击点可查看该节点。切片会保留当前轴附近的节点，阈值会高亮当前指标较高的区域。</p>
        </div>
        <div class="card-body">
          <div id="mainPlot" class="plot"></div>
        </div>
      </article>

      <article class="card">
        <div class="card-header">
          <h2>节点详情</h2>
          <p class="helper">这里优先展示该点的物理值、边界条件、拓扑复杂度和局部变化强度。</p>
        </div>
        <div class="card-body">
          <ul class="stat-list" id="nodeStats"></ul>
          <div id="nodeHint" class="hint-box"></div>
          <table class="neighbor-table">
            <thead>
              <tr>
                <th>邻居 node_id</th>
                <th>距离</th>
                <th>RMises</th>
                <th>RTA</th>
                <th>bc_mask</th>
              </tr>
            </thead>
            <tbody id="neighborBody"></tbody>
          </table>
        </div>
      </article>
    </section>

    <section class="grid bottom-grid">
      <article class="card wide">
        <div class="card-header">
          <h2>2D 切面边界视图</h2>
          <p class="helper">打开切片后，这里会把切面投影到二维平面。底色是当前指标，黑色空心点是非自由点，橙色圆环是 RMises 高梯度候选边界。</p>
        </div>
        <div class="card-body">
          <div id="slicePlot" class="plot"></div>
        </div>
      </article>

      <article class="card">
        <div class="card-header">
          <h2>距离-响应曲线</h2>
          <p class="helper">横轴是到选中点的欧氏距离，纵轴是当前选择的距离曲线指标。散点帮助看离散分布，折线帮助看整体衰减趋势。</p>
        </div>
        <div class="card-body">
          <div id="distancePlot" class="plot"></div>
        </div>
      </article>

      <article class="card">
        <div class="card-header">
          <h2>当前指标分布</h2>
          <p class="helper">看当前过滤条件下，指标是集中在少量热点，还是呈现大范围平滑分布。</p>
        </div>
        <div class="card-body">
          <div id="histogramPlot" class="plot"></div>
        </div>
      </article>
    </section>
  </div>

  <script>
    const payload = {payload_json};
    const state = {{
      metric: payload.default_metric,
      distanceMetric: payload.distance_metric_default,
      bcFilter: "all",
      percentile: 0,
      sliceAxis: "none",
      sliceCenterNormalized: 0.5,
      sliceThicknessPercent: 20,
      boundaryPercentile: 90,
      opacity: 0.82,
      selectedNodeId: payload.default_node_id,
      selectedNodeIndex: null,
    }};

    const metricSelect = document.getElementById("metricSelect");
    const distanceMetricSelect = document.getElementById("distanceMetricSelect");
    const bcFilterSelect = document.getElementById("bcFilter");
    const percentileRange = document.getElementById("percentileRange");
    const percentileValue = document.getElementById("percentileValue");
    const boundaryPercentileRange = document.getElementById("boundaryPercentileRange");
    const boundaryPercentileValue = document.getElementById("boundaryPercentileValue");
    const sliceAxisSelect = document.getElementById("sliceAxisSelect");
    const sliceCenterRange = document.getElementById("sliceCenterRange");
    const sliceCenterValue = document.getElementById("sliceCenterValue");
    const sliceThicknessRange = document.getElementById("sliceThicknessRange");
    const sliceThicknessValue = document.getElementById("sliceThicknessValue");
    const opacityRange = document.getElementById("opacityRange");
    const opacityValue = document.getElementById("opacityValue");
    const nodeIdInput = document.getElementById("nodeIdInput");
    const jumpNodeButton = document.getElementById("jumpNodeButton");
    const jumpHotspotButton = document.getElementById("jumpHotspotButton");
    const heroMetrics = document.getElementById("heroMetrics");
    const nodeStats = document.getElementById("nodeStats");
    const nodeHint = document.getElementById("nodeHint");
    const neighborBody = document.getElementById("neighborBody");

    const coords = payload.coords;
    const metrics = payload.metrics;
    const nodeIds = payload.node_ids;
    const allIndices = Array.from({{ length: nodeIds.length }}, (_, index) => index);
    const nodeIdToIndex = new Map(nodeIds.map((nodeId, index) => [nodeId, index]));
    const constrainedSet = new Set(payload.constrained_indices);

    function formatNumber(value, digits = 6) {{
      if (!Number.isFinite(value)) {{
        return "--";
      }}
      return Number(value).toLocaleString("zh-CN", {{
        maximumFractionDigits: digits,
        minimumFractionDigits: Math.abs(value) >= 100 ? 0 : 0,
      }});
    }}

    function formatCompact(value) {{
      if (!Number.isFinite(value)) {{
        return "--";
      }}
      if (Math.abs(value) >= 10000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {{
        return Number(value).toExponential(3);
      }}
      return formatNumber(value, 6);
    }}

    function metricLabel(metric) {{
      const map = {{
        RMises: "RMises",
        RTA: "RTA",
        bc_mask: "bc_mask",
        graph_degree: "graph_degree",
        rmises_grad_max: "RMises 局部最大梯度",
        rmises_grad_mean: "RMises 局部平均梯度",
        rta_grad_max: "RTA 局部最大梯度",
        rta_grad_mean: "RTA 局部平均梯度",
      }};
      return map[metric] || metric;
    }}

    function initializeOptions() {{
      payload.metric_order.forEach(metric => {{
        const option = document.createElement("option");
        option.value = metric;
        option.textContent = metricLabel(metric);
        metricSelect.appendChild(option);

        const distanceOption = document.createElement("option");
        distanceOption.value = metric;
        distanceOption.textContent = metricLabel(metric);
        distanceMetricSelect.appendChild(distanceOption);
      }});
      metricSelect.value = state.metric;
      distanceMetricSelect.value = state.distanceMetric;
      nodeIdInput.value = state.selectedNodeId;
    }}

    function renderHeroMetrics() {{
      const rmisesSummary = payload.metric_summaries.RMises;
      const rtaSummary = payload.metric_summaries.RTA;
      const constrainedRatio = payload.node_count ? (payload.constrained_count / payload.node_count) * 100 : 0;
      const cards = [
        ["节点总数", formatNumber(payload.node_count, 0)],
        ["非自由点", `${{formatNumber(payload.constrained_count, 0)}} (${{formatNumber(constrainedRatio, 2)}}%)`],
        ["RMises 最大值", formatCompact(rmisesSummary.max)],
        ["RTA 最大值", formatCompact(rtaSummary.max)],
      ];
      heroMetrics.innerHTML = cards.map(([label, value]) => `
        <div class="hero-metric">
          <strong>${{label}}</strong>
          <span>${{value}}</span>
        </div>
      `).join("");
    }}

    function getAxisRange(axis) {{
      const values = coords[axis];
      let minValue = Number.POSITIVE_INFINITY;
      let maxValue = Number.NEGATIVE_INFINITY;
      for (const value of values) {{
        if (value < minValue) minValue = value;
        if (value > maxValue) maxValue = value;
      }}
      return {{ min: minValue, max: maxValue, span: Math.max(maxValue - minValue, 1e-9) }};
    }}

    function updateSliceLabels() {{
      sliceThicknessValue.textContent = `${{state.sliceThicknessPercent}}%`;
      opacityValue.textContent = Number(state.opacity).toFixed(2);
      percentileValue.textContent = `${{state.percentile}}%`;
      boundaryPercentileValue.textContent = `${{state.boundaryPercentile}}%`;

      if (state.sliceAxis === "none") {{
        sliceCenterValue.textContent = "未启用";
        return;
      }}
      const range = getAxisRange(state.sliceAxis);
      const center = range.min + range.span * state.sliceCenterNormalized;
      const halfThickness = 0.5 * range.span * state.sliceThicknessPercent / 100;
      sliceCenterValue.textContent = `${{state.sliceAxis.toUpperCase()}} = ${{formatCompact(center)}} ± ${{formatCompact(halfThickness)}}`;
    }}

    function computeThreshold(metricValues, indices, percentile) {{
      if (percentile <= 0 || indices.length === 0) {{
        return Number.NEGATIVE_INFINITY;
      }}
      const values = indices.map(index => metricValues[index]).sort((a, b) => a - b);
      const position = Math.max(0, Math.floor((percentile / 100) * (values.length - 1)));
      return values[position];
    }}

    function nodePassesSlice(index) {{
      if (state.sliceAxis === "none") {{
        return true;
      }}
      const range = getAxisRange(state.sliceAxis);
      const center = range.min + range.span * state.sliceCenterNormalized;
      const halfThickness = 0.5 * range.span * state.sliceThicknessPercent / 100;
      return Math.abs(coords[state.sliceAxis][index] - center) <= halfThickness;
    }}

    function nodePassesBc(index) {{
      const bcValue = metrics.bc_mask[index];
      if (state.bcFilter === "free") {{
        return bcValue === 0;
      }}
      if (state.bcFilter === "fixed") {{
        return bcValue !== 0;
      }}
      return true;
    }}

    function getFilteredIndices(sourceIndices, applyMetricThreshold = true) {{
      const metricValues = metrics[state.metric];
      const prefiltered = sourceIndices.filter(index => nodePassesBc(index) && nodePassesSlice(index));
      if (!applyMetricThreshold) {{
        return prefiltered;
      }}
      const threshold = computeThreshold(metricValues, prefiltered, state.percentile);
      return prefiltered.filter(index => metricValues[index] >= threshold);
    }}

    function getDisplayIndices() {{
      return getFilteredIndices(payload.display_indices, true);
    }}

    function getSliceProjectionConfig() {{
      if (state.sliceAxis === "x") {{
        return {{ horizontal: "y", vertical: "z", plane: "YZ" }};
      }}
      if (state.sliceAxis === "y") {{
        return {{ horizontal: "x", vertical: "z", plane: "XZ" }};
      }}
      return {{ horizontal: "x", vertical: "y", plane: "XY" }};
    }}

    function buildColorScale(metric) {{
      if (metric === "bc_mask") {{
        return [
          [0.0, "#ece8df"],
          [0.4999, "#ece8df"],
          [0.5, "#b85a38"],
          [1.0, "#b85a38"],
        ];
      }}
      return [
        [0.0, "#440154"],
        [0.25, "#3b528b"],
        [0.5, "#21918c"],
        [0.75, "#5ec962"],
        [1.0, "#fde725"],
      ];
    }}

    function render3DPlot() {{
      const indices = getDisplayIndices();
      const metricValues = metrics[state.metric];
      const values = indices.map(index => metricValues[index]);
      const summary = payload.metric_summaries[state.metric];
      const trace = {{
        type: "scatter3d",
        mode: "markers",
        x: indices.map(index => coords.x[index]),
        y: indices.map(index => coords.y[index]),
        z: indices.map(index => coords.z[index]),
        customdata: indices.map(index => nodeIds[index]),
        text: indices.map(index => [
          `node_id=${{nodeIds[index]}}`,
          `x=${{formatCompact(coords.x[index])}}`,
          `y=${{formatCompact(coords.y[index])}}`,
          `z=${{formatCompact(coords.z[index])}}`,
          `RMises=${{formatCompact(metrics.RMises[index])}}`,
          `RTA=${{formatCompact(metrics.RTA[index])}}`,
          `bc_mask=${{formatCompact(metrics.bc_mask[index])}}`,
        ].join("<br>")),
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{
          size: 3.4,
          opacity: state.opacity,
          color: values,
          colorscale: buildColorScale(state.metric),
          cmin: summary.min,
          cmax: summary.max,
          colorbar: {{
            title: metricLabel(state.metric),
            thickness: 18,
            len: 0.72,
            tickformat: ".5g",
          }},
        }},
      }};

      const selectedIndex = state.selectedNodeIndex;
      const selectedTrace = selectedIndex === null ? null : {{
        type: "scatter3d",
        mode: "markers",
        x: [coords.x[selectedIndex]],
        y: [coords.y[selectedIndex]],
        z: [coords.z[selectedIndex]],
        hovertemplate: `已选节点 node_id=${{nodeIds[selectedIndex]}}<extra></extra>`,
        marker: {{
          size: 8,
          color: "#d33636",
          line: {{
            width: 2,
            color: "#ffffff",
          }},
        }},
        showlegend: false,
      }};

      const constrainedIndices = indices.filter(index => constrainedSet.has(index));
      const constrainedTrace = constrainedIndices.length === 0 ? null : {{
        type: "scatter3d",
        mode: "markers",
        x: constrainedIndices.map(index => coords.x[index]),
        y: constrainedIndices.map(index => coords.y[index]),
        z: constrainedIndices.map(index => coords.z[index]),
        marker: {{
          size: 4.6,
          color: "rgba(25, 25, 25, 0)",
          line: {{
            width: 1.2,
            color: "#111111",
          }},
        }},
        hoverinfo: "skip",
        showlegend: false,
      }};

      const traces = [trace];
      if (constrainedTrace) traces.push(constrainedTrace);
      if (selectedTrace) traces.push(selectedTrace);

      const layout = {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: {{ l: 0, r: 0, t: 8, b: 0 }},
        scene: {{
          xaxis: {{
            title: "x",
            backgroundcolor: "#fbf9f5",
            gridcolor: "#ddd4c8",
            zerolinecolor: "#cec1b1",
          }},
          yaxis: {{
            title: "y",
            backgroundcolor: "#fbf9f5",
            gridcolor: "#ddd4c8",
            zerolinecolor: "#cec1b1",
          }},
          zaxis: {{
            title: "z",
            backgroundcolor: "#fbf9f5",
            gridcolor: "#ddd4c8",
            zerolinecolor: "#cec1b1",
          }},
          aspectmode: "data",
          camera: {{
            eye: {{ x: 1.55, y: 1.45, z: 0.9 }},
          }},
        }},
        annotations: indices.length === 0 ? [{{
          text: "当前过滤条件下没有节点",
          showarrow: false,
          x: 0.5,
          y: 0.5,
          xref: "paper",
          yref: "paper",
          font: {{ size: 18, color: "#8f2e2e" }},
        }}] : [],
      }};
      Plotly.react("mainPlot", traces, layout, {{
        responsive: true,
        displaylogo: false,
        scrollZoom: true,
      }});
    }}

    function renderSlicePlot() {{
      if (state.sliceAxis === "none") {{
        Plotly.react("slicePlot", [], {{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: {{ l: 20, r: 20, t: 20, b: 20 }},
          annotations: [{{
            text: "先选择切片轴，再在这里观察切面上的临界边界。",
            showarrow: false,
            x: 0.5,
            y: 0.5,
            xref: "paper",
            yref: "paper",
            font: {{ size: 18, color: "#8f2e2e" }},
          }}],
          xaxis: {{ visible: false }},
          yaxis: {{ visible: false }},
        }}, {{
          responsive: true,
          displaylogo: false,
        }});
        return;
      }}

      const projection = getSliceProjectionConfig();
      const sliceIndices = getFilteredIndices(allIndices, false);
      const metricValues = metrics[state.metric];
      const gradientValues = metrics.rmises_grad_max;
      const summary = payload.metric_summaries[state.metric];

      if (!sliceIndices.length) {{
        Plotly.react("slicePlot", [], {{
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          margin: {{ l: 20, r: 20, t: 20, b: 20 }},
          annotations: [{{
            text: "当前切片和边界条件过滤下没有节点。",
            showarrow: false,
            x: 0.5,
            y: 0.5,
            xref: "paper",
            yref: "paper",
            font: {{ size: 18, color: "#8f2e2e" }},
          }}],
          xaxis: {{ visible: false }},
          yaxis: {{ visible: false }},
        }}, {{
          responsive: true,
          displaylogo: false,
        }});
        return;
      }}

      const boundaryThreshold = computeThreshold(gradientValues, sliceIndices, state.boundaryPercentile);
      const metricThreshold = computeThreshold(metricValues, sliceIndices, state.percentile);
      const boundaryIndices = sliceIndices.filter(index => gradientValues[index] >= boundaryThreshold);
      const constrainedIndices = sliceIndices.filter(index => metrics.bc_mask[index] !== 0);
      const hotspotIndices = state.percentile > 0
        ? sliceIndices.filter(index => metricValues[index] >= metricThreshold)
        : [];

      const baseTrace = {{
        type: "scattergl",
        mode: "markers",
        x: sliceIndices.map(index => coords[projection.horizontal][index]),
        y: sliceIndices.map(index => coords[projection.vertical][index]),
        customdata: sliceIndices.map(index => nodeIds[index]),
        text: sliceIndices.map(index => [
          `node_id=${{nodeIds[index]}}`,
          `${{projection.horizontal}}=${{formatCompact(coords[projection.horizontal][index])}}`,
          `${{projection.vertical}}=${{formatCompact(coords[projection.vertical][index])}}`,
          `RMises=${{formatCompact(metrics.RMises[index])}}`,
          `RTA=${{formatCompact(metrics.RTA[index])}}`,
          `bc_mask=${{formatCompact(metrics.bc_mask[index])}}`,
          `RMises局部最大梯度=${{formatCompact(metrics.rmises_grad_max[index])}}`,
        ].join("<br>")),
        hovertemplate: "%{{text}}<extra></extra>",
        marker: {{
          size: 7,
          opacity: Math.min(state.opacity + 0.08, 0.95),
          color: sliceIndices.map(index => metricValues[index]),
          colorscale: buildColorScale(state.metric),
          cmin: summary.min,
          cmax: summary.max,
          colorbar: {{
            title: metricLabel(state.metric),
            tickformat: ".5g",
            thickness: 16,
            len: 0.82,
          }},
        }},
        name: metricLabel(state.metric),
      }};

      const boundaryTrace = boundaryIndices.length ? {{
        type: "scatter",
        mode: "markers",
        x: boundaryIndices.map(index => coords[projection.horizontal][index]),
        y: boundaryIndices.map(index => coords[projection.vertical][index]),
        customdata: boundaryIndices.map(index => nodeIds[index]),
        hovertemplate: "高梯度候选边界<br>node_id=%{{customdata}}<extra></extra>",
        marker: {{
          size: 12,
          color: "rgba(255, 140, 0, 0)",
          line: {{
            width: 2,
            color: "#e37d11",
          }},
          symbol: "circle-open",
        }},
        name: `RMises 高梯度 top ${{100 - state.boundaryPercentile}}%`,
      }} : null;

      const constrainedTrace = constrainedIndices.length ? {{
        type: "scatter",
        mode: "markers",
        x: constrainedIndices.map(index => coords[projection.horizontal][index]),
        y: constrainedIndices.map(index => coords[projection.vertical][index]),
        customdata: constrainedIndices.map(index => nodeIds[index]),
        hovertemplate: "非自由点<br>node_id=%{{customdata}}<extra></extra>",
        marker: {{
          size: 10,
          color: "rgba(0, 0, 0, 0)",
          line: {{
            width: 1.5,
            color: "#161616",
          }},
          symbol: "square-open",
        }},
        name: "bc_mask != 0",
      }} : null;

      const hotspotTrace = hotspotIndices.length ? {{
        type: "scatter",
        mode: "markers",
        x: hotspotIndices.map(index => coords[projection.horizontal][index]),
        y: hotspotIndices.map(index => coords[projection.vertical][index]),
        customdata: hotspotIndices.map(index => nodeIds[index]),
        hovertemplate: "当前指标高值带<br>node_id=%{{customdata}}<extra></extra>",
        marker: {{
          size: 9,
          color: "rgba(255, 255, 255, 0)",
          line: {{
            width: 1.2,
            color: "#fff7d6",
          }},
          symbol: "diamond-open",
        }},
        name: `当前指标高值 >= P${{state.percentile}}`,
      }} : null;

      const selectedVisible = state.selectedNodeIndex !== null && sliceIndices.includes(state.selectedNodeIndex);
      const selectedTrace = selectedVisible ? {{
        type: "scatter",
        mode: "markers",
        x: [coords[projection.horizontal][state.selectedNodeIndex]],
        y: [coords[projection.vertical][state.selectedNodeIndex]],
        customdata: [nodeIds[state.selectedNodeIndex]],
        hovertemplate: "当前选中节点 node_id=%{{customdata}}<extra></extra>",
        marker: {{
          size: 16,
          color: "#d33636",
          line: {{
            width: 2,
            color: "#ffffff",
          }},
          symbol: "x",
        }},
        name: "当前节点",
      }} : null;

      const traces = [baseTrace];
      if (boundaryTrace) traces.push(boundaryTrace);
      if (constrainedTrace) traces.push(constrainedTrace);
      if (hotspotTrace) traces.push(hotspotTrace);
      if (selectedTrace) traces.push(selectedTrace);

      const layout = {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: {{ l: 58, r: 18, t: 12, b: 56 }},
        xaxis: {{
          title: projection.horizontal,
          gridcolor: "#ddd4c8",
          zerolinecolor: "#cec1b1",
          scaleanchor: "y",
          scaleratio: 1,
        }},
        yaxis: {{
          title: projection.vertical,
          gridcolor: "#ddd4c8",
          zerolinecolor: "#cec1b1",
        }},
        legend: {{
          orientation: "h",
          x: 0,
          y: 1.12,
        }},
        annotations: [{{
          text: `${{projection.plane}} 切面 | 橙圈 = RMises 高梯度候选边界 | 黑框 = 非自由点`,
          showarrow: false,
          x: 0,
          y: 1.02,
          xref: "paper",
          yref: "paper",
          xanchor: "left",
          font: {{ size: 13, color: "#6b5f51" }},
        }}],
      }};

      Plotly.react("slicePlot", traces, layout, {{
        responsive: true,
        displaylogo: false,
      }});
    }}

    function pickNearestNeighbors(targetIndex, count) {{
      const distances = [];
      const x0 = coords.x[targetIndex];
      const y0 = coords.y[targetIndex];
      const z0 = coords.z[targetIndex];
      for (let index = 0; index < nodeIds.length; index += 1) {{
        if (index === targetIndex) {{
          continue;
        }}
        const dx = coords.x[index] - x0;
        const dy = coords.y[index] - y0;
        const dz = coords.z[index] - z0;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        distances.push([distance, index]);
      }}
      distances.sort((left, right) => left[0] - right[0]);
      return distances.slice(0, count);
    }}

    function computeNearestConstraintDistance(targetIndex) {{
      if (!payload.constrained_indices.length) {{
        return null;
      }}
      const x0 = coords.x[targetIndex];
      const y0 = coords.y[targetIndex];
      const z0 = coords.z[targetIndex];
      let best = Number.POSITIVE_INFINITY;
      for (const index of payload.constrained_indices) {{
        if (index === targetIndex) {{
          return 0;
        }}
        const dx = coords.x[index] - x0;
        const dy = coords.y[index] - y0;
        const dz = coords.z[index] - z0;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (distance < best) {{
          best = distance;
        }}
      }}
      return best;
    }}

    function updateNodePanel() {{
      const index = state.selectedNodeIndex;
      if (index === null) {{
        return;
      }}
      const nearest = pickNearestNeighbors(index, 8);
      const neighborhood = pickNearestNeighbors(index, 32).map(item => item[1]);
      const currentMetricValues = neighborhood.map(neighborIndex => metrics[state.metric][neighborIndex]);
      const rmisesNeighborhood = neighborhood.map(neighborIndex => metrics.RMises[neighborIndex]);
      const nearestConstraintDistance = computeNearestConstraintDistance(index);

      const statRows = [
        ["node_id", nodeIds[index]],
        ["x", formatCompact(coords.x[index])],
        ["y", formatCompact(coords.y[index])],
        ["z", formatCompact(coords.z[index])],
        ["RMises", formatCompact(metrics.RMises[index])],
        ["RTA", formatCompact(metrics.RTA[index])],
        ["bc_mask", formatCompact(metrics.bc_mask[index])],
        ["graph_degree", formatCompact(metrics.graph_degree[index])],
        ["RMises 局部最大梯度", formatCompact(metrics.rmises_grad_max[index])],
        [`${{metricLabel(state.metric)}} 邻域均值`, formatCompact(currentMetricValues.length ? currentMetricValues.reduce((sum, value) => sum + value, 0) / currentMetricValues.length : NaN)],
        ["最近非自由点距离", nearestConstraintDistance === null ? "--" : formatCompact(nearestConstraintDistance)],
        ["邻域 RMises 最大值", formatCompact(rmisesNeighborhood.length ? Math.max(...rmisesNeighborhood) : NaN)],
      ];
      nodeStats.innerHTML = statRows.map(([label, value]) => `
        <li>
          <strong>${{label}}</strong>
          <span>${{value}}</span>
        </li>
      `).join("");

      const suspiciousDegree = metrics.graph_degree[index] >= payload.heuristic_thresholds.graph_degree_p90;
      const suspiciousGradient = metrics.rmises_grad_max[index] >= payload.heuristic_thresholds.rmises_grad_max_p90;
      let hintText = "";
      let hintClass = "hint-box";
      if (metrics.bc_mask[index] !== 0) {{
        hintText += "该点本身是非自由点，通常值得和高应力区的空间位置一起看。";
      }} else {{
        hintText += "该点本身是自由点，更适合结合与最近非自由点的距离一起判断衰减是否合理。";
      }}
      if (suspiciousDegree && suspiciousGradient) {{
        hintText += " 这个点的拓扑度数和 RMises 局部梯度都偏高，可以把它看成连接/几何敏感候选区，但这不是明确的连接标签。";
        hintClass += " warn-box";
      }} else if (suspiciousGradient) {{
        hintText += " 这个点的局部梯度偏高，适合重点看是否处于应力变化临界带。";
      }} else if (suspiciousDegree) {{
        hintText += " 这个点的 graph_degree 偏高，说明它附近的网格连通更复杂，可以辅助判断是否靠近结构交汇区域。";
      }} else {{
        hintText += " 从拓扑和局部梯度看，这个点更像普通区域节点。";
      }}
      nodeHint.className = hintClass;
      nodeHint.textContent = hintText;

      neighborBody.innerHTML = nearest.map(([distance, neighborIndex]) => `
        <tr>
          <td>${{nodeIds[neighborIndex]}}</td>
          <td>${{formatCompact(distance)}}</td>
          <td>${{formatCompact(metrics.RMises[neighborIndex])}}</td>
          <td>${{formatCompact(metrics.RTA[neighborIndex])}}</td>
          <td>${{formatCompact(metrics.bc_mask[neighborIndex])}}</td>
        </tr>
      `).join("");
    }}

    function buildDistanceSeries(targetIndex, metricName) {{
      const x0 = coords.x[targetIndex];
      const y0 = coords.y[targetIndex];
      const z0 = coords.z[targetIndex];
      const metricValues = metrics[metricName];

      const allDistances = new Array(nodeIds.length);
      for (let index = 0; index < nodeIds.length; index += 1) {{
        const dx = coords.x[index] - x0;
        const dy = coords.y[index] - y0;
        const dz = coords.z[index] - z0;
        allDistances[index] = Math.sqrt(dx * dx + dy * dy + dz * dz);
      }}

      const scatterIndices = payload.distance_sample_indices.slice();
      if (!scatterIndices.includes(targetIndex)) {{
        scatterIndices.push(targetIndex);
      }}
      const scatterX = scatterIndices.map(index => allDistances[index]);
      const scatterY = scatterIndices.map(index => metricValues[index]);
      const maxDistance = Math.max(...allDistances);
      const binCount = 60;
      const binWidth = Math.max(maxDistance / binCount, 1e-9);
      const sums = new Array(binCount).fill(0);
      const counts = new Array(binCount).fill(0);
      for (let index = 0; index < allDistances.length; index += 1) {{
        const distance = allDistances[index];
        const bin = Math.min(binCount - 1, Math.floor(distance / binWidth));
        sums[bin] += metricValues[index];
        counts[bin] += 1;
      }}
      const lineX = [];
      const lineY = [];
      for (let bin = 0; bin < binCount; bin += 1) {{
        if (!counts[bin]) {{
          continue;
        }}
        lineX.push((bin + 0.5) * binWidth);
        lineY.push(sums[bin] / counts[bin]);
      }}

      return {{ scatterX, scatterY, lineX, lineY }};
    }}

    function renderDistancePlot() {{
      const targetIndex = state.selectedNodeIndex;
      if (targetIndex === null) {{
        return;
      }}
      const metricName = state.distanceMetric;
      const series = buildDistanceSeries(targetIndex, metricName);
      const traces = [
        {{
          type: "scatter",
          mode: "markers",
          x: series.scatterX,
          y: series.scatterY,
          marker: {{
            size: 5,
            color: "rgba(36, 109, 115, 0.36)",
          }},
          name: "抽样散点",
          hovertemplate: "距离=%{{x:.5g}}<br>值=%{{y:.5g}}<extra></extra>",
        }},
        {{
          type: "scatter",
          mode: "lines",
          x: series.lineX,
          y: series.lineY,
          line: {{
            width: 3,
            color: "#b85a38",
          }},
          name: "分桶均值",
          hovertemplate: "距离=%{{x:.5g}}<br>均值=%{{y:.5g}}<extra></extra>",
        }},
      ];
      const layout = {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: {{ l: 55, r: 18, t: 20, b: 52 }},
        xaxis: {{
          title: "到选中点的欧氏距离",
          gridcolor: "#ddd4c8",
        }},
        yaxis: {{
          title: metricLabel(metricName),
          gridcolor: "#ddd4c8",
        }},
        legend: {{
          orientation: "h",
          x: 0,
          y: 1.1,
        }},
      }};
      Plotly.react("distancePlot", traces, layout, {{
        responsive: true,
        displaylogo: false,
      }});
    }}

    function renderHistogram() {{
      const indices = getDisplayIndices();
      const values = indices.map(index => metrics[state.metric][index]);
      const selectedIndex = state.selectedNodeIndex;
      const selectedValue = selectedIndex === null ? null : metrics[state.metric][selectedIndex];
      const traces = [{{
        type: "histogram",
        x: values,
        marker: {{
          color: "#246d73",
          line: {{
            color: "#ffffff",
            width: 0.4,
          }},
        }},
        nbinsx: 42,
        hovertemplate: "区间=%{{x}}<br>计数=%{{y}}<extra></extra>",
      }}];
      if (selectedValue !== null) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: [selectedValue, selectedValue],
          y: [0, Math.max(values.length / 6, 1)],
          line: {{
            width: 3,
            color: "#b33535",
            dash: "dash",
          }},
          name: "当前节点",
          hovertemplate: "当前节点值=%{{x:.5g}}<extra></extra>",
        }});
      }}
      const layout = {{
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: {{ l: 48, r: 18, t: 20, b: 48 }},
        xaxis: {{
          title: metricLabel(state.metric),
          gridcolor: "#ddd4c8",
        }},
        yaxis: {{
          title: "节点数",
          gridcolor: "#ddd4c8",
        }},
        showlegend: selectedValue !== null,
      }};
      Plotly.react("histogramPlot", traces, layout, {{
        responsive: true,
        displaylogo: false,
      }});
    }}

    function setSelectedNodeById(nodeId) {{
      if (!nodeIdToIndex.has(nodeId)) {{
        window.alert(`没有找到 node_id=${{nodeId}}`);
        return;
      }}
      state.selectedNodeId = nodeId;
      state.selectedNodeIndex = nodeIdToIndex.get(nodeId);
      nodeIdInput.value = nodeId;
      renderAll();
    }}

    function renderAll() {{
      updateSliceLabels();
      render3DPlot();
      renderSlicePlot();
      updateNodePanel();
      renderDistancePlot();
      renderHistogram();
    }}

    function bootstrapState() {{
      state.selectedNodeIndex = nodeIdToIndex.get(state.selectedNodeId) ?? 0;
      const defaultAxisRange = getAxisRange("z");
      const defaultCenter = coords.z[state.selectedNodeIndex];
      state.sliceCenterNormalized = defaultAxisRange.span ? (defaultCenter - defaultAxisRange.min) / defaultAxisRange.span : 0.5;
      renderHeroMetrics();
      initializeOptions();
      updateSliceLabels();
    }}

    metricSelect.addEventListener("change", event => {{
      state.metric = event.target.value;
      renderAll();
    }});

    distanceMetricSelect.addEventListener("change", event => {{
      state.distanceMetric = event.target.value;
      renderDistancePlot();
    }});

    bcFilterSelect.addEventListener("change", event => {{
      state.bcFilter = event.target.value;
      renderAll();
    }});

    percentileRange.addEventListener("input", event => {{
      state.percentile = Number(event.target.value);
      percentileValue.textContent = `${{state.percentile}}%`;
      renderAll();
    }});

    boundaryPercentileRange.addEventListener("input", event => {{
      state.boundaryPercentile = Number(event.target.value);
      boundaryPercentileValue.textContent = `${{state.boundaryPercentile}}%`;
      renderSlicePlot();
    }});

    sliceAxisSelect.addEventListener("change", event => {{
      state.sliceAxis = event.target.value;
      if (state.sliceAxis !== "none") {{
        const range = getAxisRange(state.sliceAxis);
        const currentCoordinate = coords[state.sliceAxis][state.selectedNodeIndex];
        state.sliceCenterNormalized = range.span ? (currentCoordinate - range.min) / range.span : 0.5;
      }}
      renderAll();
    }});

    sliceCenterRange.addEventListener("input", event => {{
      state.sliceCenterNormalized = Number(event.target.value);
      renderAll();
    }});

    sliceThicknessRange.addEventListener("input", event => {{
      state.sliceThicknessPercent = Number(event.target.value);
      renderAll();
    }});

    opacityRange.addEventListener("input", event => {{
      state.opacity = Number(event.target.value);
      renderAll();
    }});

    jumpNodeButton.addEventListener("click", () => {{
      setSelectedNodeById(Number(nodeIdInput.value));
    }});

    jumpHotspotButton.addEventListener("click", () => {{
      setSelectedNodeById(payload.default_node_id);
    }});

    bootstrapState();
    renderAll();
    document.getElementById("mainPlot").on("plotly_click", event => {{
      const point = event?.points?.[0];
      if (!point) {{
        return;
      }}
      setSelectedNodeById(Number(point.customdata));
    }});
    document.getElementById("slicePlot").on("plotly_click", event => {{
      const point = event?.points?.[0];
      if (!point || point.customdata === undefined) {{
        return;
      }}
      setSelectedNodeById(Number(point.customdata));
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nodes_df, edges_df = _load_case(case_dir)
    nodes_df, metrics, summaries = _prepare_metrics(nodes_df, edges_df)
    payload = _build_payload(
        case_dir=case_dir,
        nodes_df=nodes_df,
        metrics=metrics,
        metric_summaries=summaries,
        display_sample_size=int(args.display_sample_size),
        distance_scatter_sample_size=int(args.distance_scatter_sample_size),
        seed=int(args.seed),
    )
    title = args.title or f"{case_dir.name} | Raw Data Dashboard"
    html_document = _build_html_document(title=title, payload=payload, plotly_cdn=args.plotly_cdn)
    output_path.write_text(html_document, encoding="utf-8")
    print(
        json.dumps(
            {
                "case_dir": str(case_dir),
                "output": str(output_path),
                "node_count": payload["node_count"],
                "metric_count": len(metrics),
                "display_sample_size": min(int(args.display_sample_size), int(payload["node_count"])),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
