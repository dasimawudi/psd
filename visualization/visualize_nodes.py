from __future__ import annotations

from pathlib import Path

import argparse
import html
import json
import math

import numpy as np
import pandas as pd


DEFAULT_VIEWS = ("xy", "xz", "yz")
SEQUENTIAL_STOPS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)
DIVERGING_STOPS = (
    (49, 54, 149),
    (69, 117, 180),
    (224, 243, 248),
    (254, 224, 144),
    (215, 48, 39),
    (165, 0, 38),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize node coordinates and a numeric value column from nodes.csv or field_prediction.csv."
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV containing x/y/z and one numeric value column.")
    parser.add_argument("--value-column", type=str, required=True, help="Column used for color mapping, e.g. RMises.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for SVG and HTML outputs.")
    parser.add_argument("--x-column", type=str, default="x", help="X coordinate column name.")
    parser.add_argument("--y-column", type=str, default="y", help="Y coordinate column name.")
    parser.add_argument("--z-column", type=str, default="z", help="Z coordinate column name.")
    parser.add_argument(
        "--views",
        type=str,
        nargs="+",
        default=list(DEFAULT_VIEWS),
        choices=list(DEFAULT_VIEWS),
        help="Projection views to render.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional row count to sample before plotting. Useful for very large meshes.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used with --sample-size.")
    parser.add_argument("--point-radius", type=float, default=1.4, help="SVG point radius in pixels.")
    parser.add_argument("--point-opacity", type=float, default=0.78, help="Point opacity between 0 and 1.")
    parser.add_argument("--color-min", type=float, default=None, help="Manual lower bound for color mapping.")
    parser.add_argument("--color-max", type=float, default=None, help="Manual upper bound for color mapping.")
    parser.add_argument(
        "--center-zero",
        action="store_true",
        help="Use a diverging palette centered at 0. Helpful for signed values.",
    )
    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip generating the interactive 3D HTML view.",
    )
    parser.add_argument(
        "--plotly-cdn",
        type=str,
        default="https://cdn.plot.ly/plotly-2.35.2.min.js",
        help="Plotly CDN URL used by the generated 3D HTML.",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional title shown in the outputs.")
    return parser.parse_args()


def _format_number(value: float) -> str:
    return f"{value:.6g}"


def _hex_color(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _interpolate_color(position: float, stops: tuple[tuple[int, int, int], ...]) -> str:
    clamped = min(max(float(position), 0.0), 1.0)
    if len(stops) == 1:
        return _hex_color(stops[0])
    scaled = clamped * (len(stops) - 1)
    left_index = int(math.floor(scaled))
    right_index = min(left_index + 1, len(stops) - 1)
    frac = scaled - left_index
    left = stops[left_index]
    right = stops[right_index]
    blended = tuple(int(round(lv + (rv - lv) * frac)) for lv, rv in zip(left, right))
    return _hex_color(blended)


def _normalize_value(value: float, value_min: float, value_max: float, center_zero: bool) -> float:
    if not math.isfinite(value):
        return 0.5
    if math.isclose(value_min, value_max):
        return 0.5
    if center_zero and value_min < 0.0 < value_max:
        if value >= 0.0:
            return 0.5 + 0.5 * (value / value_max if value_max else 0.0)
        return 0.5 * ((value - value_min) / (0.0 - value_min))
    return (value - value_min) / (value_max - value_min)


def _active_stops(value_min: float, value_max: float, center_zero: bool) -> tuple[tuple[int, int, int], ...]:
    return DIVERGING_STOPS if center_zero and value_min < 0.0 < value_max else SEQUENTIAL_STOPS


def _value_to_color(value: float, value_min: float, value_max: float, center_zero: bool) -> str:
    return _interpolate_color(
        _normalize_value(value, value_min, value_max, center_zero),
        _active_stops(value_min, value_max, center_zero),
    )


def _build_plotly_colorscale(value_min: float, value_max: float, center_zero: bool) -> list[list[float | str]]:
    stops = _active_stops(value_min, value_max, center_zero)
    if len(stops) == 1:
        color = _hex_color(stops[0])
        return [[0.0, color], [1.0, color]]
    return [[index / (len(stops) - 1), _hex_color(stop)] for index, stop in enumerate(stops)]


def _read_frame(csv_path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    missing = [column for column in columns if column not in header.columns]
    if missing:
        available = ", ".join(str(column) for column in header.columns)
        raise KeyError(f"Missing columns {missing} in {csv_path}. Available columns: {available}")

    frame = pd.read_csv(csv_path, usecols=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=columns).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No valid numeric rows remain after reading {csv_path}")
    return frame


def _sample_frame(frame: pd.DataFrame, sample_size: int | None, random_seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(frame):
        return frame
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive when provided")
    return frame.sample(n=sample_size, random_state=random_seed).sort_index().reset_index(drop=True)


def _project_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    width: int,
    height: int,
    left_margin: int,
    right_margin: int,
    top_margin: int,
    bottom_margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))

    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)

    scale = min(plot_width / x_span, plot_height / y_span)
    used_width = x_span * scale
    used_height = y_span * scale
    x_offset = left_margin + 0.5 * (plot_width - used_width)
    y_offset = top_margin + 0.5 * (plot_height - used_height)

    x_px = x_offset + (x_values - x_min) * scale
    y_px = y_offset + used_height - (y_values - y_min) * scale
    return x_px, y_px


def _build_colorbar(
    value_min: float,
    value_max: float,
    center_zero: bool,
    left: int,
    top: int,
    height: int,
    width: int,
    label: str,
) -> str:
    parts: list[str] = []
    steps = 120
    for step in range(steps):
        start_ratio = step / steps
        end_ratio = (step + 1) / steps
        y = top + (1.0 - end_ratio) * height
        rect_height = max(height / steps + 0.2, 1.0)
        value = value_min + (value_max - value_min) * start_ratio
        color = _value_to_color(value, value_min, value_max, center_zero)
        parts.append(
            f'<rect x="{left}" y="{y:.2f}" width="{width}" height="{rect_height:.2f}" fill="{color}" stroke="none" />'
        )

    parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#444" stroke-width="1" />')
    parts.append(
        f'<text x="{left + width / 2:.1f}" y="{top - 12}" font-size="16" text-anchor="middle" fill="#222">{html.escape(label)}</text>'
    )
    parts.append(
        f'<text x="{left + width + 10}" y="{top + 5}" font-size="13" fill="#222">{html.escape(_format_number(value_max))}</text>'
    )
    if center_zero and value_min < 0.0 < value_max:
        zero_y = top + height * (1.0 - _normalize_value(0.0, value_min, value_max, center_zero))
        parts.append(
            f'<line x1="{left - 6}" y1="{zero_y:.2f}" x2="{left + width + 6}" y2="{zero_y:.2f}" stroke="#333" stroke-dasharray="4 3" />'
        )
        parts.append(
            f'<text x="{left + width + 10}" y="{zero_y + 5:.2f}" font-size="13" fill="#222">0</text>'
        )
    parts.append(
        f'<text x="{left + width + 10}" y="{top + height + 5}" font-size="13" fill="#222">{html.escape(_format_number(value_min))}</text>'
    )
    return "\n".join(parts)


def _write_projection_svg(
    output_path: Path,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    value_min: float,
    value_max: float,
    center_zero: bool,
    point_radius: float,
    point_opacity: float,
    title: str,
    subtitle: str,
) -> None:
    width = 1120
    height = 820
    left_margin = 88
    right_margin = 170
    top_margin = 92
    bottom_margin = 82
    plot_right = width - right_margin
    plot_bottom = height - bottom_margin

    x_values = frame[x_column].to_numpy(dtype=np.float64)
    y_values = frame[y_column].to_numpy(dtype=np.float64)
    value_values = frame[value_column].to_numpy(dtype=np.float64)
    x_px, y_px = _project_points(
        x_values=x_values,
        y_values=y_values,
        width=width,
        height=height,
        left_margin=left_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd" />',
        f'<rect x="{left_margin}" y="{top_margin}" width="{plot_right - left_margin}" height="{plot_bottom - top_margin}" fill="#ffffff" stroke="#b8bcc7" stroke-width="1.2" />',
        f'<text x="{width / 2:.1f}" y="42" font-size="28" text-anchor="middle" fill="#111">{html.escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="68" font-size="15" text-anchor="middle" fill="#555">{html.escape(subtitle)}</text>',
    ]

    for x_coord, y_coord, value in zip(x_px, y_px, value_values):
        color = _value_to_color(value, value_min, value_max, center_zero)
        parts.append(
            f'<circle cx="{x_coord:.2f}" cy="{y_coord:.2f}" r="{point_radius:.2f}" fill="{color}" fill-opacity="{point_opacity:.3f}" stroke="none" />'
        )

    parts.extend(
        [
            f'<text x="{(left_margin + plot_right) / 2:.1f}" y="{height - 26}" font-size="20" text-anchor="middle" fill="#222">{html.escape(x_column)}</text>',
            f'<text x="32" y="{height / 2:.1f}" font-size="20" text-anchor="middle" fill="#222" transform="rotate(-90 32 {height / 2:.1f})">{html.escape(y_column)}</text>',
            f'<text x="{left_margin}" y="{height - 8}" font-size="13" fill="#444">min={html.escape(_format_number(float(np.min(x_values))))}</text>',
            f'<text x="{plot_right - 10}" y="{height - 8}" font-size="13" text-anchor="end" fill="#444">max={html.escape(_format_number(float(np.max(x_values))))}</text>',
            f'<text x="{left_margin - 12}" y="{plot_bottom + 4}" font-size="13" text-anchor="end" fill="#444">min={html.escape(_format_number(float(np.min(y_values))))}</text>',
            f'<text x="{left_margin - 12}" y="{top_margin + 5}" font-size="13" text-anchor="end" fill="#444">max={html.escape(_format_number(float(np.max(y_values))))}</text>',
            _build_colorbar(
                value_min=value_min,
                value_max=value_max,
                center_zero=center_zero,
                left=width - 104,
                top=120,
                height=560,
                width=28,
                label=value_column,
            ),
            "</svg>",
        ]
    )

    output_path.write_text("\n".join(parts), encoding="utf-8")


def _write_3d_html(
    output_path: Path,
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    value_column: str,
    value_min: float,
    value_max: float,
    center_zero: bool,
    point_opacity: float,
    point_radius: float,
    title: str,
    subtitle: str,
    plotly_cdn: str,
) -> None:
    x_values = frame[x_column].to_numpy(dtype=np.float64).tolist()
    y_values = frame[y_column].to_numpy(dtype=np.float64).tolist()
    z_values = frame[z_column].to_numpy(dtype=np.float64).tolist()
    value_values = frame[value_column].to_numpy(dtype=np.float64).tolist()
    hover_text = [
        "<br>".join(
            [
                f"{html.escape(x_column)}={_format_number(float(x_value))}",
                f"{html.escape(y_column)}={_format_number(float(y_value))}",
                f"{html.escape(z_column)}={_format_number(float(z_value))}",
                f"{html.escape(value_column)}={_format_number(float(value_value))}",
            ]
        )
        for x_value, y_value, z_value, value_value in zip(x_values, y_values, z_values, value_values)
    ]
    payload = {
        "x": x_values,
        "y": y_values,
        "z": z_values,
        "value": value_values,
        "hover_text": hover_text,
        "colorscale": _build_plotly_colorscale(value_min=value_min, value_max=value_max, center_zero=center_zero),
        "scene_range": {
            "x": [float(np.min(frame[x_column])), float(np.max(frame[x_column]))],
            "y": [float(np.min(frame[y_column])), float(np.max(frame[y_column]))],
            "z": [float(np.min(frame[z_column])), float(np.max(frame[z_column]))],
        },
        "value_min": value_min,
        "value_max": value_max,
        "value_column": value_column,
        "x_column": x_column,
        "y_column": y_column,
        "z_column": z_column,
        "title": title,
        "subtitle": subtitle,
        "point_opacity": point_opacity,
        "point_size": max(point_radius * 3.0, 2.0),
        "center_zero": bool(center_zero and value_min < 0.0 < value_max),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} | 3D</title>
  <script src="{html.escape(plotly_cdn, quote=True)}"></script>
  <style>
    :root {{
      color-scheme: light;
      --bg: #eef1f7;
      --card: rgba(255, 255, 255, 0.9);
      --text: #151925;
      --muted: #5c6678;
      --border: rgba(159, 169, 187, 0.35);
      --shadow: 0 24px 60px rgba(18, 25, 39, 0.15);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(107, 160, 255, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 195, 113, 0.2), transparent 24%),
        linear-gradient(180deg, #f8faff 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1480px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 24px 0 28px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .hero {{
      padding: 22px 26px;
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: 30px;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.6;
      font-size: 15px;
    }}
    .plot-wrap {{
      padding: 10px;
    }}
    #plot {{
      width: 100%;
      height: min(78vh, 920px);
      min-height: 620px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel hero">
      <h1>{html.escape(title)} | 3D</h1>
      <p>{html.escape(subtitle)}。鼠标左键旋转，滚轮缩放，悬停可查看节点坐标和值。</p>
    </section>
    <section class="panel plot-wrap">
      <div id="plot"></div>
    </section>
  </main>
  <script>
    const payload = {payload_json};
    const trace = {{
      type: "scatter3d",
      mode: "markers",
      x: payload.x,
      y: payload.y,
      z: payload.z,
      text: payload.hover_text,
      hovertemplate: "%{{text}}<extra></extra>",
      marker: {{
        size: payload.point_size,
        opacity: payload.point_opacity,
        color: payload.value,
        colorscale: payload.colorscale,
        cmin: payload.value_min,
        cmax: payload.value_max,
        colorbar: {{
          title: payload.value_column,
          tickformat: ".5g",
          thickness: 18,
          len: 0.78
        }},
        line: {{
          width: 0
        }}
      }}
    }};
    if (payload.center_zero) {{
      trace.marker.cmid = 0;
    }}

    const layout = {{
      title: {{
        text: `${{payload.title}}<br><sup>${{payload.subtitle}}</sup>`,
        x: 0.5
      }},
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: {{ l: 0, r: 0, t: 70, b: 0 }},
      scene: {{
        xaxis: {{
          title: payload.x_column,
          range: payload.scene_range.x,
          backgroundcolor: "#f7f8fc",
          gridcolor: "#d9deea",
          zerolinecolor: "#c7cfdf"
        }},
        yaxis: {{
          title: payload.y_column,
          range: payload.scene_range.y,
          backgroundcolor: "#f7f8fc",
          gridcolor: "#d9deea",
          zerolinecolor: "#c7cfdf"
        }},
        zaxis: {{
          title: payload.z_column,
          range: payload.scene_range.z,
          backgroundcolor: "#f7f8fc",
          gridcolor: "#d9deea",
          zerolinecolor: "#c7cfdf"
        }},
        aspectmode: "data",
        camera: {{
          eye: {{ x: 1.55, y: 1.45, z: 0.95 }}
        }}
      }}
    }};

    Plotly.newPlot("plot", [trace], layout, {{
      responsive: true,
      displaylogo: false,
      scrollZoom: true
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def _write_index_html(
    output_path: Path,
    title: str,
    input_path: Path,
    views: list[str],
    svg_paths: dict[str, Path],
    three_d_path: Path | None,
    stats: dict[str, float | int | str],
) -> None:
    stats_items = "\n".join(
        f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
        for key, value in stats.items()
    )
    gallery_items: list[str] = []
    if three_d_path is not None:
        gallery_items.append(
            (
                '<section class="card">'
                "<h2>3D View</h2>"
                f'<p><a class="button" href="{html.escape(three_d_path.name)}" target="_blank" rel="noopener noreferrer">打开交互 3D 图</a></p>'
                f'<iframe src="{html.escape(three_d_path.name)}" title="3D node view" loading="lazy"></iframe>'
                "</section>"
            )
        )
    gallery_items.extend(
        (
            '<section class="card">'
            f'<h2>{html.escape(view.upper())} Projection</h2>'
            f'<img src="{html.escape(svg_paths[view].name)}" alt="{html.escape(view)} projection" />'
            "</section>"
        )
        for view in views
    )
    gallery = "\n".join(gallery_items)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f5f9;
      --card: #ffffff;
      --text: #171a22;
      --muted: #5d6470;
      --border: #d9dde6;
      --shadow: 0 18px 48px rgba(19, 26, 40, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(180deg, #f7f8fb 0%, var(--bg) 100%);
      color: var(--text);
    }}
    main {{
      width: min(1400px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .hero {{
      padding: 24px 28px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.88);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
    }}
    .stats {{
      margin: 18px 0 0;
      padding-left: 20px;
      color: var(--text);
      line-height: 1.6;
    }}
    .grid {{
      display: grid;
      gap: 20px;
      margin-top: 22px;
    }}
    .card {{
      padding: 18px;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--card);
      box-shadow: var(--shadow);
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    .card img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: #fff;
    }}
    .card iframe {{
      display: block;
      width: 100%;
      height: min(76vh, 820px);
      min-height: 520px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: #fff;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid #b9c2d8;
      background: linear-gradient(180deg, #fbfdff 0%, #edf3ff 100%);
      color: #1f376a;
      text-decoration: none;
      font-weight: 600;
      margin-bottom: 14px;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      background: #eff2f8;
      padding: 0.15rem 0.35rem;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>输入文件：<code>{html.escape(str(input_path))}</code></p>
      <ul class="stats">
        {stats_items}
      </ul>
    </section>
    <section class="grid">
      {gallery}
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [args.x_column, args.y_column, args.z_column, args.value_column]
    frame = _read_frame(input_path, columns)
    original_rows = len(frame)
    frame = _sample_frame(frame, sample_size=args.sample_size, random_seed=args.random_seed)

    value_series = frame[args.value_column].to_numpy(dtype=np.float64)
    value_min = float(np.min(value_series) if args.color_min is None else args.color_min)
    value_max = float(np.max(value_series) if args.color_max is None else args.color_max)
    if value_min > value_max:
        raise ValueError("--color-min cannot be greater than --color-max")

    title_base = args.title or f"{input_path.stem} | {args.value_column}"
    subtitle = f"rows={len(frame)}"
    if len(frame) != original_rows:
        subtitle += f" (sampled from {original_rows})"

    view_to_axes = {
        "xy": (args.x_column, args.y_column),
        "xz": (args.x_column, args.z_column),
        "yz": (args.y_column, args.z_column),
    }

    svg_paths: dict[str, Path] = {}
    for view in args.views:
        x_column, y_column = view_to_axes[view]
        svg_path = output_dir / f"{input_path.stem}_{args.value_column}_{view}.svg"
        _write_projection_svg(
            output_path=svg_path,
            frame=frame,
            x_column=x_column,
            y_column=y_column,
            value_column=args.value_column,
            value_min=value_min,
            value_max=value_max,
            center_zero=bool(args.center_zero),
            point_radius=float(args.point_radius),
            point_opacity=float(args.point_opacity),
            title=f"{title_base} | {view.upper()}",
            subtitle=subtitle,
        )
        svg_paths[view] = svg_path

    three_d_path: Path | None = None
    if not args.skip_3d:
        three_d_path = output_dir / f"{input_path.stem}_{args.value_column}_3d.html"
        _write_3d_html(
            output_path=three_d_path,
            frame=frame,
            x_column=args.x_column,
            y_column=args.y_column,
            z_column=args.z_column,
            value_column=args.value_column,
            value_min=value_min,
            value_max=value_max,
            center_zero=bool(args.center_zero),
            point_opacity=float(args.point_opacity),
            point_radius=float(args.point_radius),
            title=title_base,
            subtitle=subtitle,
            plotly_cdn=args.plotly_cdn,
        )

    stats = {
        "rows_used": int(len(frame)),
        "rows_original": int(original_rows),
        "value_column": args.value_column,
        "value_min": _format_number(float(np.min(value_series))),
        "value_max": _format_number(float(np.max(value_series))),
        "value_mean": _format_number(float(np.mean(value_series))),
        "value_std": _format_number(float(np.std(value_series))),
        "three_d_enabled": "yes" if three_d_path is not None else "no",
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_index_html(
        output_path=output_dir / "index.html",
        title=title_base,
        input_path=input_path,
        views=list(args.views),
        svg_paths=svg_paths,
        three_d_path=three_d_path,
        stats=stats,
    )
    print(json.dumps({"output_dir": str(output_dir), **stats}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
