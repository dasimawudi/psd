#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
DOCX = ROOT / "node/doc/toBdoc/（终）应力仿真 AI 代理技术报告.docx"
ASSET_DIR = ROOT / "node/doc/toBdoc/generated_assets"
DATA_DIR = ROOT / "node/doc/toBdoc/generated_data"
REPORT = ROOT / "node/doc/remove_inner_disk_floor200模型报告_20260628.md"
HIST_SOURCE = ROOT / "node/doc/remove_inner_disk_floor200_数据分布直方图_20260629/train_fullpart_earpiece_disk_log1p_hist_100bins_floor200.png"
HIST_IMG = ASSET_DIR / "tobdoc_train_distribution_low_response_threshold.png"
OUT_DIR = ROOT / "node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200/final_eval_epoch100_20260628"
RESOLVED_CONFIG = OUT_DIR / "resolved_config_epoch100_snapshot.yaml"
PSD_CSV = OUT_DIR / "top_percent_eval_floor200/test_top_percent_relative_metrics.csv"
RMISES_CSV = OUT_DIR / "final_rmises_with_per_node_test/rmises_region_top_percent/final_rmises_region_top_percent_metrics.csv"
SPATIAL_CASE_DIR = OUT_DIR / "final_rmises_with_per_node_test/cases"

FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
LATIN_BOLD_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

PSD_REGION_MAP = [
    ("全零件", "overall"),
    ("耳片区域", "耳片区域"),
    ("圆盘区域", "圆盘区域"),
]

RMISES_REGION_MAP = [
    ("全零件", "fullpart"),
    ("耳片区域", "earpiece_region"),
    ("圆盘区域", "disk_region"),
]

TOP_GROUP_MAP = [
    ("整体", "overall", "overall"),
    ("top1%", "top1%", "top1pct"),
    ("top5%", "top5%", "top5pct"),
    ("top25%", "top25%", "top25pct"),
]

BASELINE_PSD_METRICS = {
    ("全零件", "整体"): (54.19, 81.66),
    ("全零件", "top1%"): (70.80, 20.79),
    ("全零件", "top5%"): (61.38, 27.16),
    ("全零件", "top25%"): (48.13, 40.10),
    ("耳片区域", "整体"): (43.26, 124.87),
    ("耳片区域", "top1%"): (73.14, 18.77),
    ("耳片区域", "top5%"): (70.01, 21.41),
    ("耳片区域", "top25%"): (59.73, 28.96),
    ("圆盘区域", "整体"): (57.54, 68.39),
    ("圆盘区域", "top1%"): (53.33, 27.50),
    ("圆盘区域", "top5%"): (47.35, 32.24),
    ("圆盘区域", "top25%"): (43.86, 44.42),
}

LOW_RESPONSE_STATS = [
    ("全零件", "1,057,104,666", "68.42%", "0.0038%", "31.75%"),
    ("耳片区域", "245,545,540", "41.25%", "0.0015%", "18.92%"),
    ("圆盘区域", "811,559,126", "76.64%", "0.0149%", "39.79%"),
]

DATASET_SCALE = [
    ("训练集", "82,538", "1,057,104,666", "245,545,540", "811,559,126"),
    ("验证集", "10,113", "129,825,689", "29,184,331", "100,641,358"),
    ("测试集", "10,607", "135,184,771", "31,666,289", "103,518,482"),
]

MODEL_CONFIG = [
    ("模型名", "节点级应力PSD代理模型（终版）"),
    ("训练完成轮次", "100"),
    ("输入特征", "278维"),
    ("隐藏层结构", "[256, 256, 128]"),
    ("激活函数", "SiLU"),
    ("dropout", "0.1"),
    ("层归一化", "启用"),
    ("低秩频率曲线头", "启用，rank=8，残差权重=0.15"),
    ("训练点批量", "4,000,000 points"),
    ("样本批量", "训练384 / 评估16"),
    ("学习率", "6e-5"),
    ("权重衰减", "1e-4"),
    ("低响应阈值处理", "目标应力谱密度 < 200 时按 200 参与训练与评估"),
]

FOLLOWUP_WORK_PLAN = [
    [
        "原始口径复核与误差闭环",
        "当前终版指标采用低响应阈值口径；需要补充原始目标口径下的误差形态，明确低响应背景区域的真实误差边界。",
        "在独立测试集上并行输出阈值口径与原始口径指标；按响应强度、区域和频率段拆分误差；沉淀逐节点误差样本库。",
        "原始口径对照评估报告、误差分层表、典型样本误差图谱。",
        "形成可解释的模型适用边界，并给出需要有限元复核的触发条件。",
    ],
    [
        "区域与响应分层专项优化",
        "最终 RMises 口径已较稳定，但逐频 PSD、耳片低/中响应区域以及圆盘中高响应扩展带仍存在提升空间；平坦响应区实验显示目标量级不是唯一误差来源。",
        "按低/中/高响应区间建立诊断集；对耳片、圆盘、板孔邻域做局部特征增强、分层损失权重和区域化校准。",
        "分层优化模型、区域误差对比表、低/中/高响应误差分布图。",
        "耳片逐频 PSD overall 指标和圆盘中高响应分层指标取得可量化提升，且高响应 RMises 指标不下降。",
    ],
    [
        "频率响应曲线表达升级",
        "当前模型已使用低秩频率曲线表达；已有频率曲线分解实验说明，大量节点曲线可由少量主导曲线模式近似表达。",
        "引入 SVD 频率基函数诊断；对比不同 rank 的低秩曲线头；增加峰值频率、曲线面积和模态邻域误差指标。",
        "频率基函数解释报告、rank 对比实验、逐频曲线拟合诊断图。",
        "提升逐频 PSD 曲线连续性和峰值频率定位能力，降低积分前误差向 RMises 的传递风险。",
    ],
    [
        "峰值应力校准与安全侧策略",
        "当前高响应热点命中率较高，但极端单点峰值仍可能低估；工程定型不能只依赖平均指标。",
        "建立峰值低估样本集；开展分位数校准、保守修正系数和有限元复核触发规则；输出风险等级提示。",
        "峰值校准曲线、保守修正建议、有限元复核规则清单。",
        "热点区域识别保持稳定，并对极端峰值给出偏安全侧的工程解释口径。",
    ],
    [
        "工程化工具与闭环验证",
        "当前方法已形成数据组织、特征构建、模型预测、分层评估和空间可视化链路，需要固化为可重复使用的分析流程。",
        "建设批量预测入口、自动报告模板、空间热力图、模型版本记录和有限元回灌机制；开展小批量工程试用。",
        "AI 快速分析原型工具、操作说明、试用方案清单和闭环验证记录。",
        "实现从输入工况到应力分布、热点清单和复核建议的自动化输出。",
    ],
]

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
}

for prefix, uri in NS.items():
    if prefix not in {"rel", "ct"}:
        ET.register_namespace(prefix, uri)

EMU_PER_INCH = 914400


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value:.2f} pp"


def ensure_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def copy_process_data() -> None:
    for stale in [
        "test_top_percent_relative_metrics_floor200.csv",
        "final_rmises_region_top_percent_metrics_floor200.csv",
        "low_response_floor200_stats.csv",
        "baseline_comparison_within25.csv",
        "relative_error_spatial_region_summary.csv",
    ]:
        stale_path = DATA_DIR / stale
        if stale_path.exists():
            stale_path.unlink()
    for stale_asset in [
        "tobdoc_train_distribution_floor200.png",
        "tobdoc_relative_error_earpiece_views.png",
        "tobdoc_relative_error_disk_views.png",
    ]:
        stale_asset_path = ASSET_DIR / stale_asset
        if stale_asset_path.exists():
            stale_asset_path.unlink()

    shutil.copy2(PSD_CSV, DATA_DIR / "psd_top_percent_metrics_threshold_strategy.csv")
    shutil.copy2(RMISES_CSV, DATA_DIR / "rmises_region_top_percent_metrics_threshold_strategy.csv")
    source_bins = ROOT / "node/doc/remove_inner_disk_floor200_数据分布直方图_20260629/x_axis_bin_stress_ranges_100bins.csv"
    if source_bins.exists():
        shutil.copy2(source_bins, DATA_DIR / "x_axis_bin_stress_ranges_100bins.csv")

    with (DATA_DIR / "baseline_comparison_topk_relative_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "范围",
                "前序基线模型_within25_pct",
                "前序基线模型_relative_mae_pct",
                "当前模型_within25_pct",
                "当前模型_relative_mae_pct",
                "within25_change_pp",
                "relative_mae_reduction_pp",
            ]
        )
        for row in build_baseline_comparison_rows():
            writer.writerow(
                [
                    row["scope"],
                    f"{row['baseline_within25']:.2f}",
                    f"{row['baseline_relative_mae']:.2f}",
                    f"{row['current_within25']:.2f}",
                    f"{row['current_relative_mae']:.2f}",
                    f"{row['within25_change']:.2f}",
                    f"{row['relative_mae_reduction']:.2f}",
                ]
            )

    with (DATA_DIR / "final_result_topk_metrics_compact.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["评估区域", "评估维度", "整体_within25_pct", "整体_relative_mae_pct", "top1_within25_pct", "top1_relative_mae_pct", "top5_within25_pct", "top5_relative_mae_pct", "top25_within25_pct", "top25_relative_mae_pct"])
        for row in build_final_result_metric_rows():
            writer.writerow(row)

    with (DATA_DIR / "low_response_threshold_stats.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["区域", "总点数", "<200点占比", "<200 target总量占比", "<200 log贡献"])
        writer.writerows(LOW_RESPONSE_STATS)

    with (DATA_DIR / "dataset_scale.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["划分", "样本数", "总点数", "耳片区域点数", "圆盘区域点数"])
        writer.writerows(DATASET_SCALE)

    with (DATA_DIR / "model_config_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["配置项", "值"])
        writer.writerows(MODEL_CONFIG)

    with (DATA_DIR / "followup_work_plan.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["方向", "依据", "主要工作", "阶段交付物", "预期效果"])
        writer.writerows(FOLLOWUP_WORK_PLAN)

    (DATA_DIR / "README.md").write_text(
        "\n".join(
            [
                "# toBdoc 6-9章过程数据",
                "",
                "本目录保存《（终）应力仿真 AI 代理技术报告.docx》6-9章补充内容使用的可复用过程数据。",
                "",
                "- `psd_top_percent_metrics_threshold_strategy.csv`：逐频 PSD 独立测试集高响应分层指标。",
                "- `rmises_region_top_percent_metrics_threshold_strategy.csv`：最终 RMises 分区域高响应分层指标。",
                "- `baseline_comparison_topk_relative_metrics.csv`：前序基线模型与当前模型的 within25 / 相对 MAE 分层对比数据。",
                "- `final_result_topk_metrics_compact.csv`：报告表 7-1 使用的终版模型 topk 分层指标。",
                "- `relative_error_spatial_selected_cases.csv`：三个展示样本的最终 RMises 整体 within25 / 相对 MAE 等摘要指标。",
                "- `relative_error_spatial_representative_case.csv`：三个展示样本的最终 RMises 逐节点相对误差空间数据。",
                "- `relative_error_psd_node_aggregate_selected_cases.csv`：三个展示样本的逐频 PSD 节点聚合误差摘要指标。",
                "- `relative_error_psd_node_aggregate.csv`：三个展示样本的逐频 PSD 节点聚合误差空间数据。",
                "- `low_response_threshold_stats.csv`：target<200 低响应点统计。",
                "- `dataset_scale.csv`：训练/验证/测试集规模。",
                "- `model_config_summary.csv`：终版模型配置摘要。",
                "- `followup_work_plan.csv`：第 10 章后续工作计划结构化数据。",
                "- `x_axis_bin_stress_ranges_100bins.csv`：log1p 直方图 bin 到原始应力范围映射。",
                "",
                "注意：低响应阈值处理策略相关指标均采用 target<200 统一按 200 处理的训练/评估口径，不等同于原始未阈值处理口径。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def configure_matplotlib_font() -> None:
    if FONT_PATH.exists():
        fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Droid Sans Fallback"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 220


def cjk_font(size: int | None = None, weight: str | None = None) -> fm.FontProperties:
    kwargs = {"family": ["DejaVu Sans", "Droid Sans Fallback"]}
    if size is not None:
        kwargs["size"] = size
    if weight is not None:
        kwargs["weight"] = weight
    return fm.FontProperties(**kwargs)


def apply_cjk_axis(ax, size: int = 10) -> None:
    ax.xaxis.label.set_fontproperties(cjk_font(size))
    ax.yaxis.label.set_fontproperties(cjk_font(size))
    ax.title.set_fontproperties(cjk_font(size + 2))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(cjk_font(size))


def pil_font(size: int, bold: bool = False, cjk: bool = False) -> ImageFont.ImageFont:
    if cjk and FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    path = LATIN_BOLD_FONT_PATH if bold and LATIN_BOLD_FONT_PATH.exists() else LATIN_FONT_PATH
    if path.exists():
        return ImageFont.truetype(str(path), size)
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return code > 127 and not (0xFF10 <= code <= 0xFF19)


def mixed_text_width(draw: ImageDraw.ImageDraw, text: str, size: int, bold: bool = False) -> int:
    return int(sum(draw.textlength(ch, font=pil_font(size, bold=bold, cjk=is_cjk_char(ch))) for ch in text))


def draw_mixed_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill: str, bold: bool = False) -> None:
    x, y = xy
    for ch in text:
        fnt = pil_font(size, bold=bold, cjk=is_cjk_char(ch))
        draw.text((x, y), ch, fill=fill, font=fnt)
        x += int(draw.textlength(ch, font=fnt))


def draw_center_mixed(draw: ImageDraw.ImageDraw, center_x: int, y: int, text: str, size: int, fill: str, bold: bool = False) -> None:
    draw_mixed_text(draw, (center_x - mixed_text_width(draw, text, size, bold) // 2, y), text, size, fill, bold)


def draw_right_mixed(draw: ImageDraw.ImageDraw, right_x: int, y: int, text: str, size: int, fill: str, bold: bool = False) -> None:
    draw_mixed_text(draw, (right_x - mixed_text_width(draw, text, size, bold), y), text, size, fill, bold)


def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, items: list[tuple[str, str]]) -> None:
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 46
        draw.rounded_rectangle([x, yy + 4, x + 34, yy + 28], radius=4, fill=color)
        draw_mixed_text(draw, (x + 48, yy), label, 28, "#334155")


def draw_legend_horizontal(draw: ImageDraw.ImageDraw, x: int, y: int, items: list[tuple[str, str]]) -> None:
    widths = [mixed_text_width(draw, label, 24) for label, _color in items]
    total_w = 30 + sum(34 + width + 36 for width in widths)
    draw.rounded_rectangle([x, y, x + total_w, y + 54], radius=12, fill="#ffffff", outline="#e2e8f0", width=2)
    cursor = x + 18
    for (label, color), width in zip(items, widths):
        draw.rounded_rectangle([cursor, y + 17, cursor + 28, y + 37], radius=4, fill=color)
        draw_mixed_text(draw, (cursor + 38, y + 12), label, 24, "#334155")
        cursor += 38 + width + 36


def join_deliverable_and_effect(deliverable: str, effect: str) -> str:
    return f"{deliverable.rstrip('。')}；{effect.rstrip('。')}。"


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
    ticks: list[float],
    title: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill="#ffffff", outline="#dbe3ee", width=2)
    draw_center_mixed(draw, (x0 + x1) // 2, y0 + 24, title, 30, "#0f172a", bold=True)
    plot = (x0 + 105, y0 + 100, x1 - 55, y1 - 100)
    px0, py0, px1, py1 = plot
    draw.line([px0, py0, px0, py1], fill="#334155", width=3)
    draw.line([px0, py1, px1, py1], fill="#334155", width=3)
    for tick in ticks:
        yy = py1 - int((tick - y_min) / (y_max - y_min) * (py1 - py0))
        draw.line([px0, yy, px1, yy], fill="#e5e7eb", width=2)
        draw_right_mixed(draw, px0 - 12, yy - 15, f"{tick:g}", 24, "#475569")
    draw_mixed_text(draw, (px0 - 72, py0 - 8), "%", 26, "#475569")
    return plot


def draw_bar_group(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_min: float,
    y_max: float,
) -> None:
    px0, py0, px1, py1 = plot
    n = len(labels)
    group_w = (px1 - px0) / n
    bar_w = min(72, group_w / (len(series) + 1.6))
    for i, label in enumerate(labels):
        center = px0 + group_w * (i + 0.5)
        draw_center_mixed(draw, int(center), py1 + 28, label, 26, "#0f172a")
        for j, (_, vals, color) in enumerate(series):
            x = center + (j - (len(series) - 1) / 2) * (bar_w * 1.15)
            value = vals[i]
            top = py1 - (value - y_min) / (y_max - y_min) * (py1 - py0)
            draw.rectangle([int(x - bar_w / 2), int(top), int(x + bar_w / 2), py1], fill=color)
            draw_center_mixed(draw, int(x), int(top) - 34, f"{value:.1f}%", 24, "#1f2937")


def draw_line_series(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    y_min: float,
    y_max: float,
) -> None:
    px0, py0, px1, py1 = plot
    n = len(labels)
    xs = [px0 + (px1 - px0) * i / (n - 1) for i in range(n)]
    for i, label in enumerate(labels):
        draw_center_mixed(draw, int(xs[i]), py1 + 28, label, 24, "#0f172a")
    for _, vals, color in series:
        points = []
        for x, value in zip(xs, vals):
            y = py1 - (value - y_min) / (y_max - y_min) * (py1 - py0)
            points.append((int(x), int(y)))
        draw.line(points, fill=color, width=5)
        for x, y in points:
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=color, outline="#ffffff", width=3)


def read_psd_rows() -> list[dict[str, str]]:
    with PSD_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rmises_rows() -> list[dict[str, str]]:
    with RMISES_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_metric(rows: list[dict[str, str]], region: str, group_key: str, group: str) -> dict[str, str]:
    for row in rows:
        if row["region"] == region and row[group_key] == group:
            return row
    raise KeyError((region, group_key, group))


def pct_value(row: dict[str, str], key: str) -> float:
    return float(row[key]) * 100.0


def pct_text(value: float) -> str:
    return f"{value:.2f}%"


def pct_pair(within25: float, relative_mae: float) -> str:
    return f"{within25:.2f}% / {relative_mae:.2f}%"


def build_final_result_metric_rows() -> list[list[str]]:
    psd_rows = read_psd_rows()
    rm_rows = read_rmises_rows()
    rows: list[list[str]] = []
    for display, psd_region in PSD_REGION_MAP:
        values: list[float] = []
        for group_display, psd_group, _rm_group in TOP_GROUP_MAP:
            metric = find_metric(psd_rows, psd_region, "top_group", psd_group)
            values.extend([pct_value(metric, "within25"), pct_value(metric, "relative_mae")])
        rows.append([display, "逐频 PSD"] + [pct_text(v) for v in values])
        rm_region = dict(RMISES_REGION_MAP)[display]
        values = []
        for _group_display, _psd_group, rm_group in TOP_GROUP_MAP:
            metric = find_metric(rm_rows, rm_region, "group", rm_group)
            values.extend([pct_value(metric, "within25_ratio"), pct_value(metric, "relative_mae")])
        rows.append([display, "最终 RMises"] + [pct_text(v) for v in values])
    return rows


def build_final_result_pair_rows() -> list[list[str]]:
    psd_rows = read_psd_rows()
    rm_rows = read_rmises_rows()
    rows: list[list[str]] = []
    for display, psd_region in PSD_REGION_MAP:
        values = []
        for _group_display, psd_group, _rm_group in TOP_GROUP_MAP:
            metric = find_metric(psd_rows, psd_region, "top_group", psd_group)
            values.append(pct_pair(pct_value(metric, "within25"), pct_value(metric, "relative_mae")))
        rows.append([display, "逐频 PSD"] + values)

        rm_region = dict(RMISES_REGION_MAP)[display]
        values = []
        for _group_display, _psd_group, rm_group in TOP_GROUP_MAP:
            metric = find_metric(rm_rows, rm_region, "group", rm_group)
            values.append(pct_pair(pct_value(metric, "within25_ratio"), pct_value(metric, "relative_mae")))
        rows.append([display, "最终 RMises"] + values)
    return rows


def build_baseline_comparison_rows() -> list[dict[str, float | str]]:
    psd_rows = read_psd_rows()
    output: list[dict[str, float | str]] = []
    for display, psd_region in PSD_REGION_MAP:
        for group_display, psd_group, _rm_group in TOP_GROUP_MAP:
            baseline_within25, baseline_relative_mae = BASELINE_PSD_METRICS[(display, group_display)]
            metric = find_metric(psd_rows, psd_region, "top_group", psd_group)
            current_within25 = pct_value(metric, "within25")
            current_relative_mae = pct_value(metric, "relative_mae")
            output.append(
                {
                    "scope": f"{display}-{group_display}",
                    "baseline_within25": baseline_within25,
                    "baseline_relative_mae": baseline_relative_mae,
                    "current_within25": current_within25,
                    "current_relative_mae": current_relative_mae,
                    "within25_change": current_within25 - baseline_within25,
                    "relative_mae_reduction": baseline_relative_mae - current_relative_mae,
                }
            )
    return output


def save_model_architecture(path: Path) -> None:
    def font(size: int, bold: bool = False, cjk: bool = False) -> ImageFont.ImageFont:
        if cjk and FONT_PATH.exists():
            return ImageFont.truetype(str(FONT_PATH), size)
        path = LATIN_BOLD_FONT_PATH if bold and LATIN_BOLD_FONT_PATH.exists() else LATIN_FONT_PATH
        if path.exists():
            return ImageFont.truetype(str(path), size)
        if FONT_PATH.exists():
            return ImageFont.truetype(str(FONT_PATH), size)
        return ImageFont.load_default()

    def char_font(ch: str, size: int, bold: bool = False) -> ImageFont.ImageFont:
        return font(size, bold=bold, cjk=ord(ch) > 127)

    def text_width(draw_obj: ImageDraw.ImageDraw, text: str, size: int, bold: bool = False) -> int:
        width = 0
        for ch in text:
            width += int(draw_obj.textlength(ch, font=char_font(ch, size, bold)))
        return width

    def draw_text(draw_obj: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, fill: str, bold: bool = False) -> None:
        x, y = xy
        for ch in text:
            fnt = char_font(ch, size, bold)
            draw_obj.text((x, y), ch, fill=fill, font=fnt)
            x += int(draw_obj.textlength(ch, font=fnt))

    def draw_center_text(draw_obj: ImageDraw.ImageDraw, center_x: int, y: int, text: str, size: int, fill: str, bold: bool = False) -> None:
        draw_text(draw_obj, (center_x - text_width(draw_obj, text, size, bold) // 2, y), text, size, fill, bold)

    def wrap_text(draw_obj: ImageDraw.ImageDraw, text: str, size: int, max_width: int, bold: bool = False) -> list[str]:
        lines: list[str] = []
        current = ""
        for ch in text:
            candidate = current + ch
            if text_width(draw_obj, candidate, size, bold) <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = ch
        if current:
            lines.append(current)
        return lines

    w, h = 2600, 1320
    img = Image.new("RGB", (w, h), "#f7f9fb")
    draw = ImageDraw.Draw(img)

    def rounded_box(x0, y0, x1, y1, title, body, fill, outline):
        draw.rounded_rectangle([x0, y0, x1, y1], radius=22, fill=fill, outline=outline, width=3)
        draw_text(draw, (x0 + 28, y0 + 24), title, 34, "#0f172a", bold=True)
        y = y0 + 86
        for item in body:
            for line in wrap_text(draw, item, 25, x1 - x0 - 84):
                draw_text(draw, (x0 + 40, y), line, 25, "#334155")
                y += 37
            y += 8

    def arrow(x0, y0, x1, y1, color="#64748b"):
        draw.line([x0, y0, x1, y1], fill=color, width=7)
        angle = math.atan2(y1 - y0, x1 - x0)
        head = 24
        left = (x1 - head * math.cos(angle - 0.45), y1 - head * math.sin(angle - 0.45))
        right = (x1 - head * math.cos(angle + 0.45), y1 - head * math.sin(angle + 0.45))
        draw.polygon([(x1, y1), left, right], fill=color)

    draw_text(draw, (90, 70), "节点级应力 PSD 代理模型架构", 48, "#0f172a", bold=True)
    draw_text(draw, (92, 140), "按 case × frequency × node 组织样本，融合几何、频率、模态与区域先验，输出单频应力 PSD，并积分得到最终 RMises。", 28, "#475569")

    boxes = [
        (90, 260, 455, 585, "输入数据", ["结构几何与网格节点", "PSD 载荷谱", "前10阶固有频率/模态", "有限元应力PSD标签"], "#ffffff", "#cbd5e1"),
        (555, 260, 920, 585, "样本构建", ["样本=结构方案×频率帧", "逐节点预测应力PSD", "按结构方案切分数据集", "低响应点统一置为200"], "#ffffff", "#cbd5e1"),
        (1020, 260, 1385, 585, "278维特征", ["坐标/极坐标/边界距离", "耳片/圆盘/板孔几何", "PSD值与log PSD", "模态/FRF/共振加权"], "#ffffff", "#cbd5e1"),
        (1485, 260, 1850, 585, "节点MLP主干", ["隐藏层：256/256/128", "SiLU激活", "LayerNorm", "Dropout=0.1"], "#eef6ff", "#93c5fd"),
        (1950, 260, 2315, 585, "低秩曲线头", ["rank=8", "残差权重=0.15", "增强频率曲线表达", "支撑逐频预测"], "#eefdf7", "#5eead4"),
    ]
    for box in boxes:
        rounded_box(*box)
    for x in [455, 920, 1385, 1850]:
        arrow(x + 18, 405, x + 62, 405)

    rounded_box(330, 730, 910, 1040, "输出与积分", ["输出：逐频应力PSD", "按频率方向梯形积分", "得到最终RMises", "支持全场分布和热点排序"], "#ffffff", "#cbd5e1")
    rounded_box(1010, 730, 1590, 1040, "训练目标", ["优化log1p和相对误差", "阈值处理降低背景点权重", "保留背景空间分布信号", "避免近零target主导"], "#fff7ed", "#fdba74")
    rounded_box(1690, 730, 2270, 1040, "评估诊断", ["within25：相对误差≤25%", "relative MAE / log MAE", "top1%到top25%热点分层", "逐频PSD + RMises双口径"], "#ffffff", "#cbd5e1")
    arrow(2132, 590, 620, 730)
    arrow(910, 885, 1010, 885)
    arrow(1590, 885, 1690, 885)

    draw.rounded_rectangle([90, 1140, 2510, 1245], radius=20, fill="#e8f3ef", outline="#99d2bf", width=2)
    draw_text(draw, (125, 1172), "本阶段终版模型：完成100轮训练；训练集10.57亿逐频节点点，测试集1.35亿逐频节点点；最终评估覆盖全零件、耳片区域和圆盘区域。", 31, "#0f5132")

    img.save(path)


def save_result_overview(path: Path) -> None:
    psd_rows = read_psd_rows()
    rm_rows = read_rmises_rows()
    regions = [
        ("overall", "fullpart", "全零件"),
        ("耳片区域", "earpiece_region", "耳片区域"),
        ("圆盘区域", "disk_region", "圆盘区域"),
    ]
    psd_w = [float(find_metric(psd_rows, p, "top_group", "overall")["within25"]) * 100 for p, _, _ in regions]
    psd_mae = [float(find_metric(psd_rows, p, "top_group", "overall")["relative_mae"]) * 100 for p, _, _ in regions]
    rm_w = [float(find_metric(rm_rows, r, "group", "overall")["within25_ratio"]) * 100 for _, r, _ in regions]
    rm_mae = [float(find_metric(rm_rows, r, "group", "overall")["relative_mae"]) * 100 for _, r, _ in regions]
    labels = [x[2] for x in regions]
    img = Image.new("RGB", (2600, 1120), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw_center_mixed(draw, 1300, 42, "终版模型测试集总体效果", 42, "#0f172a", bold=True)
    draw_mixed_text(draw, (110, 100), "测试集：10,607个频率样本，约1.35亿逐频节点点；RMises评估覆盖100个结构case。", 26, "#475569")
    left_plot = draw_panel_axes(draw, (80, 160, 1260, 1020), 0, 100, [0, 20, 40, 60, 80, 100], "25%误差内命中率（越高越好）")
    right_plot = draw_panel_axes(draw, (1340, 160, 2520, 1020), 0, 45, [0, 10, 20, 30, 40], "相对平均误差（越低越好）")
    draw_bar_group(draw, left_plot, labels, [("逐频PSD", psd_w, "#2563eb"), ("最终RMises", rm_w, "#10b981")], 0, 100)
    draw_bar_group(draw, right_plot, labels, [("逐频PSD", psd_mae, "#f97316"), ("最终RMises", rm_mae, "#14b8a6")], 0, 45)
    draw_legend_horizontal(draw, 190, 220, [("逐频PSD", "#2563eb"), ("最终RMises", "#10b981")])
    draw_legend_horizontal(draw, 1450, 220, [("逐频PSD", "#f97316"), ("最终RMises", "#14b8a6")])
    img.save(path)


def save_top_percent_curve(path: Path) -> None:
    psd_rows = read_psd_rows()
    rm_rows = read_rmises_rows()
    groups_psd = ["overall", "top1%", "top5%", "top10%", "top15%", "top25%"]
    groups_rm = ["overall", "top1pct", "top5pct", "top10pct", "top15pct", "top25pct"]
    labels = ["全部", "top1%", "top5%", "top10%", "top15%", "top25%"]
    regions = [
        ("overall", "fullpart", "全零件", "#2563eb"),
        ("耳片区域", "earpiece_region", "耳片区域", "#16a34a"),
        ("圆盘区域", "disk_region", "圆盘区域", "#f97316"),
    ]
    img = Image.new("RGB", (2600, 1120), "#f8fafc")
    draw = ImageDraw.Draw(img)
    draw_center_mixed(draw, 1300, 42, "高响应热点分层评估：从整体到top25%节点", 42, "#0f172a", bold=True)
    left_plot = draw_panel_axes(draw, (80, 150, 1260, 1015), 60, 100, [60, 70, 80, 90, 100], "逐频PSD维度：分层命中率")
    right_plot = draw_panel_axes(draw, (1340, 150, 2520, 1015), 78, 100, [80, 85, 90, 95, 100], "最终RMises维度：分层命中率")
    psd_series = []
    for p_region, _, name, color in regions:
        vals = [float(find_metric(psd_rows, p_region, "top_group", g)["within25"]) * 100 for g in groups_psd]
        psd_series.append((name, vals, color))
    draw_line_series(draw, left_plot, labels, psd_series, 60, 100)

    rm_series = []
    for _, r_region, name, color in regions:
        vals = [float(find_metric(rm_rows, r_region, "group", g)["within25_ratio"]) * 100 for g in groups_rm]
        rm_series.append((name, vals, color))
    draw_line_series(draw, right_plot, labels, rm_series, 78, 100)
    draw_legend_horizontal(draw, 190, 215, [(name, color) for _, _, name, color in regions])
    draw_legend_horizontal(draw, 1450, 215, [(name, color) for _, _, name, color in regions])
    img.save(path)


def save_baseline_comparison(path: Path) -> None:
    rows = build_baseline_comparison_rows()
    labels = [str(row["scope"]) for row in rows]
    baseline_within = [float(row["baseline_within25"]) for row in rows]
    current_within = [float(row["current_within25"]) for row in rows]
    baseline_rel = [float(row["baseline_relative_mae"]) for row in rows]
    current_rel = [float(row["current_relative_mae"]) for row in rows]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.0), facecolor="#f8fafc")
    fig.suptitle("前序基线模型与当前模型对比（逐频PSD维度）", fontproperties=cjk_font(15, "bold"), y=0.98)

    colors = ["#94a3b8", "#0f9f6e"]
    bar_h = 0.34
    left_bars = axes[0].barh(y + bar_h / 2, baseline_within, height=bar_h, color=colors[0], label="前序基线模型")
    right_bars = axes[0].barh(y - bar_h / 2, current_within, height=bar_h, color=colors[1], label="当前模型")
    axes[0].set_title("within25 命中率（越高越好）", fontproperties=cjk_font(11, "bold"))
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("%", fontproperties=cjk_font(9))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontproperties=cjk_font(8))
    axes[0].invert_yaxis()

    axes[1].barh(y + bar_h / 2, baseline_rel, height=bar_h, color=colors[0], label="前序基线模型")
    axes[1].barh(y - bar_h / 2, current_rel, height=bar_h, color=colors[1], label="当前模型")
    axes[1].set_title("相对 MAE（越低越好）", fontproperties=cjk_font(11, "bold"))
    axes[1].set_xlim(0, 135)
    axes[1].set_xlabel("%", fontproperties=cjk_font(9))
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()

    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(cjk_font(8))
        for spine in ax.spines.values():
            spine.set_color("#dbe3ee")
        ax.legend(loc="upper left", prop=cjk_font(8), frameon=True, facecolor="white", edgecolor="#e2e8f0")

    for bars, vals in [(left_bars, baseline_within), (right_bars, current_within)]:
        axes[0].bar_label(bars, labels=[f"{v:.1f}%" for v in vals], padding=2, fontsize=6)

    fig.text(
        0.08,
        0.025,
        "注：两组模型评估口径不同；该图用于说明低响应阈值处理策略对背景点主导问题的改善效果。",
        fontproperties=cjk_font(9),
        color="#64748b",
    )
    fig.tight_layout(rect=[0.04, 0.06, 0.99, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


SPATIAL_POINT_COLUMNS = [
    "case_name",
    "node_index",
    "target_rmises",
    "pred_rmises",
    "node_label",
    "x",
    "y",
    "z",
    "RMises_native",
    "abs_error",
    "relative_error",
    "pred_target_ratio",
    "target_rank",
    "target_rank_fraction",
]


def summarize_spatial_case(path: Path) -> dict[str, float | int | str | Path]:
    import pandas as pd

    rows = pd.read_csv(path, usecols=["case_name", "relative_error"])
    rel = rows["relative_error"].to_numpy(dtype=float)
    case_name = str(rows["case_name"].iloc[0]) if len(rows) else path.stem.replace("_final_rmises_per_node", "")
    return {
        "path": path,
        "case_name": case_name,
        "nodes": int(len(rows)),
        "within25": float((rel <= 0.25).mean()) if len(rel) else 0.0,
        "gt50": float((rel > 0.50).mean()) if len(rel) else 0.0,
        "relative_mae": float(np.mean(rel)) if len(rel) else 0.0,
        "p90": float(np.quantile(rel, 0.90)) if len(rel) else 0.0,
        "p95": float(np.quantile(rel, 0.95)) if len(rel) else 0.0,
        "max": float(np.max(rel)) if len(rel) else 0.0,
    }


def select_spatial_case_summaries(limit: int = 3) -> list[dict[str, float | int | str | Path]]:
    paths = sorted(SPATIAL_CASE_DIR.glob("*_final_rmises_per_node.csv"))
    if not paths:
        raise FileNotFoundError(f"No spatial per-node CSV files found under {SPATIAL_CASE_DIR}")
    summaries = [summarize_spatial_case(path) for path in paths]
    summaries.sort(key=lambda row: (-float(row["within25"]), float(row["relative_mae"]), float(row["gt50"]), float(row["p95"])))
    return summaries[:limit]


def load_selected_spatial_cases() -> list[dict[str, object]]:
    import pandas as pd

    selected: list[dict[str, object]] = []
    for rank, summary in enumerate(select_spatial_case_summaries(), start=1):
        rows = pd.read_csv(Path(summary["path"]), usecols=SPATIAL_POINT_COLUMNS)
        selected.append(
            {
                "rank": rank,
                "display": f"代表样本 {rank}",
                "summary": summary,
                "rows": rows,
            }
        )
    return selected


def save_spatial_process_data(cases: list[dict[str, object]]) -> None:
    import pandas as pd

    with (DATA_DIR / "relative_error_spatial_selected_cases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "展示编号",
                "原始样本名",
                "节点数",
                "overall_within25_pct",
                "overall_relative_mae_pct",
                "relative_error_gt50_pct",
                "relative_error_p90_pct",
                "relative_error_p95_pct",
                "relative_error_max_pct",
            ]
        )
        for case in cases:
            summary = case["summary"]
            writer.writerow(
                [
                    case["display"],
                    summary["case_name"],
                    str(summary["nodes"]),
                    f"{float(summary['within25']) * 100:.2f}",
                    f"{float(summary['relative_mae']) * 100:.2f}",
                    f"{float(summary['gt50']) * 100:.2f}",
                    f"{float(summary['p90']) * 100:.2f}",
                    f"{float(summary['p95']) * 100:.2f}",
                    f"{float(summary['max']) * 100:.2f}",
                ]
            )

    exports = []
    for case in cases:
        rows = case["rows"].copy()
        rows.insert(0, "展示编号", case["display"])
        rows.insert(1, "选择排序", case["rank"])
        rows["relative_error_for_color"] = rows["relative_error"].to_numpy(dtype=float)
        exports.append(rows)
    pd.concat(exports, ignore_index=True).to_csv(DATA_DIR / "relative_error_spatial_representative_case.csv", index=False)


def spatial_error_norm(vmax: float) -> matplotlib.colors.FuncNorm:
    threshold = 0.25
    high_span = max(vmax - threshold, 1e-9)

    def as_vector(values):
        arr = np.asarray(values, dtype=float)
        scalar = arr.ndim == 0
        vec = np.atleast_1d(arr)
        return arr, vec, scalar

    def color_forward(values):
        arr, vec, scalar = as_vector(values)
        below = vec <= threshold
        out = np.empty_like(vec, dtype=float)
        out[below] = 0.5 * np.clip(vec[below] / threshold, 0.0, 1.0)
        out[~below] = 0.5 + 0.5 * np.log1p(vec[~below] - threshold) / np.log1p(high_span)
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if scalar else out.reshape(arr.shape)

    def color_inverse(values):
        arr, vec, scalar = as_vector(values)
        below = vec <= 0.5
        out = np.empty_like(vec, dtype=float)
        out[below] = threshold * np.clip(vec[below] / 0.5, 0.0, 1.0)
        out[~below] = threshold + np.expm1((vec[~below] - 0.5) / 0.5 * np.log1p(high_span))
        return float(out[0]) if scalar else out.reshape(arr.shape)

    return matplotlib.colors.FuncNorm((color_forward, color_inverse), vmin=0.0, vmax=vmax)


def save_spatial_error_cases(path: Path) -> None:
    cases = load_selected_spatial_cases()
    save_spatial_process_data(cases)

    vmax = max(max(float(case["summary"]["max"]) for case in cases), 1.0)
    norm = spatial_error_norm(vmax)
    cmap = "RdYlGn_r"

    fig, axes = plt.subplots(len(cases), 2, figsize=(12.2, 10.8), facecolor="#f8fafc")
    fig.suptitle("表现较好测试样本的最终 RMises 相对误差空间分布", fontproperties=cjk_font(15, "bold"), y=0.988)
    fig.text(
        0.08,
        0.945,
        "颜色由绿色、黄色到红色表示相对误差由低到高；黄色附近对应 25% 工程误差阈值。",
        fontproperties=cjk_font(9),
        color="#475569",
    )

    for row_idx, case in enumerate(cases):
        rows = case["rows"].sort_values("relative_error")
        summary = case["summary"]
        rel = rows["relative_error"].to_numpy(dtype=float)
        x = rows["x"].to_numpy(dtype=float)
        y = rows["y"].to_numpy(dtype=float)
        z = rows["z"].to_numpy(dtype=float)
        radius = np.sqrt(np.square(x) + np.square(y))

        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]

        left_ax.scatter(x, y, c=rel, s=5.5, cmap=cmap, norm=norm, linewidths=0, alpha=0.9)
        left_ax.set_title(f"{case['display']} 俯视图（X-Y）", fontproperties=cjk_font(10, "bold"))
        left_ax.set_xlabel("X 坐标", fontproperties=cjk_font(8))
        left_ax.set_ylabel("Y 坐标", fontproperties=cjk_font(8))
        left_ax.set_aspect("equal", adjustable="box")

        right_ax.scatter(radius, z, c=rel, s=5.5, cmap=cmap, norm=norm, linewidths=0, alpha=0.9)
        right_ax.set_title(f"{case['display']} 侧视图（半径-Z）", fontproperties=cjk_font(10, "bold"))
        right_ax.set_xlabel("半径位置", fontproperties=cjk_font(8))
        right_ax.set_ylabel("Z 坐标", fontproperties=cjk_font(8))

        stats_text = "\n".join(
            [
                str(case["display"]),
                f"整体 within25：{float(summary['within25']) * 100:.1f}%",
                f"整体相对 MAE：{float(summary['relative_mae']) * 100:.1f}%",
            ]
        )
        left_ax.text(
            0.02,
            0.98,
            stats_text,
            transform=left_ax.transAxes,
            va="top",
            ha="left",
            fontproperties=cjk_font(8),
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#e2e8f0", "alpha": 0.94},
        )

        for ax in (left_ax, right_ax):
            ax.set_facecolor("#ffffff")
            ax.grid(color="#e5e7eb", linewidth=0.6)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(cjk_font(7))
            for spine in ax.spines.values():
                spine.set_color("#dbe3ee")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.08, 0.905, 0.54, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.set_title("相对误差", fontproperties=cjk_font(8), pad=4)
    ticks = [0.0, 0.25, 1.0]
    if vmax > 1.0 + 1e-12:
        ticks.append(vmax)
    ticks = [tick for tick in ticks if tick <= vmax + 1e-12]
    tick_labels = []
    for tick in ticks:
        if math.isclose(tick, 0.0):
            tick_labels.append("0%\n低")
        elif math.isclose(tick, 0.25):
            tick_labels.append("25%\n阈值")
        elif math.isclose(tick, 1.0):
            tick_labels.append("100%")
        else:
            tick_labels.append(f"{tick * 100:.0f}%\n最大")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    for label in cbar.ax.get_xticklabels():
        label.set_fontproperties(cjk_font(7))

    fig.tight_layout(rect=[0.04, 0.04, 0.99, 0.88], h_pad=1.2, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _selected_case_names_from_rmises_cases(cases: list[dict[str, object]]) -> list[str]:
    return [str(case["summary"]["case_name"]) for case in cases]


def compute_psd_node_aggregate_cases(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    import sys

    import pandas as pd
    import torch

    node_dir = ROOT / "node"
    if str(node_dir) not in sys.path:
        sys.path.insert(0, str(node_dir))

    from case7_node_mlp.data import discover_case_index, expand_case_sample_paths
    from case7_node_mlp.runtime import read_config, resolve_device
    from case7_node_mlp.scalers import StandardScaler
    from case7_node_mlp.trainer import build_model, make_loader, regression_output, _point_chunks

    ckpt_path = OUT_DIR / "best_epoch100_snapshot.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = read_config(RESOLVED_CONFIG)
    dataset_cfg = dict(config.get("dataset", {}))
    feature_cfg = dict(config.get("features", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))
    dataset_root = Path(dataset_cfg.get("root", ""))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root

    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    feature_schema = dict(checkpoint.get("feature_schema", {}))
    input_dim = int(feature_schema.get("input_dim", 0))
    device = resolve_device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = build_model(config, input_dim=input_dim, feature_schema=feature_schema)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    case_index = discover_case_index(dataset_root)
    selected_names = _selected_case_names_from_rmises_cases(cases)
    selected_dirs = [case_index[name] for name in selected_names]
    sample_paths = expand_case_sample_paths(selected_dirs, dataset_cfg)
    loader = make_loader(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        sample_batch_size=8,
        num_workers=2,
        shuffle=False,
        persistent_workers=False,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    aggregates: dict[str, dict[int, dict[str, float]]] = {
        case_name: {} for case_name in selected_names
    }
    with torch.no_grad():
        for batch in loader:
            if batch.num_points <= 0:
                continue
            pred_parts = []
            for chunk in _point_chunks(batch, point_batch_size=1_500_000, shuffle=False):
                features = batch.features[chunk].to(device, non_blocking=True)
                pred_parts.append(regression_output(model(features)).detach())
            pred_scaled = torch.cat(pred_parts, dim=0)
            y_mean = y_scaler.mean.to(device)
            y_std = y_scaler.std.to(device)
            pred_log = (pred_scaled * y_std + y_mean).reshape(-1)
            pred_raw = torch.expm1(pred_log.clamp_max(20.0)).clamp_min(0.0).cpu().numpy()
            target_raw = batch.target_raw.reshape(-1).cpu().numpy()
            sample_index = batch.sample_index.reshape(-1).cpu().numpy()
            node_indices = batch.node_indices.reshape(-1).cpu().numpy()
            rel = np.abs(pred_raw - target_raw) / np.maximum(np.abs(target_raw), 1e-12)

            for sample_idx, case_name in enumerate(batch.case_names):
                mask = sample_index == sample_idx
                if not np.any(mask):
                    continue
                case_aggs = aggregates[str(case_name)]
                for node_index, rel_value in zip(node_indices[mask], rel[mask]):
                    item = case_aggs.setdefault(int(node_index), {"points": 0.0, "rel_sum": 0.0, "within25": 0.0})
                    item["points"] += 1.0
                    item["rel_sum"] += float(rel_value)
                    item["within25"] += 1.0 if float(rel_value) <= 0.25 else 0.0

    result: list[dict[str, object]] = []
    rmises_by_case = {str(case["summary"]["case_name"]): case for case in cases}
    for rank, case_name in enumerate(selected_names, start=1):
        rmises_rows = rmises_by_case[case_name]["rows"].copy()
        case_aggs = aggregates[case_name]
        rows = []
        for row in rmises_rows.itertuples(index=False):
            node_index = int(getattr(row, "node_index"))
            item = case_aggs.get(node_index)
            if item is None or item["points"] <= 0:
                continue
            points = max(float(item["points"]), 1.0)
            rows.append(
                {
                    "case_name": case_name,
                    "node_index": node_index,
                    "x": float(getattr(row, "x")),
                    "y": float(getattr(row, "y")),
                    "z": float(getattr(row, "z")),
                    "frequency_points": int(points),
                    "psd_relative_mae": float(item["rel_sum"] / points),
                    "psd_within25": float(item["within25"] / points),
                }
            )
        rows_df = pd.DataFrame(rows)
        rel_values = rows_df["psd_relative_mae"].to_numpy(dtype=float)
        frequency_points = rows_df["frequency_points"].to_numpy(dtype=float) if len(rows_df) else np.empty(0, dtype=float)
        weights = frequency_points / max(float(frequency_points.sum()), 1.0) if len(frequency_points) else frequency_points
        summary = {
            "case_name": case_name,
            "nodes": int(len(rows_df)),
            "frequency_points": int(rows_df["frequency_points"].sum()) if len(rows_df) else 0,
            "within25": float(np.sum(rows_df["psd_within25"].to_numpy(dtype=float) * weights)) if len(rows_df) else 0.0,
            "relative_mae": float(np.sum(rel_values * weights)) if len(rel_values) else 0.0,
            "node_mean_mae_within25": float((rel_values <= 0.25).mean()) if len(rel_values) else 0.0,
            "p90": float(np.quantile(rel_values, 0.90)) if len(rel_values) else 0.0,
            "p95": float(np.quantile(rel_values, 0.95)) if len(rel_values) else 0.0,
            "max": float(np.max(rel_values)) if len(rel_values) else 0.0,
        }
        result.append(
            {
                "rank": rank,
                "display": f"代表样本 {rank}",
                "summary": summary,
                "rows": rows_df,
            }
        )
    return result


def save_psd_node_aggregate_process_data(cases: list[dict[str, object]]) -> None:
    import pandas as pd

    with (DATA_DIR / "relative_error_psd_node_aggregate_selected_cases.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "展示编号",
                "原始样本名",
                "节点数",
                "逐频点数",
                "frequency_point_within25_pct",
                "node_aggregate_relative_mae_pct",
                "node_mean_mae_within25_pct",
                "node_aggregate_p90_pct",
                "node_aggregate_p95_pct",
                "node_aggregate_max_pct",
            ]
        )
        for case in cases:
            summary = case["summary"]
            writer.writerow(
                [
                    case["display"],
                    summary["case_name"],
                    str(summary["nodes"]),
                    str(summary["frequency_points"]),
                    f"{float(summary['within25']) * 100:.2f}",
                    f"{float(summary['relative_mae']) * 100:.2f}",
                    f"{float(summary['node_mean_mae_within25']) * 100:.2f}",
                    f"{float(summary['p90']) * 100:.2f}",
                    f"{float(summary['p95']) * 100:.2f}",
                    f"{float(summary['max']) * 100:.2f}",
                ]
            )

    exports = []
    for case in cases:
        rows = case["rows"].copy()
        rows.insert(0, "展示编号", case["display"])
        rows.insert(1, "选择排序", case["rank"])
        rows["relative_error_for_color"] = rows["psd_relative_mae"].to_numpy(dtype=float)
        exports.append(rows)
    pd.concat(exports, ignore_index=True).to_csv(DATA_DIR / "relative_error_psd_node_aggregate.csv", index=False)


def save_spatial_case_grid(
    cases: list[dict[str, object]],
    *,
    value_column: str,
    title: str,
    subtitle: str,
    metric_prefix: str,
    path: Path,
) -> None:
    vmax = max(max(float(case["summary"]["max"]) for case in cases), 1.0)
    norm = spatial_error_norm(vmax)
    cmap = "RdYlGn_r"

    fig, axes = plt.subplots(len(cases), 2, figsize=(12.2, 10.8), facecolor="#f8fafc")
    fig.suptitle(title, fontproperties=cjk_font(15, "bold"), y=0.988)
    fig.text(0.08, 0.945, subtitle, fontproperties=cjk_font(9), color="#475569")

    for row_idx, case in enumerate(cases):
        rows = case["rows"].sort_values(value_column)
        summary = case["summary"]
        rel = rows[value_column].to_numpy(dtype=float)
        x = rows["x"].to_numpy(dtype=float)
        y = rows["y"].to_numpy(dtype=float)
        z = rows["z"].to_numpy(dtype=float)
        radius = np.sqrt(np.square(x) + np.square(y))

        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]
        left_ax.scatter(x, y, c=rel, s=5.5, cmap=cmap, norm=norm, linewidths=0, alpha=0.9)
        left_ax.set_title(f"{case['display']} 俯视图（X-Y）", fontproperties=cjk_font(10, "bold"))
        left_ax.set_xlabel("X 坐标", fontproperties=cjk_font(8))
        left_ax.set_ylabel("Y 坐标", fontproperties=cjk_font(8))
        left_ax.set_aspect("equal", adjustable="box")

        right_ax.scatter(radius, z, c=rel, s=5.5, cmap=cmap, norm=norm, linewidths=0, alpha=0.9)
        right_ax.set_title(f"{case['display']} 侧视图（半径-Z）", fontproperties=cjk_font(10, "bold"))
        right_ax.set_xlabel("半径位置", fontproperties=cjk_font(8))
        right_ax.set_ylabel("Z 坐标", fontproperties=cjk_font(8))

        stats_text = "\n".join(
            [
                str(case["display"]),
                f"{metric_prefix} within25：{float(summary['within25']) * 100:.1f}%",
                f"{metric_prefix}相对 MAE：{float(summary['relative_mae']) * 100:.1f}%",
            ]
        )
        left_ax.text(
            0.02,
            0.98,
            stats_text,
            transform=left_ax.transAxes,
            va="top",
            ha="left",
            fontproperties=cjk_font(8),
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#e2e8f0", "alpha": 0.94},
        )

        for ax in (left_ax, right_ax):
            ax.set_facecolor("#ffffff")
            ax.grid(color="#e5e7eb", linewidth=0.6)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(cjk_font(7))
            for spine in ax.spines.values():
                spine.set_color("#dbe3ee")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.08, 0.905, 0.54, 0.018])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.ax.set_title("相对误差", fontproperties=cjk_font(8), pad=4)
    ticks = [0.0, 0.25, 1.0]
    if vmax > 1.0 + 1e-12:
        ticks.append(vmax)
    ticks = [tick for tick in ticks if tick <= vmax + 1e-12]
    tick_labels = []
    for tick in ticks:
        if math.isclose(tick, 0.0):
            tick_labels.append("0%\n低")
        elif math.isclose(tick, 0.25):
            tick_labels.append("25%\n阈值")
        elif math.isclose(tick, 1.0):
            tick_labels.append("100%")
        else:
            tick_labels.append(f"{tick * 100:.0f}%\n最大")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    for label in cbar.ax.get_xticklabels():
        label.set_fontproperties(cjk_font(7))

    fig.tight_layout(rect=[0.04, 0.04, 0.99, 0.88], h_pad=1.2, w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_psd_node_aggregate_error_cases(path: Path) -> None:
    rmises_cases = load_selected_spatial_cases()
    psd_cases = compute_psd_node_aggregate_cases(rmises_cases)
    save_psd_node_aggregate_process_data(psd_cases)
    save_spatial_case_grid(
        psd_cases,
        value_column="psd_relative_mae",
        title="表现较好测试样本的逐频 PSD 节点聚合相对误差空间分布",
        subtitle="每个节点先跨所有频率帧计算相对 MAE，再映射到空间位置；颜色由绿色、黄色到红色表示误差由低到高。",
        metric_prefix="逐频聚合",
        path=path,
    )


def save_sanitized_histogram(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, _ = img.size
    draw.rectangle([0, 0, w, 92], fill="#ffffff")
    draw_center_mixed(draw, w // 2, 28, "训练集目标值分布：红色虚线为低响应阈值=200", 32, "#111827", bold=True)
    for y in [154, 777, 1400]:
        draw.rectangle([620, y - 92, 770, y + 46], fill="#ffffff")
        draw_mixed_text(draw, (636, y + 6), "阈值=200", 22, "#ef4444", bold=True)
    img.save(dst)


def generate_assets() -> dict[str, Path]:
    ensure_assets()
    copy_process_data()
    save_sanitized_histogram(HIST_SOURCE, HIST_IMG)
    configure_matplotlib_font()
    paths = {
        "model_arch": ASSET_DIR / "tobdoc_model_architecture.png",
        "result_overview": ASSET_DIR / "tobdoc_result_overview.png",
        "top_curve": ASSET_DIR / "tobdoc_top_percent_curve.png",
        "baseline": ASSET_DIR / "tobdoc_baseline_comparison.png",
        "spatial_cases": ASSET_DIR / "tobdoc_relative_error_fullpart_good_cases.png",
        "spatial_psd_cases": ASSET_DIR / "tobdoc_relative_error_psd_node_aggregate_cases.png",
        "hist": HIST_IMG,
    }
    save_model_architecture(paths["model_arch"])
    save_result_overview(paths["result_overview"])
    save_top_percent_curve(paths["top_curve"])
    save_baseline_comparison(paths["baseline"])
    save_spatial_error_cases(paths["spatial_cases"])
    save_psd_node_aggregate_error_cases(paths["spatial_psd_cases"])
    return paths


def w_el(tag: str, attrs: dict[str, str] | None = None, text: str | None = None) -> ET.Element:
    elem = ET.Element(qn("w", tag), attrs or {})
    if text is not None:
        elem.text = text
    return elem


def set_attr(elem: ET.Element, prefix: str, attr: str, value: str) -> None:
    elem.set(qn(prefix, attr), value)


def r_pr(size: int = 22, bold: bool = False, color: str | None = None) -> ET.Element:
    pr = w_el("rPr")
    fonts = w_el("rFonts")
    set_attr(fonts, "w", "eastAsia", "等线")
    set_attr(fonts, "w", "ascii", "Arial")
    set_attr(fonts, "w", "hAnsi", "Arial")
    set_attr(fonts, "w", "cs", "Arial")
    pr.append(fonts)
    if bold:
        b = w_el("b")
        set_attr(b, "w", "val", "true")
        pr.append(b)
    if color:
        c = w_el("color")
        set_attr(c, "w", "val", color)
        pr.append(c)
    sz = w_el("sz")
    set_attr(sz, "w", "val", str(size))
    pr.append(sz)
    return pr


def run(text: str, size: int = 22, bold: bool = False, color: str | None = None) -> ET.Element:
    r = w_el("r")
    r.append(r_pr(size=size, bold=bold, color=color))
    t = w_el("t", text=text)
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def para_pr(
    style: str | None = None,
    before: int = 120,
    after: int = 120,
    line: int = 288,
    first_line: bool = False,
    left: int = 0,
    center: bool = False,
    outline: int | None = None,
) -> ET.Element:
    pr = w_el("pPr")
    if style:
        ps = w_el("pStyle")
        set_attr(ps, "w", "val", style)
        pr.append(ps)
    sp = w_el("spacing")
    set_attr(sp, "w", "before", str(before))
    set_attr(sp, "w", "after", str(after))
    set_attr(sp, "w", "line", str(line))
    set_attr(sp, "w", "lineRule", "auto")
    pr.append(sp)
    ind = w_el("ind")
    set_attr(ind, "w", "left", str(left))
    if first_line:
        set_attr(ind, "w", "firstLine", "420")
    pr.append(ind)
    jc = w_el("jc")
    set_attr(jc, "w", "val", "center" if center else "left")
    pr.append(jc)
    if outline is not None:
        ol = w_el("outlineLvl")
        set_attr(ol, "w", "val", str(outline))
        pr.append(ol)
    return pr


def paragraph(
    parts: str | list[tuple[str, bool] | tuple[str, bool, str]],
    size: int = 22,
    first_line: bool = True,
    center: bool = False,
    before: int = 120,
    after: int = 120,
    color: str | None = None,
) -> ET.Element:
    p = w_el("p")
    p.append(para_pr(first_line=first_line, center=center, before=before, after=after))
    if isinstance(parts, str):
        p.append(run(parts, size=size, color=color))
    else:
        for item in parts:
            if len(item) == 2:
                text, bold = item  # type: ignore[misc]
                part_color = color
            else:
                text, bold, part_color = item  # type: ignore[misc]
            p.append(run(text, size=size, bold=bold, color=part_color))
    return p


def heading(text: str, level: int) -> ET.Element:
    p = w_el("p")
    if level == 2:
        p.append(para_pr(style="2", before=320, after=120, first_line=False, outline=1))
        p.append(run(text, size=32, bold=True))
    else:
        p.append(para_pr(style="3", before=260, after=100, first_line=False, outline=2))
        p.append(run(text, size=26, bold=True))
    return p


def caption(text: str) -> ET.Element:
    return paragraph(text, size=18, first_line=False, center=True, before=60, after=120, color="64748B")


def cell_para(text: str, size: int = 18, bold: bool = False, color: str | None = None, center: bool = False) -> ET.Element:
    p = w_el("p")
    p.append(para_pr(first_line=False, before=40, after=40, line=250, center=center))
    p.append(run(text, size=size, bold=bold, color=color))
    return p


def table(headers: list[str], rows: list[list[str]], widths: list[int] | None = None, small: bool = False) -> ET.Element:
    tbl = w_el("tbl")
    tbl_pr = w_el("tblPr")
    tbl_w = w_el("tblW")
    set_attr(tbl_w, "w", "w", "5000")
    set_attr(tbl_w, "w", "type", "pct")
    tbl_pr.append(tbl_w)
    borders = w_el("tblBorders")
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b = w_el(side)
        set_attr(b, "w", "val", "single")
        set_attr(b, "w", "sz", "4")
        set_attr(b, "w", "space", "0")
        set_attr(b, "w", "color", "CBD5E1")
        borders.append(b)
    tbl_pr.append(borders)
    look = w_el("tblLook")
    set_attr(look, "w", "firstRow", "1")
    set_attr(look, "w", "noHBand", "0")
    set_attr(look, "w", "noVBand", "1")
    tbl_pr.append(look)
    tbl.append(tbl_pr)

    def make_cell(text: str, header: bool = False, width: int | None = None) -> ET.Element:
        tc = w_el("tc")
        tc_pr = w_el("tcPr")
        if width:
            tc_w = w_el("tcW")
            set_attr(tc_w, "w", "w", str(width))
            set_attr(tc_w, "w", "type", "dxa")
            tc_pr.append(tc_w)
        shd = w_el("shd")
        set_attr(shd, "w", "fill", "1F4E79" if header else "FFFFFF")
        tc_pr.append(shd)
        v = w_el("vAlign")
        set_attr(v, "w", "val", "center")
        tc_pr.append(v)
        tc.append(tc_pr)
        tc.append(cell_para(text, size=16 if small else 18, bold=header, color="FFFFFF" if header else None, center=header))
        return tc

    tr = w_el("tr")
    for idx, h in enumerate(headers):
        tr.append(make_cell(h, header=True, width=widths[idx] if widths else None))
    tbl.append(tr)
    for row in rows:
        tr = w_el("tr")
        for idx, value in enumerate(row):
            tr.append(make_cell(value, width=widths[idx] if widths else None))
        tbl.append(tr)
    return tbl


def image_paragraph(rid: str, image_path: Path, width_in: float, docpr_id: int, name: str) -> ET.Element:
    with Image.open(image_path) as img:
        iw, ih = img.size
    cx = int(width_in * EMU_PER_INCH)
    cy = int(cx * ih / iw)
    p = w_el("p")
    p.append(para_pr(first_line=False, center=True, before=120, after=60))
    r = w_el("r")
    drawing = w_el("drawing")
    inline = ET.Element(qn("wp", "inline"), {"distT": "0", "distR": "0", "distB": "0", "distL": "0"})
    extent = ET.Element(qn("wp", "extent"))
    extent.set("cx", str(cx))
    extent.set("cy", str(cy))
    inline.append(extent)
    doc_pr = ET.Element(qn("wp", "docPr"))
    doc_pr.set("id", str(docpr_id))
    doc_pr.set("name", name)
    doc_pr.set("descr", "")
    inline.append(doc_pr)
    graphic = ET.Element(qn("a", "graphic"))
    graphic_data = ET.Element(qn("a", "graphicData"), {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"})
    pic = ET.Element(qn("pic", "pic"))
    nv = ET.Element(qn("pic", "nvPicPr"))
    cnv = ET.Element(qn("pic", "cNvPr"))
    cnv.set("id", str(docpr_id))
    cnv.set("name", name)
    cnv.set("descr", "")
    nv.append(cnv)
    cnv_pic = ET.Element(qn("pic", "cNvPicPr"))
    locks = ET.Element(qn("a", "picLocks"), {"noChangeAspect": "true"})
    cnv_pic.append(locks)
    nv.append(cnv_pic)
    pic.append(nv)
    blip_fill = ET.Element(qn("pic", "blipFill"))
    blip = ET.Element(qn("a", "blip"))
    blip.set(qn("r", "embed"), rid)
    blip_fill.append(blip)
    stretch = ET.Element(qn("a", "stretch"))
    stretch.append(ET.Element(qn("a", "fillRect")))
    blip_fill.append(stretch)
    pic.append(blip_fill)
    sp_pr = ET.Element(qn("pic", "spPr"))
    xfrm = ET.Element(qn("a", "xfrm"))
    xfrm.append(ET.Element(qn("a", "off"), {"x": "0", "y": "0"}))
    xfrm.append(ET.Element(qn("a", "ext"), {"cx": str(cx), "cy": str(cy)}))
    sp_pr.append(xfrm)
    geom = ET.Element(qn("a", "prstGeom"), {"prst": "rect"})
    geom.append(ET.Element(qn("a", "avLst")))
    sp_pr.append(geom)
    pic.append(sp_pr)
    graphic_data.append(pic)
    graphic.append(graphic_data)
    inline.append(graphic)
    drawing.append(inline)
    r.append(drawing)
    p.append(r)
    return p


def get_para_text(elem: ET.Element) -> str:
    return "".join(t.text or "" for t in elem.findall(".//w:t", NS)).strip()


def spacer() -> ET.Element:
    return paragraph("", first_line=False, before=40, after=40)


def build_new_section_nodes(image_rids: dict[str, str], asset_paths: dict[str, Path]) -> list[ET.Element]:
    nodes: list[ET.Element] = []
    docpr = 20

    def add_image(key: str, caption_text: str, width: float = 5.9) -> None:
        nonlocal docpr
        nodes.append(image_paragraph(image_rids[key], asset_paths[key], width, docpr, f"tobdoc_{key}"))
        docpr += 1
        nodes.append(caption(caption_text))

    nodes.append(heading("6. 当前技术方案", 2))
    nodes.append(paragraph("基于前述技术研发历程，项目已经从数据链路打通、基础预测能力建立、区域问题诊断和误差口径校准，逐步收敛到面向全场节点应力响应的 AI 代理建模方案。本阶段终版方案采用“结构方案-频率帧-网格节点”的样本组织方式，将有限元仿真结果转化为可监督学习数据；模型以节点几何、载荷频率、模态响应和区域先验为输入，预测每个节点在单一频率下的应力谱密度，并进一步汇总得到工程使用的最终应力结果。"))
    nodes.append(paragraph("当前性能最优模型采用轻量化节点级神经网络主干，并配置低秩频率曲线表达模块。该建模方式延续前文“用物理特征提升节点级代理模型”的研发路线：一方面保留全区域覆盖能力，另一方面通过耳片、圆盘和板孔等局部几何特征增强关键区域识别能力；同时引入低响应阈值处理策略，降低大量近零背景点对训练目标和相对误差评估的干扰。以下先对本章使用的核心术语作统一说明，再展开数据组织、特征体系、模型结构和评估方式。"))
    nodes.append(table(
        ["术语", "本文含义", "在本方案中的作用"],
        [
            ["有限元仿真", "用工程仿真软件把结构离散成网格节点后求解应力响应的方法", "提供本项目训练和评估使用的高保真参考数据"],
            ["结构方案 / 频率帧 / 节点", "结构方案表示一个几何与工况组合；频率帧表示一个离散频率点；节点表示网格上的空间采样点", "构成模型样本的三层组织方式"],
            ["PSD（功率谱密度）", "随机振动载荷或响应在频率域上的能量分布", "描述输入载荷和节点应力响应随频率变化的过程"],
            ["Mises 应力", "用于综合三向应力状态的等效应力指标", "作为结构强度和热点风险判断的核心应力量"],
            ["MISES 应力谱密度", "节点在某一频率下的 Mises 应力 PSD 响应", "模型直接预测的逐频目标"],
            ["RMises（均方根 Mises 应力）", "将逐频 Mises 应力谱密度沿频率方向积分后得到的最终应力汇总量", "更接近工程强度校核中使用的最终场量"],
            ["节点级代理模型", "以每个网格节点为基本预测对象的 AI 快速近似模型", "替代部分重复有限元后处理，用于快速预测全场响应"],
            ["模态 / FRF", "模态表示结构固有振动形态；FRF 表示频率响应函数", "帮助模型识别共振邻域和节点在不同振型下的活跃程度"],
            ["MLP / SiLU / LayerNorm / Dropout", "MLP 为多层感知机；SiLU 为激活函数；LayerNorm 为层归一化；Dropout 为训练正则化方法", "构成当前轻量化神经网络主干，提升训练稳定性和泛化能力"],
            ["低秩频率曲线头 / rank", "用少量基向量组合表达完整频率响应曲线；rank 表示基向量数量", "在控制模型规模的同时增强频率曲线表达能力"],
            ["RBF（径向基函数）", "按节点到关键几何位置的距离构造平滑特征", "帮助模型表达孔边、中心耦合区等局部影响范围"],
            ["log1p 与相对误差", "log1p 为 log(1+x) 变换；相对误差为 |预测值-真实值|/|真实值|", "同时处理跨数量级目标和工程上关心的比例偏差"],
            ["within25", "相对误差不超过 25% 的节点占比", "衡量预测结果落入工程可接受误差带的比例"],
            ["高响应分层（top1%/top5%/top25%）", "按真实应力从高到低选取前若干比例节点", "重点检查高风险热点，而不是只看全场平均指标"],
            ["低响应阈值处理", "目标应力谱密度小于 200 的点按 200 参与训练与评估", "降低大量近零背景点对相对误差和 log 损失的主导影响"],
        ],
        [2400, 4700, 3900],
        small=True,
    ))
    nodes.append(paragraph("具体来说，模型以有限元仿真数据为基础，按“结构方案-频率帧-节点”构建样本，预测节点在单频下的 MISES 应力谱密度，并进一步沿频率方向积分得到最终 RMises。为避免海量低响应背景点主导训练，本阶段引入低响应阈值处理策略：当目标应力谱密度小于 200 时，训练和评估中统一按 200 参与相对误差与 log 误差计算。"))

    nodes.append(heading("6.1 数据组织与样本构建", 3))
    nodes.append(paragraph("模型样本按结构方案、频率帧和节点三个层次组织：一个结构方案在一个频率帧下形成一个全场节点样本，模型逐节点预测该频率下的 MISES 应力谱密度。最终 RMises 结果由逐频 PSD 响应沿频率方向积分得到。"))
    nodes.append(table(
        ["划分", "样本数", "总点数", "耳片区域点数", "圆盘区域点数"],
        [
            ["训练集", "82,538", "1,057,104,666", "245,545,540", "811,559,126"],
            ["验证集", "10,113", "129,825,689", "29,184,331", "100,641,358"],
            ["测试集", "10,607", "135,184,771", "31,666,289", "103,518,482"],
        ],
        [1500, 1600, 2500, 2400, 2400],
    ))
    nodes.append(paragraph("训练、验证和测试按结构方案切分，避免同一结构的不同频率帧同时出现在训练集和测试集，从而保证评估更接近真实工程泛化场景。圆盘区域按当前工程诊断范围统计，避免低响应背景节点掩盖关键区域指标。"))
    add_image("hist", "图 6-1 训练集目标值分布与低响应阈值位置（红色虚线为 200）", width=5.95)

    nodes.append(heading("6.2 物理先验特征体系", 3))
    nodes.append(paragraph("输入特征共 278 维，特征设计遵循结构动力学和局部应力集中机理，重点表达“节点在哪里、当前频率处于什么模态关系、该节点在相关模态下是否活跃、是否靠近孔边/根部/中心耦合区”等信息。"))
    nodes.append(table(
        ["特征组", "主要内容", "工程含义"],
        [
            ["基础几何", "归一化坐标、极坐标、边界距离等", "描述节点在结构中的全局空间位置"],
            ["PSD 与频率上下文", "当前频率、PSD 值、log PSD、模态距离", "表达随机振动输入与共振邻域关系"],
            ["耳片局部特征", "耳孔、耳轴、耳根、连接区距离与区域标识", "刻画固定孔和根部过渡的应力集中机制"],
            ["圆盘/板孔特征", "中心径向分层、板孔距离、角向周期、径向基函数（RBF）特征", "刻画中心耦合、通孔和角向周期响应"],
            ["模态与 FRF 特征", "前 10 阶模态振型、梯度、FRF、共振加权", "描述节点在不同模态下的局部活跃程度"],
            ["区域指示变量", "耳片区域、圆盘区域等区域标识", "让模型区分不同区域的响应机制"],
        ],
        [2100, 4200, 4200],
    ))

    nodes.append(heading("6.3 代理模型结构设计", 3))
    nodes.append(paragraph("当前模型采用轻量化节点级 MLP，而不是直接使用大规模图神经网络。选择该结构的原因是：现阶段关键性能主要来自物理特征和训练目标设计，节点级模型训练快、可诊断性强，便于在大规模点数据上快速迭代。"))
    add_image("model_arch", "图 6-2 节点级应力 PSD 代理模型架构示意图", width=5.95)
    nodes.append(table(
        ["配置项", "取值"],
        [
            ["MLP 隐藏层", "[256, 256, 128]"],
            ["激活函数", "SiLU"],
            ["LayerNorm / Dropout", "启用 LayerNorm，dropout=0.1"],
            ["低秩频率曲线头", "启用，rank=8，残差权重=0.15"],
            ["训练轮数", "100"],
            ["训练点批量", "4,000,000 points"],
            ["样本批量", "训练 384 / 评估 16"],
            ["学习率与权重衰减", "lr=6e-5，weight_decay=1e-4"],
        ],
        [3200, 7800],
    ))

    nodes.append(heading("6.4 训练目标与低响应阈值处理", 3))
    nodes.append(paragraph("训练目标围绕 log1p 空间误差和相对误差设计。由于全零件和圆盘区域存在大量低响应背景点，这些点真实目标值总量极低，却在点数和 log 损失中占据很高比例，容易使模型把主要能力消耗在近零背景点上。"))
    nodes.append(table(
        ["区域", "<200 点占比", "<200 目标值总量占比", "<200 log 贡献"],
        [
            ["全零件", "68.42%", "0.0038%", "31.75%"],
            ["耳片区域", "41.25%", "0.0015%", "18.92%"],
            ["圆盘区域", "76.64%", "0.0149%", "39.79%"],
        ],
        [2300, 2500, 3000, 2500],
    ))
    nodes.append(paragraph("低响应阈值处理并不是删除低响应区域，而是将目标应力谱密度小于 200 的点统一按 200 参与训练和评估。这样既保留背景点的空间分布信号，又降低接近 0 的目标值在相对误差口径下对训练的主导作用。需要注意，本报告中的相关指标均按该阈值口径计算，不直接等同于原始未阈值处理口径。"))

    nodes.append(heading("6.5 评估指标与分层诊断体系", 3))
    nodes.append(paragraph("评估同时覆盖逐频 PSD 维度和最终 RMises 维度。逐频 PSD 指标用于检查模型是否学到频率响应过程；RMises 指标更接近工程强度校核使用口径。"))
    nodes.append(table(
        ["指标", "定义", "用途"],
        [
            ["within25", "abs(pred-target)/abs(target) ≤ 25% 的点占比", "判断工程误差带内的命中率"],
            ["相对 MAE", "逐点相对误差均值", "衡量整体偏差水平"],
            ["相对 log MAE", "log1p(MISES_psd_density) 空间 MAE", "衡量跨数量级响应的拟合稳定性"],
            ["高响应分层", "按每个结构方案-频率帧或结构方案内真实目标值/RMises 从大到小排序，取 top1%、top5%、top25% 等高响应节点集合", "重点检查高应力热点，而不是只看平均误差"],
            ["topk% within25", "在 topk% 高响应节点集合内重新计算 within25；例如 top5% within25 表示真实应力最高 5% 节点中，相对误差 ≤25% 的节点占比", "衡量高风险热点区域是否落入工程误差带"],
            ["topk% 相对 MAE", "在 topk% 高响应节点集合内重新计算逐点相对误差均值", "衡量高风险热点区域的平均偏差水平，补充 within25 只看阈值命中的不足"],
            ["pred/target", "预测均值与真实均值比值", "检查系统性高估或低估"],
        ],
        [2100, 4700, 4200],
    ))
    nodes.append(paragraph("因此，第 7 章阶段性成果表中的“top1% / top5% / top25% within25/相对 MAE”均不是全场整体指标，而是在对应高响应节点集合内单独统计得到。逐频 PSD 维度按每个结构方案-频率帧内的真实应力谱密度排序；最终 RMises 维度按每个结构方案内的最终 RMises 排序。"))

    nodes.append(heading("7. 阶段性成果", 2))
    nodes.append(paragraph("终版模型已完成 100 轮训练，并在独立测试集上重算逐频 PSD 高响应分层指标和最终 RMises 指标。测试集逐频点规模约 1.35 亿，最终 RMises 评估覆盖 100 个结构方案。"))
    add_image("result_overview", "图 7-1 终版模型独立测试集总体效果", width=5.95)
    nodes.append(table(
        ["评估区域", "评估维度", "整体\nwithin25/相对MAE", "top1%\nwithin25/相对MAE", "top5%\nwithin25/相对MAE", "top25%\nwithin25/相对MAE"],
        build_final_result_pair_rows(),
        [1350, 1700, 2500, 2500, 2500, 2500],
        small=True,
    ))
    nodes.append(paragraph("从结果看，RMises 维度整体更接近工程使用需求：全零件、耳片区域和圆盘区域的 overall within25 分别达到 90.10%、81.47% 和 92.75%。耳片逐频 PSD 的 overall 指标仍低于其他区域，但在最终 RMises 的 top1% 到 top25% 热点区间内均达到 94% 以上，说明高风险节点排序和最终危险区域捕捉能力较强。"))
    add_image("top_curve", "图 7-2 高响应热点分层 within25 指标", width=5.95)
    nodes.append(paragraph("除汇总指标外，本阶段还对整体指标表现较好的三个测试样本进行空间误差可视化，用于同时识别模型拟合稳定区域和需要重点复核的区域。由于模型同时输出逐频 PSD 和最终 RMises，空间图分为两个口径：逐频 PSD 图先对同一节点跨所有频率帧计算聚合相对误差，再映射到节点空间；最终 RMises 图则直接展示积分后的最终应力相对误差。图中颜色均表示相对误差：绿色代表相对误差接近 0、模型拟合较好的节点，黄色附近对应 25% 工程误差阈值，红色代表相对误差较高、需要重点关注的节点。"))
    add_image("spatial_psd_cases", "图 7-3 三个表现较好测试样本的逐频 PSD 节点聚合相对误差分布（俯视图与侧视图）", width=5.95)
    nodes.append(paragraph("图 7-3 反映的是频率响应过程的空间稳定性。图内标注的 within25 和相对 MAE 按逐频点统计，颜色则按每个节点跨频率帧的平均相对误差显示，因此可用于观察哪些空间位置在频率过程预测中更稳定，哪些位置在多个频率帧上反复出现较大误差。"))
    add_image("spatial_cases", "图 7-4 三个表现较好测试样本的全零件最终 RMises 相对误差分布（俯视图与侧视图）", width=5.95)
    nodes.append(paragraph("图 7-4 反映的是最终 RMises 的逐节点相对误差分布。需要注意，空间误差图反映的是全零件逐节点相对误差分布，红色区域表示局部节点相对误差较高；但区域指标尤其是 topK 指标按高响应节点集合统计。耳片区域局部低/中响应节点可能出现较高相对误差，因此在图上更醒目；同时，耳片高响应热点节点的 RMises 预测稳定性较好，因此 top1% 到 top25% 指标仍表现更优。该现象说明模型对工程关注的耳片高风险热点已有较好捕捉能力，但耳片低/中响应背景区域仍是后续校准重点。"))

    nodes.append(heading("7.1 与前序基线模型的对比", 3))
    nodes.append(paragraph("与未采用低响应阈值处理的前序基线模型相比，当前策略在逐频 PSD 维度显著改善了整体 within25，尤其是低响应背景点占比更高的圆盘区域。该对比用于说明训练目标口径调整对背景点主导问题的改善效果；由于两组模型评估口径不同，不应作为完全同口径的最终精度对比。"))
    add_image("baseline", "图 7-5 前序基线模型与当前模型对比（逐频 PSD 维度）", width=5.95)
    nodes.append(table(
        ["范围", "前序基线\nwithin25/相对MAE", "当前模型\nwithin25/相对MAE", "within25变化", "相对MAE变化"],
        [
            [
                str(row["scope"]),
                pct_pair(float(row["baseline_within25"]), float(row["baseline_relative_mae"])),
                pct_pair(float(row["current_within25"]), float(row["current_relative_mae"])),
                f"+{float(row['within25_change']):.2f} pp",
                f"{float(row['relative_mae_reduction']):+.2f} pp",
            ]
            for row in build_baseline_comparison_rows()
        ],
        [2500, 2500, 2500, 1700, 1800],
        small=True,
    ))
    nodes.append(paragraph("阶段性结论是：当前低响应阈值处理对全零件和圆盘区域的提升最明确，证明低响应背景点确实是相对误差和 log 损失口径下的重要干扰源；耳片区域高响应段相对误差已降至 15% 到 21% 区间，但整体背景段仍是后续校准和专项优化重点。"))

    nodes.append(heading("8. 工程应用价值", 2))
    nodes.append(paragraph("当前模型的价值不在于替代最终有限元校核，而在于把高成本仿真前置筛查变成快速、可批量、可诊断的工程流程。模型可在方案早期给出全场应力分布趋势、重点热点区域和需要复核的频率/区域，从而提升仿真资源使用效率。"))

    nodes.append(heading("8.1 提升方案筛查效率", 3))
    nodes.append(paragraph("传统流程中，每个几何方案和配重组合都需要完整建模、求解和后处理。代理模型训练完成后，可对候选方案进行快速全场预测，用于初筛明显低风险方案和定位高风险方案。工程团队可将高保真有限元计算集中在模型提示的风险方案、临界工况和异常区域上。"))

    nodes.append(heading("8.2 强化高风险区域识别能力", 3))
    nodes.append(paragraph("模型输出不是单一最大值，而是覆盖全场节点、频率帧和最终 RMises 的分布结果。结合高响应分层评估，可以重点查看 top1%、top5%、top25% 高响应节点。当前独立测试集中，全零件 RMises top1% within25 达到 95.67%，耳片区域 RMises top5% within25 达到 95.70%，说明模型对工程关注的危险节点具有较强识别能力。"))

    nodes.append(heading("8.3 支撑区域化诊断和设计迭代", 3))
    nodes.append(paragraph("通过分区域指标，模型能够区分耳片固定孔/根部、圆盘中心耦合区、板孔邻域等不同响应机制。对于设计人员，模型结果可以转化为更直接的设计反馈：哪些区域对几何变化敏感，哪些频段可能诱发热点，哪些节点应进入下一轮有限元复核。"))

    nodes.append(heading("8.4 沉淀可复用的 AI 辅助仿真方法", 3))
    nodes.append(paragraph("本项目形成了一套可复用方法：仿真数据组织、节点级物理特征构建、长尾目标处理、热点分层评估、RMises 汇总和区域诊断。后续可迁移到其他结构件、其他载荷谱、其他材料或其他强度校核指标，降低新项目从零搭建 AI 代理模型的成本。"))

    nodes.append(heading("8.5 支撑仿真数据与实测数据融合校准", 3))
    nodes.append(paragraph("长期看，代理模型可作为仿真和试验之间的桥梁。有限元提供大规模全场标签，少量传感器实测提供真实约束；模型可在二者之间做偏差校准、场量补全和不确定性提示，帮助工程团队形成更接近真实服役状态的全场应力判断。"))
    nodes.append(table(
        ["应用场景", "模型输出", "工程使用方式"],
        [
            ["方案快速筛查", "全场 PSD / RMises 预测、within25 可信区间参考", "批量筛掉低风险方案，减少盲目高保真仿真"],
            ["热点定位", "top1% / top5% 高响应节点和区域", "指导局部网格加密、结构加强和重点复核"],
            ["频率诊断", "单频 PSD 响应曲线与模态邻域响应", "定位可能的共振频段和主导模态"],
            ["区域设计迭代", "耳片、圆盘、板孔等分区误差和响应分布", "判断几何修改对局部风险的影响"],
            ["试验融合", "仿真场预测 + 实测校准入口", "为后续数字样机和实测修正提供基础"],
        ],
        [2600, 4200, 4200],
    ))

    nodes.append(heading("9. 当前能力边界与主要风险", 2))
    nodes.append(paragraph("当前模型已具备工程辅助分析价值，但仍处于工程验证阶段，不能表述为完全替代有限元仿真的成熟产品。建议定位为“快速筛查、热点提示和仿真优先级排序工具”，关键结论仍需有限元和试验复核。"))

    nodes.append(heading("9.1 低响应阈值评估口径限制", 3))
    nodes.append(paragraph("本阶段所有终版指标均按低响应阈值口径计算，即目标应力谱密度小于 200 的点在训练和评估时统一按 200 参与相对误差、log 误差和 RMises 积分。因此，报告结果适合判断该策略下的模型效果，但不能直接等同于原始未阈值处理口径下的泛化精度。若需要评估真实低响应背景区域误差，需要补充原始口径下的对照评估。"))

    nodes.append(heading("9.2 极端峰值仍有低估风险", 3))
    nodes.append(paragraph("模型对高响应热点的整体命中率较高，但极端单点峰值仍可能低估。RMises 区域评估中，全零件 target_rmises_max 为 612,647，pred_rmises_max 为 411,525，说明最高单点幅值仍需谨慎解释。工程使用时，应优先相信模型对风险区域和频段的提示，再用有限元对峰值幅值做确认。"))

    nodes.append(heading("9.3 不同区域成熟度不一致", 3))
    nodes.append(paragraph("圆盘区域 overall 指标较稳，耳片区域在最终 RMises 高响应段表现较好，但耳片逐频 PSD overall within25 为 68.28%，说明耳片低/中响应背景区域和局部复杂几何仍更难拟合。后续应继续做耳片区域校准、局部特征增强和分域模型组合。"))

    nodes.append(heading("9.4 数据分布决定泛化边界", 3))
    nodes.append(paragraph("模型可信范围由训练数据覆盖范围决定。若新方案超出现有几何尺寸、质量范围、材料参数、边界条件、PSD 谱形或频率范围，预测可信度会下降。对于超出分布的方案，应补充有限元样本或引入不确定性提示，而不应直接把模型输出作为最终强度结论。"))

    nodes.append(heading("9.5 建议的工程使用边界", 3))
    nodes.append(table(
        ["风险项", "当前表现", "建议控制方式"],
        [
            ["低响应背景点", "阈值处理已降低其对训练的主导，但原始口径误差仍需补充确认", "保留当前结果，同时补充原始口径对照评估"],
            ["极端峰值幅值", "热点命中率高，但最高单点可能低估", "峰值结论必须回到有限元或试验复核"],
            ["耳片低/中响应区域", "逐频 PSD overall 指标低于圆盘", "继续进行耳片专项校准和区域化模型融合"],
            ["外推方案", "超出训练分布时可信度下降", "增加相似仿真样本，并标记模型置信范围"],
            ["PSD 工况变化", "若训练谱形变化有限，载荷谱外推能力受限", "补充不同 PSD 谱形和边界条件样本"],
        ],
        [2500, 3900, 4600],
    ))
    nodes.append(paragraph("综上，当前模型适合进入工程辅助验证和内部试用阶段：用于方案初筛、热点定位、仿真优先级排序和报告诊断；但在强度定型、极端峰值判定和超出训练分布的新方案上，仍需与高保真有限元及试验数据闭环使用。"))

    nodes.append(heading("10. 后续工作计划", 2))
    nodes.append(paragraph("结合当前模型训练和评估结果，后续工作不宜只依赖继续增加训练轮次，而应围绕误差口径复核、区域分层优化、频率曲线表达、峰值安全校准和工程化闭环五个方向推进。当前模型已完成 100 轮训练，验证集指标在后期仍有小幅改善，说明模型尚未出现明显过拟合；但逐频 PSD 指标、低/中响应背景区域、同量级响应区间的区域差异以及极端峰值幅值仍是主要提升空间。"))
    nodes.append(table(
        ["方向", "当前依据", "主要工作", "交付物与验收口径"],
        [[row[0], row[1], row[2], join_deliverable_and_effect(row[3], row[4])] for row in FOLLOWUP_WORK_PLAN],
        [1900, 3300, 3300, 3600],
        small=True,
    ))

    nodes.append(heading("10.1 原始口径复核与误差闭环", 3))
    nodes.append(paragraph("本阶段终版指标采用低响应阈值口径，即低响应点按统一阈值参与训练和评估。该策略已经显著改善全零件和圆盘区域的 overall within25，但它反映的是当前工程处理口径下的模型效果。为明确模型在真实低响应背景区域的能力边界，下一步需要补充原始目标口径评估。"))
    nodes.append(paragraph("具体工作包括：在同一独立测试集上并行输出阈值口径和原始口径指标；按响应强度、区域、频率段和结构方案拆分误差；对误差较大的节点生成空间分布图和样本清单。该工作完成后，应形成“哪些结果可直接用于快速筛查、哪些结果必须回到有限元复核”的触发规则。"))

    nodes.append(heading("10.2 区域与响应分层专项优化", 3))
    nodes.append(paragraph("已有同量级响应区间诊断实验表明，即使只观察 MISES 应力谱密度在 1e6 到 1e7 的节点，不同区域的拟合难度仍存在明显差异：耳片、圆盘和全零件的 overall within25 分别为 68.99%、59.84% 和 63.30%；在 top5% 高响应节点上，耳片达到 90.99%，圆盘为 71.68%。这说明误差并不只由目标值大小决定，局部几何、模态响应和区域特征表达仍是关键因素。"))
    nodes.append(paragraph("后续应建立低响应、中响应、高响应三类诊断集，并对耳片固定孔/根部、圆盘中心耦合区、板孔邻域等区域分别评估。优化手段包括局部特征增强、分层损失权重、区域化校准和必要的分域模型组合。验收重点不应只看 overall 指标，还应确认高响应 RMises top1% 到 top25% 指标不下降。"))

    nodes.append(heading("10.3 频率响应曲线表达升级", 3))
    nodes.append(paragraph("当前模型已引入低秩频率曲线表达模块。低秩表示是指用少量基础曲线组合出复杂的频率响应曲线，从而减少模型需要直接学习的自由度。前期 SVD（奇异值分解）分析也表明，高响应节点的频率响应存在可复用的主导曲线模式，因此后续可以进一步把频率曲线结构纳入模型诊断和优化。"))
    nodes.append(paragraph("建议开展不同 rank 数量的对比实验，并增加三类曲线级指标：峰值频率误差、曲线面积误差和模态邻域误差。这样可以避免模型只在积分后的 RMises 上表现较好，却在逐频曲线峰值位置或共振邻域存在偏差。该方向的目标是提升逐频 PSD 曲线连续性和峰值频率定位能力，降低积分前误差向最终 RMises 传递的风险。"))

    nodes.append(heading("10.4 峰值应力校准与安全侧策略", 3))
    nodes.append(paragraph("当前模型对高响应热点的定位能力较强，但极端单点峰值仍可能低估。工程应用中，热点位置识别和峰值幅值判定是两个不同问题：前者可用于快速筛查和复核定位，后者关系到强度定型，必须更保守。"))
    nodes.append(paragraph("后续应建立峰值低估样本集，针对 top1% 高响应节点、结构最大 RMises 节点和空间孤立峰值分别分析误差来源。在模型输出侧，可增加分位数校准、保守修正系数和风险等级提示；在流程侧，应明确当预测峰值接近强度限值、模型置信度不足或局部误差图出现连续红区时，必须触发高保真有限元复核。"))

    nodes.append(heading("10.5 工程化工具与闭环验证", 3))
    nodes.append(paragraph("后续工程化目标是把当前研究链路固化为可重复使用的快速分析流程，而不是停留在离线模型评估。工具侧应支持批量导入结构方案和 PSD 工况，自动输出全场 RMises 预测、热点节点清单、空间误差/响应热力图、分层指标和复核建议。"))
    nodes.append(paragraph("同时需要建立有限元回灌机制：对模型提示的高风险方案、模型不确定区域和工程师重点关注区域，定期补充有限元样本并回灌训练集。通过“模型预测-有限元复核-误差归档-再训练”的闭环，逐步扩大模型可信范围，并为后续接入实测数据校准打基础。"))
    nodes.append(table(
        ["推进阶段", "建议周期", "重点任务", "阶段成果"],
        [
            ["近期", "1-2 个月", "完成原始口径复核、典型误差样本库和第 10 章计划对应的数据表固化", "形成模型适用边界和有限元复核触发规则"],
            ["中期", "2-4 个月", "开展区域分层优化、低秩频率曲线 rank 对比和峰值安全校准", "形成下一版模型和分层对比报告"],
            ["工程试用", "4-6 个月", "建设批量预测、自动报告和有限元回灌流程，并选取实际方案试用", "形成可演示的 AI 快速分析原型工具"],
        ],
        [1700, 1800, 4700, 3800],
        small=True,
    ))

    return nodes


def remove_old_tobdoc_rels(rels_root: ET.Element) -> None:
    for child in list(rels_root):
        target = child.get("Target", "")
        if target.startswith("media/tobdoc_"):
            rels_root.remove(child)


def next_rid(rels_root: ET.Element) -> int:
    max_id = 0
    for rel in rels_root:
        rid = rel.get("Id", "")
        m = re.fullmatch(r"rId(\d+)", rid)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def add_image_rels(rels_root: ET.Element, asset_paths: dict[str, Path]) -> tuple[dict[str, str], dict[str, str]]:
    rid_start = next_rid(rels_root)
    image_rids: dict[str, str] = {}
    media_names: dict[str, str] = {}
    rel_ns = NS["rel"]
    for i, key in enumerate(["hist", "model_arch", "result_overview", "top_curve", "spatial_psd_cases", "spatial_cases", "baseline"]):
        rid = f"rId{rid_start + i}"
        ext = asset_paths[key].suffix.lower()
        media_name = f"media/tobdoc_{key}{ext}"
        rel = ET.Element(f"{{{rel_ns}}}Relationship")
        rel.set("Id", rid)
        rel.set("Target", media_name)
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        rels_root.append(rel)
        image_rids[key] = rid
        media_names[key] = "word/" + media_name
    return image_rids, media_names


def ensure_png_content_type(ct_root: ET.Element) -> None:
    ct_ns = NS["ct"]
    for child in ct_root.findall(f"{{{ct_ns}}}Default"):
        if child.get("Extension") == "png":
            child.set("ContentType", "image/png")
            return
    default = ET.Element(f"{{{ct_ns}}}Default")
    default.set("Extension", "png")
    default.set("ContentType", "image/png")
    ct_root.insert(0, default)


def replace_sections(document_root: ET.Element, new_nodes: list[ET.Element]) -> None:
    body = document_root.find("w:body", NS)
    if body is None:
        raise RuntimeError("word/document.xml has no w:body")

    start = end = None
    for idx, child in enumerate(list(body)):
        text = get_para_text(child) if child.tag == qn("w", "p") else ""
        if text == "6. 当前技术方案":
            start = idx
        elif start is not None and child.tag == qn("w", "sectPr"):
            end = idx
            break
    if start is None or end is None or end <= start:
        raise RuntimeError("Could not locate section 6-10 range in target docx")

    old_children = list(body)
    for child in old_children[start:end]:
        body.remove(child)
    insert_at = start
    for node in new_nodes:
        body.insert(insert_at, node)
        insert_at += 1


def write_docx(asset_paths: dict[str, Path]) -> Path:
    backup = DOCX.with_suffix(".before_sections_6_9.bak.docx")
    shutil.copy2(DOCX, backup)

    with zipfile.ZipFile(DOCX, "r") as zin:
        doc_root = ET.fromstring(zin.read("word/document.xml"))
        rels_root = ET.fromstring(zin.read("word/_rels/document.xml.rels"))
        ct_root = ET.fromstring(zin.read("[Content_Types].xml"))

        remove_old_tobdoc_rels(rels_root)
        image_rids, media_names = add_image_rels(rels_root, asset_paths)
        ensure_png_content_type(ct_root)
        new_nodes = build_new_section_nodes(image_rids, asset_paths)
        replace_sections(doc_root, new_nodes)

        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".docx", dir=str(DOCX.parent))
        os.close(tmp_fd)
        with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in {"word/document.xml", "word/_rels/document.xml.rels", "[Content_Types].xml"}:
                    continue
                if item.filename.startswith("word/media/tobdoc_"):
                    continue
                zout.writestr(item, zin.read(item.filename))
            zout.writestr("word/document.xml", ET.tostring(doc_root, encoding="utf-8", xml_declaration=True))
            zout.writestr("word/_rels/document.xml.rels", ET.tostring(rels_root, encoding="utf-8", xml_declaration=True))
            zout.writestr("[Content_Types].xml", ET.tostring(ct_root, encoding="utf-8", xml_declaration=True))
            for key, zip_name in media_names.items():
                zout.write(asset_paths[key], zip_name)
        shutil.move(tmp_name, DOCX)
    return backup


def main() -> None:
    missing = [p for p in [DOCX, REPORT, HIST_SOURCE, PSD_CSV, RMISES_CSV, RESOLVED_CONFIG, SPATIAL_CASE_DIR] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))
    assets = generate_assets()
    backup = write_docx(assets)
    print(f"Updated: {DOCX}")
    print(f"Backup:  {backup}")
    print("Generated assets:")
    for key, path in assets.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
