from __future__ import annotations

import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT1 = ROOT / "remove_inner_disk_floor200_modeling_difference.png"
OUT2 = ROOT / "earpiece_disk_center_dual_dimension_modeling_difference.png"

FONT_CANDIDATES = [
    Path("/data-ssd/libo/p100/Iterative_Refinement/script/fonts/NotoSansSC-Regular.ttf"),
    Path("/data-ssd/libo/p100/Iterative_Refinement/script/fonts/SimHei.ttf"),
    Path("/data-ssd/libo/p100/FlagEmbedding/lee_script/dataset/train_data/SimHei.ttf"),
    Path("/data-ssd/libo/p100/wechatDataBackup/frontend/dist/assets/思源黑体-Normal.df5ff3ec.otf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            try:
                ImageFont.truetype(str(candidate), size=16)
            except OSError:
                continue
            return candidate
    raise FileNotFoundError("No usable font found")


FONT_PATH = font_path()


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=6)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue
        line = ""
        for ch in raw:
            test = line + ch
            if draw.textlength(test, font=fnt) <= max_width or not line:
                line = test
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)
    return "\n".join(lines)


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = "#d8dee8",
    width: int = 2,
    radius: int = 20,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#526070",
    width: int = 5,
) -> None:
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 22
    spread = math.radians(28)
    p1 = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    p2 = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, p1, p2], fill=color)


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    fill: str = "#17202a",
    max_width: int | None = None,
    spacing: int = 7,
) -> tuple[int, int]:
    fnt = font(size)
    rendered = wrap_text(draw, text, fnt, max_width) if max_width else text
    draw.multiline_text(xy, rendered, font=fnt, fill=fill, spacing=spacing)
    return text_size(draw, rendered, fnt)


def bullet_item(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    size: int,
    color: str,
    max_width: int,
) -> int:
    x, y = xy
    radius = max(5, size // 5)
    cy = y + size // 2 + 4
    draw.ellipse((x, cy - radius, x + 2 * radius, cy + radius), fill=color)
    _, h = draw_text(draw, (x + 34, y), text, size, "#243142", max_width=max_width)
    return h


def card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: list[str],
    accent: str,
    fill: str = "#ffffff",
    title_size: int = 34,
    body_size: int = 25,
) -> None:
    x1, y1, x2, y2 = box
    rounded_rect(draw, box, fill, outline="#d8e0ea", width=2, radius=22)
    draw.rounded_rectangle((x1, y1, x2, y1 + 76), radius=22, fill=accent)
    draw.rectangle((x1, y1 + 44, x2, y1 + 76), fill=accent)
    draw_text(draw, (x1 + 28, y1 + 17), title, title_size, "#ffffff", max_width=x2 - x1 - 56)
    y = y1 + 104
    for item in body:
        if item.startswith("•"):
            content = item[1:].strip()
            h = bullet_item(draw, (x1 + 36, y), content, body_size, accent, x2 - x1 - 104)
        else:
            _, h = draw_text(draw, (x1 + 34, y), item, body_size, "#243142", max_width=x2 - x1 - 68)
        y += h + 20


def pipeline_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    fill: str,
    outline: str,
    title_size: int = 29,
    subtitle_size: int = 22,
) -> None:
    x1, y1, x2, y2 = box
    rounded_rect(draw, box, fill, outline=outline, width=2, radius=18)
    compact = (y2 - y1) < 110
    title_y = y1 + (14 if compact else 20)
    subtitle_y = y1 + (48 if compact else 70)
    draw_text(draw, (x1 + 24, title_y), title, title_size, "#15202b", max_width=x2 - x1 - 48)
    draw_text(draw, (x1 + 24, subtitle_y), subtitle, subtitle_size, "#536173", max_width=x2 - x1 - 48, spacing=5)


def metric_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    value: str,
    color: str,
    width: int = 298,
) -> None:
    x, y = xy
    rounded_rect(draw, (x, y, x + width, y + 90), "#ffffff", outline="#d8e0ea", width=2, radius=18)
    draw_text(draw, (x + 20, y + 15), label, 21, "#596577", max_width=width - 40)
    draw_text(draw, (x + 20, y + 46), value, 30, color, max_width=width - 40)


def draw_floor200_diagram() -> None:
    w, h = 2200, 1400
    img = Image.new("RGB", (w, h), "#f5f7fb")
    draw = ImageDraw.Draw(img)

    # Subtle background grid
    for x in range(0, w, 80):
        draw.line([(x, 0), (x, h)], fill="#eef2f6", width=1)
    for y in range(0, h, 80):
        draw.line([(0, y), (w, y)], fill="#eef2f6", width=1)

    draw_text(draw, (90, 70), "报告一：remove_inner_disk_floor200 的建模差异", 54, "#101923")
    draw_text(
        draw,
        (94, 145),
        "核心变化不是网络结构，而是训练/评估 target 口径：低响应点统一 floor 到 200，削弱背景点对 log1p/相对误差的主导。",
        27,
        "#5b6675",
        max_width=1500,
    )
    draw_text(draw, (1690, 88), "2026-06-28 报告提炼", 25, "#6d7787")

    card(
        draw,
        (90, 230, 1030, 535),
        "基线：remove_inner_disk",
        [
            "• 继续使用原始 target 训练和评估",
            "• exclude_inner_disk_to_plate_hole_outer_edge = true",
            "• 同样是 node MLP 点级逐频 PSD 回归",
            "• 低响应背景点仍按原始近零值参与 log/相对误差",
        ],
        "#5e6f86",
    )
    card(
        draw,
        (1170, 230, 2110, 535),
        "本模型：floor200",
        [
            "• 基于 remove_inner_disk/best.pt 继续训练",
            "• 网络和 278 维输入特征基本保持一致",
            "• target.floor_below = 200.0",
            "• 训练/评估中 target < 200 统一按 200 处理",
        ],
        "#1f8a70",
    )

    arrow(draw, (1045, 382), (1152, 382), "#637083", 6)
    draw_text(draw, (1010, 410), "差异点集中在 target preprocessing", 22, "#5a6575", max_width=210)

    # Data distribution panel
    rounded_rect(draw, (90, 600, 1028, 1058), "#ffffff", "#d8e0ea", 2, 24)
    draw_text(draw, (124, 632), "为什么需要 floor200：低响应点数量太大", 34, "#17202a")
    rows = [
        ("全零件", "68.42%", "0.0038%", "31.75%"),
        ("耳片区域", "41.25%", "0.0015%", "18.92%"),
        ("圆盘区域", "76.64%", "0.0149%", "39.79%"),
    ]
    headers = ["区域", "<200 点占比", "<200 target 总量占比", "<200 log 贡献"]
    xs = [130, 330, 570, 820]
    y0 = 710
    for i, head in enumerate(headers):
        draw_text(draw, (xs[i], y0), head, 21, "#667386")
    draw.line([(122, y0 + 42), (994, y0 + 42)], fill="#dce3ec", width=2)
    for r, row in enumerate(rows):
        y = y0 + 70 + r * 68
        fill = "#f3faf7" if r == 2 else "#ffffff"
        draw.rounded_rectangle((118, y - 10, 996, y + 48), radius=12, fill=fill)
        for i, val in enumerate(row):
            color = "#13795b" if (r == 2 and i in (1, 3)) else "#253142"
            draw_text(draw, (xs[i], y), val, 25, color)
    draw_text(
        draw,
        (128, 970),
        "解释：这些点真实 target 总量很低，但点数和 log loss 贡献高，尤其圆盘区域最明显。",
        24,
        "#596577",
        max_width=830,
    )

    # Target transform panel
    rounded_rect(draw, (1170, 600, 2110, 1058), "#ffffff", "#d8e0ea", 2, 24)
    draw_text(draw, (1204, 632), "target 口径变化", 34, "#17202a")
    pipeline_box(draw, (1220, 720, 1478, 875), "原始 target", "MISES_psd_density\n可接近 0", "#f8fafc", "#d2dae5")
    pipeline_box(draw, (1580, 720, 1838, 875), "Floor 规则", "if target < 200\nthen target = 200", "#effaf6", "#98d6c4")
    pipeline_box(draw, (1220, 895, 1478, 1010), "保留", "空间分布信号", "#f8fafc", "#d2dae5")
    pipeline_box(draw, (1580, 895, 1838, 1010), "降低", "背景噪声权重", "#f8fafc", "#d2dae5")
    arrow(draw, (1495, 797), (1562, 797), "#1f8a70", 5)
    arrow(draw, (1350, 875), (1350, 895), "#6b7687", 4)
    arrow(draw, (1710, 875), (1710, 895), "#6b7687", 4)
    draw_text(draw, (1878, 733), "红线含义：\nlog1p(200)\n作为低值阈值", 24, "#596577", max_width=180)

    # Model pipeline
    metric_badge(draw, (110, 1082), "fullpart overall within25", "85.38%", "#1f8a70", 350)
    metric_badge(draw, (485, 1082), "耳片 PSD overall", "68.28%", "#b26a2a", 285)
    metric_badge(draw, (795, 1082), "圆盘 PSD overall", "90.61%", "#1f8a70", 285)
    metric_badge(draw, (1105, 1082), "fullpart RMises overall", "90.10%", "#1f8a70", 350)
    metric_badge(draw, (1480, 1082), "口径提醒", "非 raw target", "#b24b4b", 270)

    rounded_rect(draw, (90, 1200, 2110, 1370), "#ffffff", "#d8e0ea", 2, 24)
    draw_text(draw, (124, 1224), "共同建模主干：同一套点级 node MLP + 强特征工程", 31, "#17202a")
    boxes = [
        ((132, 1272, 440, 1352), "278 维输入特征", "几何 / PSD / 模态 / FRF / 区域 mask"),
        ((556, 1272, 864, 1352), "node MLP", "hidden [256,256,128], SiLU, LN, dropout"),
        ((980, 1272, 1288, 1352), "低秩曲线头", "rank=8, residual=0.15"),
        ((1404, 1272, 1712, 1352), "逐频预测", "case-node-f: MISES PSD"),
        ((1828, 1272, 2070, 1352), "频率积分", "case-node: RMises"),
    ]
    for box, title, sub in boxes:
        pipeline_box(draw, box, title, sub, "#f8fafc", "#d2dae5", 24, 18)
    for i in range(len(boxes) - 1):
        arrow(draw, (boxes[i][0][2] + 18, 1312), (boxes[i + 1][0][0] - 18, 1312), "#637083", 4)

    img.save(OUT1, quality=95)


def draw_dual_dimension_diagram() -> None:
    w, h = 2200, 1450
    img = Image.new("RGB", (w, h), "#f6f8fb")
    draw = ImageDraw.Draw(img)
    for x in range(0, w, 90):
        draw.line([(x, 0), (x, h)], fill="#edf1f6", width=1)
    for y in range(0, h, 90):
        draw.line([(0, y), (w, y)], fill="#edf1f6", width=1)

    draw_text(draw, (90, 64), "报告二：耳片与圆盘中心双维度评估中的建模方式差异", 50, "#101923")
    draw_text(
        draw,
        (94, 136),
        "这份报告同时比较两条建模路线和两个 node MLP specialist：点级逐频回归依赖手工物理先验，GNN 依赖网格拓扑消息传递。",
        26,
        "#5d6877",
        max_width=1650,
    )
    draw_text(draw, (1735, 84), "2026-06-13 报告提炼", 24, "#6d7787")

    # Route comparison
    card(
        draw,
        (90, 230, 1035, 610),
        "路线 A：node MLP 点级逐频回归",
        [
            "• 样本单位：(case, node, frequency)",
            "• 直接预测 MISES_psd_density",
            "• 沿频率积分得到每个节点 RMises",
            "• 不显式使用网格边连接，空间关系靠特征表达",
            "• 强化频率、模态、FRF、区域 mask、热点距离等先验",
        ],
        "#226f8f",
        title_size=32,
    )
    card(
        draw,
        (1165, 230, 2110, 610),
        "路线 B：case7_gnn 图场建模",
        [
            "• 每个 case 构造成有限元图",
            "• 节点特征：[x, y, z, bc_mask]",
            "• 边特征：[dx, dy, dz, dist]",
            "• 全局特征：case 参数、PSD、freq_top3 或频率标量",
            "• 通过 edge message passing 学习空间连续性和邻域传播",
        ],
        "#8a5a2b",
        title_size=32,
    )
    arrow(draw, (1048, 420), (1148, 420), "#6b7687", 5)
    draw_text(draw, (1010, 456), "差异：手工先验 vs 拓扑传播", 23, "#5a6575", max_width=230)

    # Node MLP flow
    rounded_rect(draw, (90, 665, 2110, 880), "#ffffff", "#d8e0ea", 2, 24)
    draw_text(draw, (124, 700), "node MLP 的统一预测链路", 32, "#17202a")
    boxes = [
        ((140, 780, 438, 845), "工况 + 节点 + 频率", "case / node / f"),
        ((548, 780, 846, 845), "强物理特征", "几何、PSD、模态、FRF、mask"),
        ((956, 780, 1254, 845), "MLP 回归", "逐点预测 PSD"),
        ((1364, 780, 1662, 845), "case-node-f", "MISES_psd_density"),
        ((1772, 780, 2060, 845), "case-node", "积分后 RMises"),
    ]
    for box, title, sub in boxes:
        pipeline_box(draw, box, title, sub, "#f8fafc", "#d2dae5", 24, 18)
    for i in range(len(boxes) - 1):
        arrow(draw, (boxes[i][0][2] + 18, 812), (boxes[i + 1][0][0] - 18, 812), "#637083", 4)

    # Specialist differences
    rounded_rect(draw, (90, 930, 1035, 1260), "#ffffff", "#d8e0ea", 2, 24)
    rounded_rect(draw, (1165, 930, 2110, 1260), "#ffffff", "#d8e0ea", 2, 24)
    draw.rounded_rectangle((90, 930, 1035, 1006), radius=24, fill="#8e3f5f")
    draw.rectangle((90, 968, 1035, 1006), fill="#8e3f5f")
    draw.rounded_rectangle((1165, 930, 2110, 1006), radius=24, fill="#2d7d6f")
    draw.rectangle((1165, 968, 2110, 1006), fill="#2d7d6f")
    draw_text(draw, (124, 949), "耳片 best：区域热点先验更强", 31, "#ffffff", max_width=880)
    draw_text(draw, (1199, 949), "圆盘中心 continue70_4m：中心曲线先验更强", 31, "#ffffff", max_width=880)

    left_items = [
        "强调耳片孔、耳轴、耳根、连接圆角等应力集中距离",
        "对高响应区域稳定：逐频 top1% within25 = 87.30%",
        "最终 RMises 高应力节点最稳：top1% within25 = 98.55%",
        "短板：整体逐频覆盖弱，overall within25 = 59.05%",
    ]
    right_items = [
        "加入中心 band/RBF、center/near mask、局部模态区域特征",
        "low-rank curve head 用于刻画中心区域频率曲线形态",
        "整体覆盖更均衡：逐频 overall within25 = 82.52%",
        "积分后整体最稳：RMises overall within25 = 96.21%",
    ]
    y = 1038
    for item in left_items:
        hh = bullet_item(draw, (130, y), item, 25, "#8e3f5f", 805)
        y += hh + 20
    y = 1038
    for item in right_items:
        hh = bullet_item(draw, (1205, y), item, 25, "#2d7d6f", 805)
        y += hh + 20

    # Dual evaluation dimensions
    rounded_rect(draw, (90, 1310, 2110, 1402), "#ffffff", "#d8e0ea", 2, 24)
    draw_text(draw, (124, 1334), "双维度评估口径：逐频 PSD 看局部频域曲线；最终 RMises 看频率积分后的工程量级。", 28, "#17202a")
    draw_text(
        draw,
        (124, 1372),
        "top 百分比均按真实 target 排序：case-frequency frame 内取逐频 topX%，case 内取最终 RMises topX%。",
        22,
        "#5d6877",
        max_width=1760,
    )

    img.save(OUT2, quality=95)


def main() -> None:
    draw_floor200_diagram()
    draw_dual_dimension_diagram()
    print(f"font={FONT_PATH}")
    print(OUT1)
    print(OUT2)


if __name__ == "__main__":
    main()
