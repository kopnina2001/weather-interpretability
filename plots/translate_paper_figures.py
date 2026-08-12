"""Create English-labelled copies of the raster figures used by ``main_en.tex``.

The numerical panels are copied pixel-for-pixel from the Russian source PNGs.  Only
white text margins (titles, axis labels, panel labels, and colour-bar labels) are
repainted.  This is intentionally separate from the scientific plotting pipeline:
the source arrays needed to regenerate the maps are not stored in this checkout.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REGULAR = "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
FONT = lambda size: ImageFont.truetype(REGULAR, size=size)
WHITE = (255, 255, 255, 255)
BLACK = (0, 0, 0, 255)


def blank(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, fill=WHITE)


def centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    spacing: int = 4,
) -> None:
    draw.multiline_text(
        xy,
        text,
        font=font,
        fill=BLACK,
        anchor="mm",
        align="center",
        spacing=spacing,
    )


def vertical_centered(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    probe = ImageDraw.Draw(image)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    layer = Image.new(
        "RGBA", (right - left + 10, bottom - top + 10), (255, 255, 255, 0)
    )
    ImageDraw.Draw(layer).text((5 - left, 5 - top), text, font=font, fill=BLACK)
    layer = layer.rotate(90, expand=True, fillcolor=(255, 255, 255, 0))
    image.alpha_composite(layer, (xy[0] - layer.width // 2, xy[1] - layer.height // 2))


def find_vertical_label_box(
    image: Image.Image, search: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Find the rightmost text column group in a colour-bar margin."""
    x0, y0, x1, y1 = search
    gray = image.convert("L").crop(search)
    active = [
        x
        for x in range(gray.width)
        if gray.crop((x, 0, x + 1, gray.height)).getextrema()[0] < 245
    ]
    if not active:
        raise ValueError(f"no vertical label found in {search}")
    groups: list[list[int]] = []
    for x in active:
        if not groups or x > groups[-1][-1] + 1:
            groups.append([x])
        else:
            groups[-1].append(x)
    group = groups[-1]
    return (x0 + group[0] - 3, y0, x0 + group[-1] + 4, y1)


DOSE_CASES = {
    ("aurora", 6): ("Aurora", "0.488"),
    ("aurora", 24): ("Aurora", "0.139"),
    ("pangu", 6): ("Pangu-Weather", "0.788"),
    ("pangu", 24): ("Pangu-Weather", "0.397"),
}

DOSE_TICKS = {
    ("aurora", 6): (
        ((253, "0.5"), (339, "0.4"), (424, "0.3"), (510, "0.2"), (595, "0.1"), (681, "0.0")),
        ((839, "0.275"), (894, "0.250"), (949, "0.225"), (1004, "0.200"),
         (1059, "0.175"), (1114, "0.150"), (1170, "0.125"), (1225, "0.100")),
    ),
    ("aurora", 24): (
        ((278, "0.20"), (335, "0.18"), (393, "0.16"), (450, "0.14"),
         (508, "0.12"), (565, "0.10"), (623, "0.08"), (681, "0.06")),
        ((839, "0.30"), (909, "0.28"), (978, "0.26"), (1048, "0.24"),
         (1117, "0.22"), (1186, "0.20"), (1256, "0.18")),
    ),
    ("pangu", 6): (
        ((254, "0.8"), (308, "0.7"), (362, "0.6"), (416, "0.5"), (470, "0.4"),
         (524, "0.3"), (578, "0.2"), (633, "0.1"), (687, "0.0")),
        ((810, "0.40"), (884, "0.35"), (959, "0.30"), (1034, "0.25"),
         (1108, "0.20"), (1183, "0.15"), (1258, "0.10")),
    ),
    ("pangu", 24): (
        ((228, "0.5"), (330, "0.4"), (432, "0.3"), (534, "0.2"), (636, "0.1")),
        ((820, "0.45"), (906, "0.40"), (992, "0.35"), (1078, "0.30"),
         (1164, "0.25"), (1250, "0.20")),
    ),
}


def translate_dose(model: str, lead: int) -> Path:
    model_name, patched_slope = DOSE_CASES[(model, lead)]
    src = ROOT / "figures" / "dose_response" / (
        f"dose_panels_anom_Z1000_lead{lead}_{model}_n48.png"
    )
    dst = src.with_name(f"{src.stem}_en.png")
    image = Image.open(src).convert("RGBA")
    if image.size != (2325, 1350):
        raise ValueError(f"unexpected dose figure size for {src}: {image.size}")
    draw = ImageDraw.Draw(image)

    blank(draw, (0, 8, 2324, 103))
    centered(
        draw,
        (1162, 35),
        f"{model_name} +{lead} h: response of all 19 fields to gradual Z1000 climatology replacement (48 dates)",
        FONT(25),
    )
    centered(
        draw,
        (1162, 70),
        "Markers show measured values; dashed lines are least-squares fits with slope k. "
        "RMSE is normalized by anomaly SD; y=1 is climatology.",
        FONT(21),
    )

    panel_x = (405, 1161, 1909)
    panel_titles = (
        "Geopotential Z",
        "Specific humidity Q",
        "Temperature T",
        "Zonal wind U",
        "Meridional wind V",
        "Surface fields",
    )
    blank(draw, (80, 188, 2310, 218))
    blank(draw, (80, 744, 2310, 780))
    for x, title in zip(panel_x, panel_titles[:3]):
        centered(draw, (x, 202), title, FONT(22))
    for x, title in zip(panel_x, panel_titles[3:]):
        centered(draw, (x, 765), title, FONT(22))

    # Replace the only Russian legend fragment while preserving markers and values.
    blank(draw, (164, 232, 495, 260))
    draw.text(
        (168, 235),
        f"Z1000 (patched): k = {patched_slope}",
        font=FONT(16),
        fill=BLACK,
    )

    blank(draw, (0, 220, 113, 690))
    blank(draw, (0, 782, 113, 1268))
    vertical_centered(image, (35, 455), "RMSE / σ_w(anomaly)", FONT(20))
    vertical_centered(image, (35, 1025), "RMSE / σ_w(anomaly)", FONT(20))
    draw = ImageDraw.Draw(image)
    for y, label in DOSE_TICKS[(model, lead)][0]:
        draw.line((107, y, 114, y), fill=BLACK, width=1)
        draw.text((103, y), label, font=FONT(18), fill=BLACK, anchor="rm")
    for y, label in DOSE_TICKS[(model, lead)][1]:
        draw.line((107, y, 114, y), fill=BLACK, width=1)
        draw.text((103, y), label, font=FONT(18), fill=BLACK, anchor="rm")

    blank(draw, (95, 1301, 2315, 1349))
    for x in panel_x:
        centered(draw, (x, 1324), "alpha: climatology fraction in input Z1000", FONT(19))

    image.save(dst, optimize=True)
    return dst


BIAS_CASES = {
    ("aurora", 6): {
        "name": "Aurora",
        "values": ((99, 2.92), (100, 6.02), (98, 2.69), (98, 2.73)),
    },
    ("aurora", 24): {
        "name": "Aurora",
        "values": ((97, 1.61), (99, 2.01), (94, 1.50), (93, 1.49)),
    },
    ("pangu", 6): {
        "name": "Pangu-Weather",
        "values": ((100, 2.92), (100, 3.35), (99, 2.84), (99, 2.83)),
    },
    ("pangu", 24): {
        "name": "Pangu-Weather",
        "values": ((99, 2.44), (99, 3.06), (98, 1.86), (98, 1.84)),
    },
}


def translate_bias(model: str, lead: int) -> Path:
    case = BIAS_CASES[(model, lead)]
    prefix = "composite_bias" if model == "aurora" else "composite_bias_pangu"
    src = ROOT / "figures" / "bias_maps" / (
        f"{prefix}_Z1000_lead{lead}_alpha1_n48.png"
    )
    dst = src.with_name(f"{src.stem}_en.png")
    image = Image.open(src).convert("RGBA")
    if image.size != (2380, 1470):
        raise ValueError(f"unexpected bias figure size for {src}: {image.size}")
    draw = ImageDraw.Draw(image)

    blank(draw, (0, 16, 2379, 112))
    centered(
        draw,
        (1190, 47),
        f"{case['name']}, 48-date composite, +{lead} h: Z1000 replaced by climatology (alpha=1)",
        FONT(25),
    )
    centered(
        draw,
        (1190, 82),
        "Grid-cell temporal RMSE minus the baseline RMSE. Red indicates added error; "
        "each field uses its own fixed dose scale.",
        FONT(21),
    )

    fields = ("Q1000", "T1000", "U1000", "V1000")
    centers = ((505, 199), (1690, 199), (505, 879), (1690, 879))
    blank(draw, (100, 170, 950, 224))
    blank(draw, (1250, 170, 2150, 224))
    blank(draw, (100, 845, 950, 905))
    blank(draw, (1250, 845, 2150, 905))
    for field, (pct, ratio), xy in zip(fields, case["values"], centers):
        centered(
            draw,
            xy,
            f"{field}: RMSE(alpha=1) - RMSE(alpha=0)\nworse at {pct}% of cells; mean RMSE x{ratio:.2f}",
            FONT(18),
            spacing=1,
        )

    # Tick-label widths move the source labels between panels.  Detect each
    # source label before repainting it so that no Russian fragments remain and
    # no tick value is erased.  The translation itself has no opaque backing.
    label_specs = (
        ((980, 330, 1180, 570), (1120, 450), "Delta RMSE, kg/kg"),
        ((2160, 330, 2370, 570), (2260, 450), "Delta RMSE, K"),
        ((980, 1000, 1180, 1240), (1120, 1120), "Delta RMSE, m/s"),
        ((2160, 1000, 2370, 1240), (2260, 1120), "Delta RMSE, m/s"),
    )
    labels = [
        (find_vertical_label_box(image, search), xy, label)
        for search, xy, label in label_specs
    ]
    for box, xy, label in labels:
        blank(ImageDraw.Draw(image), box)
        vertical_centered(image, xy, label, FONT(16))

    image.save(dst, optimize=True)
    return dst


T1000_CROP = (1180, 225, 2340, 700)


def crop_t1000_panel(model: str, lead: int, language: str) -> Path:
    """Extract the complete T1000 map and colour bar for the main-text figure."""
    if language not in {"ru", "en"}:
        raise ValueError(f"unsupported language: {language}")
    prefix = "composite_bias" if model == "aurora" else "composite_bias_pangu"
    stem = f"{prefix}_Z1000_lead{lead}_alpha1_n48"
    suffix = "_en" if language == "en" else ""
    src = ROOT / "figures" / "bias_maps" / f"{stem}{suffix}.png"
    dst = src.with_name(f"{stem}_t1000_{language}.png")
    image = Image.open(src).convert("RGBA")
    if image.size != (2380, 1470):
        raise ValueError(f"unexpected bias figure size for {src}: {image.size}")
    image.crop(T1000_CROP).save(dst, optimize=True)
    return dst


def translate_acc_strip() -> Path:
    src = ROOT / "figures" / "patching_matrices" / "acc_strip_Z_both_n48.png"
    dst = src.with_name(f"{src.stem}_en.png")
    image = Image.open(src).convert("RGBA")
    if image.size != (2240, 1632):
        raise ValueError(f"unexpected ACC strip size for {src}: {image.size}")
    draw = ImageDraw.Draw(image)

    blank(draw, (0, 10, 2239, 95))
    centered(
        draw,
        (1120, 37),
        "Anomaly-correlation loss after replacing geopotential by climatology: Aurora and Pangu-Weather",
        FONT(27),
    )
    centered(
        draw,
        (1120, 71),
        "Each panel keeps the scale of its full 19 x 19 matrix; compare cell values, not colour intensity.",
        FONT(22),
    )

    titles = (
        (1120, 210, "Aurora  +6 h   (n=48 dates)"),
        (1120, 562, "Aurora  +24 h   (n=48 dates)"),
        (1120, 910, "Pangu-Weather  +6 h   (n=48 dates)"),
        (1120, 1265, "Pangu-Weather  +24 h   (n=48 dates)"),
    )
    for x, y, text in titles:
        blank(draw, (650, y - 22, 1590, y + 21))
        centered(draw, (x, y), text, FONT(23))

    blank(draw, (0, 180, 49, 1525))
    for y in (338, 690, 1042, 1391):
        vertical_centered(image, (25, y), "patched field", FONT(19))
    draw = ImageDraw.Draw(image)
    blank(draw, (1970, 150, 2045, 1525))
    for y in (338, 690, 1042, 1391):
        vertical_centered(image, (2007, y), "Delta ACC_w (blue = worse)", FONT(17))

    draw = ImageDraw.Draw(image)
    blank(draw, (730, 1581, 1510, 1631))
    centered(draw, (1120, 1605), "forecast field", FONT(21))

    image.save(dst, optimize=True)
    return dst


if __name__ == "__main__":
    outputs = []
    for model in ("aurora", "pangu"):
        for lead in (6, 24):
            outputs.append(translate_dose(model, lead))
            outputs.append(translate_bias(model, lead))
            outputs.append(crop_t1000_panel(model, lead, "ru"))
            outputs.append(crop_t1000_panel(model, lead, "en"))
    outputs.append(translate_acc_strip())
    for output in outputs:
        print(output.relative_to(ROOT))
