#import "@preview/cetz:0.5.2"

#let lang = sys.inputs.at("lang", default: "en")
#let tr(en, ru) = if lang == "ru" { ru } else { en }

#let ink = rgb("#172231")
#let muted = rgb("#4b6073")
#let line-grey = rgb("#536878")
#let blue = rgb("#3f789d")
#let blue-dark = rgb("#285c7d")
#let blue-pale = rgb("#eaf2f7")
#let green = rgb("#3f8c48")
#let green-pale = rgb("#edf6e9")
#let orange = rgb("#c56819")
#let orange-pale = rgb("#fff0d8")
#let purple = rgb("#8a4d97")
#let purple-dark = rgb("#6d3478")
#let purple-pale = rgb("#f2e8f6")
#let panel-stroke = rgb("#b9cad5")
#let panel-fill = rgb("#f8fafb")

#set page(width: 15cm, height: 9cm, margin: 0pt, fill: white)
#set text(font: "Linux Biolinum", size: 8.4pt, fill: ink)
#show math.equation: set text(font: "New Computer Modern Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Fix the canvas bounds so text can never shift the drawing origin.
  rect((0, 0), (15, 9), fill: none, stroke: none)

  let panel(p1, p2, color, number, title) = {
    rect(p1, p2, radius: 0.18, fill: panel-fill, stroke: panel-stroke + 0.7pt)
    line(
      (p1.at(0) + 0.18, p2.at(1) - 0.78),
      (p2.at(0) - 0.18, p2.at(1) - 0.78),
      stroke: color + 0.55pt,
    )
    circle(
      (p1.at(0) + 0.34, p2.at(1) - 0.41),
      radius: 0.20,
      fill: color,
      stroke: none,
    )
    content(
      (p1.at(0) + 0.34, p2.at(1) - 0.41),
      text(size: 7.6pt, weight: "bold", fill: white, str(number)),
      anchor: "center",
    )
    content(
      (p1.at(0) + 0.68, p2.at(1) - 0.41),
      text(size: 9.1pt, weight: "bold", fill: ink, title),
      anchor: "west",
    )
  }

  let node(p1, p2, fill-color, stroke-color, body, radius: 0.13) = {
    rect(p1, p2, radius: radius, fill: fill-color, stroke: stroke-color + 0.75pt)
    content(
      ((p1.at(0) + p2.at(0)) / 2, (p1.at(1) + p2.at(1)) / 2),
      align(
        center,
        block(width: (p2.at(0) - p1.at(0) - 0.18) * 1cm, body),
      ),
      anchor: "center",
    )
  }

  let arrow(..points, color: line-grey, width: 0.72pt) = {
    line(..points.pos(), stroke: color + width, mark: (end: ">"))
  }

  let route(..points, color: line-grey, width: 0.72pt) = {
    line(..points.pos(), stroke: color + width)
  }

  // Wide two-row layout. The numbered path runs clockwise and then back to the left.
  panel((0.10, 4.65), (7.35, 8.90), blue, 1,
    tr([Controlled intervention], [Контролируемая замена]))
  panel((7.65, 4.65), (14.90, 8.90), green, 2,
    tr([Matched rollouts], [Парные прогоны]))
  panel((7.65, 0.10), (14.90, 4.35), blue, 3,
    tr([Paired diagnostics], [Парная диагностика]))
  panel((0.10, 0.10), (7.35, 4.35), purple, 4,
    tr([Evidence products], [Результаты аудита]))

  // 1. Controlled intervention: the date and 18 fields are unchanged.
  content(
    (0.34, 7.83),
    text(size: 7.1pt, weight: "bold", fill: blue-dark,
      tr([Reference branch], [Базовая ветвь])),
    anchor: "west",
  )
  node(
    (0.34, 6.66), (2.12, 7.46),
    white, blue-dark,
    stack(
      dir: ttb, spacing: 0.5pt,
      text(size: 8.0pt, weight: "bold", tr([ERA5 state], [ERA5])),
      text(size: 11.0pt)[$x(t)$],
    ),
  )
  node(
    (4.72, 6.66), (7.05, 7.46),
    white, blue-dark,
    stack(
      dir: ttb, spacing: 0.5pt,
      text(size: 7.7pt, tr([unchanged input], [вход без замены])),
      text(size: 10.8pt)[$x^((0)) = x$],
    ),
  )
  arrow((2.12, 7.06), (4.72, 7.06), color: line-grey)

  content(
    (0.34, 6.30),
    text(size: 7.1pt, weight: "bold", fill: orange,
      tr([Patched branch], [Ветвь с заменой])),
    anchor: "west",
  )
  node(
    (0.34, 4.99), (2.72, 5.97),
    white, blue-dark,
    stack(
      dir: ttb, spacing: 0.4pt,
      text(size: 7.6pt, weight: "bold", tr([WB2 climatology], [Климатология WB2])),
      text(size: 9.3pt)[$c_v(d,h)$],
      text(size: 6.9pt, fill: muted, tr([same season + UTC], [тот же сезон + UTC])),
    ),
  )
  node(
    (3.20, 4.86), (7.05, 6.12),
    orange-pale, orange,
    stack(
      dir: ttb, spacing: 2.2pt,
      text(size: 10.4pt)[$x_v^((alpha)) = (1-alpha)x_v + alpha c_v$],
      text(size: 7.0pt, fill: muted)[$alpha = 0, .2, ..., 1$],
      text(size: 6.6pt, fill: muted,
        tr([date and other 18 fields fixed], [дата и остальные 18 полей фиксированы])),
    ),
  )
  arrow((2.72, 5.48), (3.20, 5.48), color: orange)
  arrow(
    (1.23, 6.66), (1.23, 6.42), (5.08, 6.42), (5.08, 6.12),
    color: orange,
  )

  // 2. Both states pass through exactly the same frozen model.
  content(
    (7.92, 7.83),
    text(size: 7.1pt, weight: "bold", fill: green,
      tr([Baseline], [Базовый прогноз])),
    anchor: "west",
  )
  node(
    (7.94, 6.58), (10.22, 7.50),
    green-pale, green,
    stack(
      dir: ttb, spacing: 3.0pt,
      text(size: 10.5pt)[$f_theta$],
      text(size: 7.1pt, fill: muted, tr([frozen weights], [фиксированные веса])),
    ),
  )
  node(
    (10.92, 6.58), (13.24, 7.50),
    purple-pale, purple-dark,
    stack(
      dir: ttb, spacing: 2.6pt,
      text(size: 10.0pt)[$hat(y)_u^((0))$],
      text(size: 7.4pt, fill: muted)[$+6 h, +24 h$],
    ),
  )
  arrow((7.05, 7.06), (7.94, 7.06), color: line-grey)
  arrow((10.22, 7.04), (10.92, 7.04), color: line-grey)

  content(
    (7.92, 6.30),
    text(size: 7.1pt, weight: "bold", fill: green,
      tr([Patched], [С заменой])),
    anchor: "west",
  )
  node(
    (7.94, 4.95), (10.22, 5.87),
    green-pale, green,
    stack(
      dir: ttb, spacing: 3.0pt,
      text(size: 10.5pt)[$f_theta$],
      text(size: 7.1pt, fill: muted, tr([same weights], [те же веса])),
    ),
  )
  node(
    (10.92, 4.95), (13.24, 5.87),
    purple-pale, purple-dark,
    stack(
      dir: ttb, spacing: 2.6pt,
      text(size: 9.7pt)[$hat(y)_u^((v, alpha))$],
      text(size: 7.4pt, fill: muted)[$+6 h, +24 h$],
    ),
  )
  arrow((7.05, 5.48), (7.94, 5.48), color: orange)
  arrow((10.22, 5.41), (10.92, 5.41), color: orange)

  circle((14.18, 6.25), radius: 0.31, fill: white, stroke: blue + 0.75pt)
  content(
    (14.18, 6.25),
    text(size: 6.9pt, weight: "bold", fill: ink, tr([PAIR], [ПАРА])),
    anchor: "center",
  )
  arrow(
    (13.24, 7.04), (13.62, 7.04), (13.62, 6.43), (13.89, 6.43),
    color: line-grey,
  )
  arrow(
    (13.24, 5.41), (13.62, 5.41), (13.62, 6.07), (13.89, 6.07),
    color: orange,
  )
  content(
    (11.10, 4.83),
    block(
      width: 6.3cm,
      align(center,
        text(size: 6.8pt, fill: muted,
          tr([same date, lead $tau$, and model parameters $theta$],
             [одна дата, срок $tau$ и параметры модели $theta$]))),
    ),
    anchor: "center",
  )

  // The pair is passed down the right margin, leaving both headers unobstructed.
  arrow(
    (14.49, 6.25), (14.70, 6.25), (14.70, 2.23), (14.49, 2.23),
    color: blue,
  )

  // 3. Forecast displacement and verified skill change are not conflated.
  circle((14.18, 2.23), radius: 0.31, fill: white, stroke: blue + 0.75pt)
  content(
    (14.18, 2.23),
    text(size: 6.9pt, weight: "bold", fill: ink, tr([SCORE], [МЕРЫ])),
    anchor: "center",
  )
  node(
    (10.64, 2.34), (13.38, 3.48),
    blue-pale, blue,
    stack(
      dir: ttb, spacing: 2.2pt,
      text(size: 9.8pt)[$S_(v u)$],
      text(size: 7.2pt, weight: "bold", tr([forecast displacement], [сдвиг прогноза])),
      text(size: 6.8pt, fill: muted, tr([patched vs. baseline], [замена против базовой ветви])),
    ),
  )
  node(
    (10.64, 0.84), (13.38, 1.98),
    blue-pale, blue,
    stack(
      dir: ttb, spacing: 2.2pt,
      text(size: 9.5pt)[$Delta "ACC"_(v u)$],
      text(size: 7.2pt, weight: "bold", tr([verified skill change], [изменение точности])),
      text(size: 6.8pt, fill: muted, tr([against ERA5 targets], [относительно ERA5])),
    ),
  )
  arrow(
    (13.87, 2.39), (13.67, 2.39), (13.67, 2.91), (13.38, 2.91),
    color: blue,
  )
  arrow(
    (13.87, 2.07), (13.67, 2.07), (13.67, 1.41), (13.38, 1.41),
    color: blue,
  )

  circle((9.55, 2.23), radius: 0.31, fill: white, stroke: purple + 0.75pt)
  content(
    (9.55, 2.23),
    text(size: 6.9pt, weight: "bold", fill: ink, tr([POOL], [СВОД])),
    anchor: "center",
  )
  arrow(
    (10.64, 2.91), (10.20, 2.91), (10.20, 2.41), (9.84, 2.41),
    color: blue,
  )
  arrow(
    (10.64, 1.41), (10.20, 1.41), (10.20, 2.05), (9.84, 2.05),
    color: blue,
  )
  content(
    (11.99, 0.52),
    block(
      width: 5.6cm,
      align(center,
        text(size: 6.7pt, fill: muted,
          tr([area-weighted and evaluated separately for each input $v$ and output $u$],
             [с весами площади и раздельно для каждого входа $v$ и выхода $u$]))),
    ),
    anchor: "center",
  )

  // 4. Paired results are aggregated over controlled experimental axes.
  arrow((9.24, 2.23), (7.06, 2.23), color: purple)
  circle((6.75, 2.23), radius: 0.31, fill: white, stroke: purple + 0.75pt)
  content(
    (6.75, 2.23),
    text(size: 6.9pt, weight: "bold", fill: ink, tr([OUT], [ИТОГ])),
    anchor: "center",
  )

  node(
    (3.18, 2.72), (6.02, 3.53),
    purple-pale, purple,
    stack(
      dir: ttb, spacing: 0.3pt,
      text(size: 7.4pt, weight: "bold", tr([Directed transfer matrices], [Матрицы переноса])),
      text(size: 8.2pt)[$19 times 19$],
    ),
  )
  node(
    (3.18, 1.81), (6.02, 2.62),
    purple-pale, purple,
    stack(
      dir: ttb, spacing: 0.3pt,
      text(size: 7.4pt, weight: "bold", tr([Dose--response curves], [Кривые доза--отклик])),
      text(size: 8.0pt)[$alpha = 0, .2, ..., 1$],
    ),
  )
  node(
    (3.18, 0.90), (6.02, 1.71),
    purple-pale, purple,
    stack(
      dir: ttb, spacing: 0.3pt,
      text(size: 7.4pt, weight: "bold", tr([Error geography], [География ошибок])),
      text(size: 6.9pt, fill: muted, tr([maps + latitude profiles], [карты + широтные профили])),
    ),
  )
  route((6.44, 2.23), (6.27, 2.23), (6.27, 3.12), color: purple)
  arrow((6.27, 3.12), (6.02, 3.12), color: purple)
  arrow((6.27, 2.23), (6.02, 2.23), color: purple)
  route((6.27, 2.23), (6.27, 1.30), color: purple)
  arrow((6.27, 1.30), (6.02, 1.30), color: purple)

  node(
    (0.36, 1.12), (2.72, 3.44),
    white, panel-stroke,
    stack(
      dir: ttb, spacing: 3.0pt,
      text(size: 6.9pt,
        tr([1. Rank cross-field transfer], [1. Ранжировать перенос ошибок])),
      text(size: 6.9pt,
        tr([2. Test field loss weights], [2. Проверить веса по полям])),
      text(size: 6.9pt,
        tr([3. Test latitude weights], [3. Проверить широтные веса])),
      text(size: 6.7pt, fill: muted,
        tr([Confirm by retraining], [Подтвердить переобучением])),
    ),
  )
  content(
    (4.61, 0.52),
    block(
      width: 5.4cm,
      align(center,
        text(size: 6.7pt, fill: muted,
          tr([repeat over dates, leads, doses, fields, and grid cells],
             [повтор по датам, срокам, дозам, полям и узлам сетки]))),
    ),
    anchor: "center",
  )
})
