// 图 3:运行时
//   CPU 提交一次程序;任务管理器按顺序从 DRAM 读 TD,写配置寄存器,按 task_type 派发给 NE 或 PE
// 配色:Ant Design 色板,和图 1 一致
// 编译: typst compile runtime.typ runtime.svg
#set page(width: 660pt, height: 248pt, margin: 0pt, fill: white)
#set text(font: ("PingFang SC", "Helvetica Neue"), size: 9.5pt, fill: rgb("#262626"), lang: "zh")

#let dim = rgb("#595959")
#let wire = rgb("#595959")
#let cat = (
  host: (bg: rgb("#fafafa"), fg: rgb("#8c8c8c"), ink: rgb("#262626")),
  ctrl: (bg: rgb("#e6f4ff"), fg: rgb("#0958d9"), ink: rgb("#003eb3")),
  mem: (bg: rgb("#f9f0ff"), fg: rgb("#531dab"), ink: rgb("#391085")),
  comp: (bg: rgb("#e6fffb"), fg: rgb("#08979c"), ink: rgb("#006d75")),
)
#let mono(body, ..args) = text(font: ("Menlo", "PingFang SC"), ..args, body)
#let note(body) = text(fill: dim, body)

// ── 绘图原语(坐标单位 pt,原点在左上;文字以 y 为垂直中心) ─────
#let put(x, y, body) = place(top + left, dx: x * 1pt, dy: y * 1pt, body)
#let txt(x, y, body, size: 9.5pt) = put(x, y - size / 1pt * 0.42, text(size: size, body))
#let ctxt(x, y, body, size: 9.5pt) = put(x - 200, y - size / 1pt * 0.42,
  box(width: 400pt, align(center, text(size: size, body))))
#let seg(x1, y1, x2, y2, c: wire, dash: none) = place(top + left,
  line(start: (x1 * 1pt, y1 * 1pt), end: (x2 * 1pt, y2 * 1pt), stroke: (paint: c, thickness: 1pt, dash: dash)))
#let tip(x, y, d, c) = {
  let s = 6
  let p = if d == "r" { ((x, y), (x - s, y - s / 2), (x - s, y + s / 2)) } else if d == "l" {
    ((x, y), (x + s, y - s / 2), (x + s, y + s / 2))
  } else if d == "d" { ((x, y), (x - s / 2, y - s), (x + s / 2, y - s)) } else {
    ((x, y), (x - s / 2, y + s), (x + s / 2, y + s))
  }
  place(top + left, polygon(fill: c, stroke: none, ..p.map(q => (q.at(0) * 1pt, q.at(1) * 1pt))))
}
#let arrow(x1, y1, x2, y2, c: wire, dash: none) = {
  seg(x1, y1, x2, y2, c: c, dash: dash)
  tip(x2, y2, if x2 > x1 { "r" } else if x2 < x1 { "l" } else if y2 > y1 { "d" } else { "u" }, c)
}
#let box-at(x, y, w, h, k, width: 1.1pt, radius: 5pt) = put(x, y,
  rect(width: w * 1pt, height: h * 1pt, fill: cat.at(k).bg, stroke: width + cat.at(k).fg, radius: radius))

// ── 上:DRAM 里的 TD 序列 ───────────────────────────────────────
#box-at(150, 10, 360, 70, "mem", radius: 6pt)
#txt(162, 26, [#text(weight: "bold", fill: cat.mem.ink)[DRAM] #note[· TD 序列,一条接一条存放]])
#let TDW = 54
#let td-x(i) = 162 + i * (TDW + 13)
#for (i, s) in ("TD0", "TD1", "TD2", "···", "TDn").enumerate() {
  box-at(td-x(i), 42, TDW, 26, "ctrl", width: 0.8pt, radius: 3pt)
  ctxt(td-x(i) + TDW / 2, 55, text(fill: cat.ctrl.ink, s), size: 9pt)
  if i < 4 { arrow(td-x(i) + TDW + 1, 55, td-x(i + 1) - 1, 55, c: cat.ctrl.fg) }
}

// ── 左:CPU 提交一次 ─────────────────────────────────────────────
#box-at(12, 136, 108, 58, "host")
#ctxt(66, 156, text(weight: "bold")[CPU], size: 10pt)
#ctxt(66, 176, note[整个程序提交一次], size: 8.6pt)
#arrow(120, 165, 206, 165)
#ctxt(163, 155, note[启动], size: 8.6pt)

// ── 中:任务管理器,循环 4 步 ───────────────────────────────────
#let CX = 208
#let CW = 186
#box-at(CX, 112, CW, 126, "ctrl")
#txt(CX + 12, 128, text(weight: "bold", fill: cat.ctrl.ink)[任务管理器(CP)])
#let steps = ([读下一条 TD], [有 DTID 就先等], [写配置寄存器], [按 #mono("task_type") 派发])
#let sy(i) = 152 + i * 21
#for (i, s) in steps.enumerate() {
  txt(CX + 14, sy(i), text(fill: cat.ctrl.fg, str(i + 1)), size: 9pt)
  txt(CX + 28, sy(i), s, size: 9pt)
}
// 第 4 步做完回到第 1 步,直到最后一条 TD
#seg(CX + 148, sy(3), CX + 168, sy(3), c: cat.ctrl.fg)
#seg(CX + 168, sy(3), CX + 168, sy(0), c: cat.ctrl.fg)
#arrow(CX + 168, sy(0), CX + 112, sy(0), c: cat.ctrl.fg)
#txt(CX + 126, (sy(1) + sy(2)) / 2, note[循环], size: 8.4pt)

// DRAM → 任务管理器:按顺序读
#arrow(CX + CW / 2, 80, CX + CW / 2, 112, c: cat.ctrl.fg, dash: "dashed")
#txt(CX + CW / 2 + 8, 96, note[按顺序读], size: 8.6pt)

// ── 右:NE 和 PE ───────────────────────────────────────────────
#let UX = 500
#box-at(UX, 112, 148, 52, "comp")
#txt(UX + 12, 130, text(weight: "bold", fill: cat.comp.ink)[NE])
#txt(UX + 12, 150, note[卷积、池化、BYPASS], size: 8.6pt)
#box-at(UX, 186, 148, 52, "comp")
#txt(UX + 12, 204, text(weight: "bold", fill: cat.comp.ink)[PE])
#txt(UX + 12, 224, note[逐元素、归约], size: 8.6pt)
#arrow(CX + CW, 138, UX, 138, c: cat.comp.fg)
#ctxt((CX + CW + UX) / 2, 128, mono("task_type = NE", size: 8.4pt, fill: dim), size: 8.4pt)
#arrow(CX + CW, 212, UX, 212, c: cat.comp.fg)
#ctxt((CX + CW + UX) / 2, 202, mono("task_type = PE_*", size: 8.4pt, fill: dim), size: 8.4pt)
