// 图 4:使用流程
//   左:模型 → .mlpackage → .mlmodelc → model.hwx → ANE;箭头左边是步骤,右边是工具和时间
//   右:从左边的节点展开 —— model.mil(右上)、ANE 编译的阶段(右中)、model.hwx 的段(右下)
// 配色:Ant Design 色板,和图 1 一致;代码高亮:GitHub 浅色主题,和图 2 一致
// 编译: typst compile pipeline.typ pipeline.svg
#set page(width: 890pt, height: 512pt, margin: 0pt, fill: white)
#set text(font: ("PingFang SC", "Helvetica Neue"), size: 10.5pt, fill: rgb("#262626"), lang: "zh")

#let dim = rgb("#595959")
#let wire = rgb("#595959")
#let frame = rgb("#bfbfbf")
#let cat = (
  host: (bg: rgb("#fafafa"), fg: rgb("#8c8c8c"), ink: rgb("#262626")),
  ctrl: (bg: rgb("#e6f4ff"), fg: rgb("#0958d9"), ink: rgb("#003eb3")),
  mem: (bg: rgb("#f9f0ff"), fg: rgb("#531dab"), ink: rgb("#391085")),
  comp: (bg: rgb("#e6fffb"), fg: rgb("#08979c"), ink: rgb("#006d75")),
)
#let gh = (
  text: rgb("#24292f"), keyword: rgb("#cf222e"), entity: rgb("#8250df"), string: rgb("#0a3069"),
  constant: rgb("#0550ae"), variable: rgb("#953800"),
)
#let mono(body, ..args) = text(font: ("Menlo", "PingFang SC"), ..args, body)
#let note(body) = text(fill: dim, body)

// ── 绘图原语(坐标单位 pt,原点在左上;文字以 y 为垂直中心) ─────
#let put(x, y, body) = place(top + left, dx: x * 1pt, dy: y * 1pt, body)
#let txt(x, y, body, size: 10.5pt) = put(x, y - size / 1pt * 0.42, text(size: size, body))
#let ctxt(x, y, body, size: 10.5pt) = put(x - 200, y - size / 1pt * 0.42,
  box(width: 400pt, align(center, text(size: size, body))))
#let rtxt(x, y, body, size: 10.5pt) = put(x - 400, y - size / 1pt * 0.42,
  box(width: 400pt, align(right, text(size: size, body))))
#let seg(x1, y1, x2, y2, c: wire, dash: none) = place(top + left,
  line(start: (x1 * 1pt, y1 * 1pt), end: (x2 * 1pt, y2 * 1pt), stroke: (paint: c, thickness: 1pt, dash: dash)))
#let tip(x, y, d, c) = {
  let s = 6
  let p = if d == "r" { ((x, y), (x - s, y - s / 2), (x - s, y + s / 2)) } else {
    ((x, y), (x - s / 2, y - s), (x + s / 2, y - s))
  }
  place(top + left, polygon(fill: c, stroke: none, ..p.map(q => (q.at(0) * 1pt, q.at(1) * 1pt))))
}
#let arrow(x1, y1, x2, y2, c: wire) = {
  seg(x1, y1, x2, y2, c: c)
  tip(x2, y2, if x2 > x1 { "r" } else { "d" }, c)
}
#let box-at(x, y, w, h, k, width: 1.1pt, radius: 5pt) = put(x, y,
  rect(width: w * 1pt, height: h * 1pt, fill: cat.at(k).bg, stroke: width + cat.at(k).fg, radius: radius))
// 展开面板:虚线框 + 标题
#let panel(x, y, w, h, title) = {
  put(x, y, rect(width: w * 1pt, height: h * 1pt, fill: white,
    stroke: (paint: frame, thickness: 0.9pt, dash: "dashed"), radius: 5pt))
  txt(x + 12, y + 16, title, size: 10.5pt)
}
#let zoom(..pts) = {
  let p = pts.pos()
  for i in range(p.len() - 1) {
    seg(p.at(i).at(0), p.at(i).at(1), p.at(i + 1).at(0), p.at(i + 1).at(1), c: dim, dash: "dotted")
  }
}

// ── MIL 高亮:按词法单元上色 ───────────────────────────────────
#let mil-keywords = ("program", "func")
#let mil-types = ("tensor", "fp16", "int32")
#let mil-functions = ("main", "const", "conv", "BLOBFILE")
#let mil-params = ("val", "path", "offset", "weight", "strides", "x", "name")
#let mil-hl(code) = code.matches(regex("\"[^\"]*\"|[A-Za-z_][A-Za-z_0-9]*|[0-9]+(\\.[0-9]+)?|.")).map(m => {
  let t = m.text
  let c = if t.starts-with("\"") { gh.string } else if t in mil-keywords { gh.keyword } else if t in mil-types {
    gh.constant
  } else if t in mil-functions { gh.entity } else if t.match(regex("^[0-9]")) != none { gh.constant } else if (
    t in mil-params and code.slice(m.end).starts-with(" =")
  ) { gh.variable } else { gh.text }
  text(fill: c, t)
}).join()

// ── 左:流程。5 个节点,节点之间是 4 个步骤 ───────────────────────
#let BX = 14
#let BW = 176
#let BH = 44
#let CX = BX + BW / 2
#let by(i) = 12 + i * 111

#let nodes = (
  ("host", [模型], [PyTorch、TensorFlow]),
  ("mem", mono(".mlpackage"), [#mono("model.mlmodel", size: 8.8pt) + #mono("weight.bin", size: 8.8pt)]),
  ("mem", mono(".mlmodelc"), [#mono("model.mil", size: 8.8pt) + #mono("weight.bin", size: 8.8pt)]),
  ("mem", mono("model.hwx"), [TD 序列 + 权重]),
  ("comp", [ANE], [每次推理执行一次 TD 序列]),
)
#for (i, (k, name, sub)) in nodes.enumerate() {
  box-at(BX, by(i), BW, BH, k)
  ctxt(CX, by(i) + 15, text(weight: "bold", fill: cat.at(k).ink, name), size: 11.5pt)
  ctxt(CX, by(i) + 32, note(sub), size: 9.2pt)
}

#let steps = (
  ([① 转换], mono("coremltools"), [开发时]),
  ([② Core ML 编译], mono("coremlc"), [开发时或设备上]),
  ([③ ANE 编译], mono("ANECompiler"), [首次加载模型时]),
  ([④ 加载和执行], [#mono("aned") + 内核驱动], [每次加载模型时]),
)
#let sy(i) = (by(i) + BH + by(i + 1)) / 2
#for (i, (step, tool, when)) in steps.enumerate() {
  arrow(CX, by(i) + BH + 1, CX, by(i + 1) - 1, c: cat.ctrl.fg)
  rtxt(CX - 10, sy(i), text(fill: cat.ctrl.ink, step), size: 10pt)
  txt(CX + 10, sy(i) - 9, text(size: 9.6pt, tool), size: 9.6pt)
  txt(CX + 10, sy(i) + 9, note(when), size: 9.2pt)
}

// ── 右:展开面板 ────────────────────────────────────────────────
#let PX = 240
#let PW = 640
#let EX = 222   // 展开线的折点

// 右上:model.mil,从 .mlmodelc 展开
#let AY = 12
#let AH = 186
#panel(PX, AY, PW, AH, [#mono("model.mil", weight: "bold") #note[· MIL 文本 · 3×3 卷积模型,节选,… 是省略的内容]])
#zoom((BX + BW, by(2) + BH / 2), (EX, by(2) + BH / 2), (EX, AY + AH / 2), (PX, AY + AH / 2))
// (缩进的字符数, 代码, 说明)
#let mil = (
  (0, "program(1.3)", [MIL 的版本]),
  (0, "func main<ios18>(tensor<fp16, [1, 32, 16, 16]> x) {", [函数 #mono("main"),算子集 ios18,输入 #mono("x")]),
  (2, "tensor<fp16, [32, 32, 3, 3]> probe_weight_0 = const()[…,", none),
  (4, "val = …(BLOBFILE(path = \"…/weights/weight.bin\", offset = 64))];", [大常量:值在 #mono("weight.bin") 中]),
  (2, "tensor<int32, [2]> probe_strides_0 = const()[…, val = …([1, 1])];", [小常量:值在文本中]),
  (2, "tensor<fp16, [1, 32, 16, 16]> probe = conv(weight = probe_weight_0,", none),
  (4, "strides = probe_strides_0, x = x, …)[name = \"probe\"];", [一行一个算子,一个名称赋值一次(SSA)]),
  (0, "} -> (probe);", [函数的输出]),
)
#let CODE = 9pt
#let CHAR = 9 * 0.602   // Menlo 的字符宽度
#let MX = PX + 406      // 说明列
#for (i, (indent, code, what)) in mil.enumerate() {
  let y = AY + 42 + i * 18.6
  txt(PX + 14 + indent * CHAR, y, mono(mil-hl(code), size: CODE), size: CODE)
  if what != none {
    seg(MX - 12, y, MX - 4, y, c: frame)
    txt(MX, y, note(what), size: 9.4pt)
  }
}

// 右中:ANE 编译的阶段,从步骤 ③ 展开
#let CY = 210
#let CH = 148
#panel(PX, CY, PW, CH, [*ANE 编译* #note[· #mono("mil-hwx-compiler", size: 9.6pt) 的阶段(开源实现,目标是 M4 的 H16G)]])
#zoom((CX + 82, sy(2)), (PX, sy(2)))
#let SW = 143
#let SH = 40
#let sx(c) = PX + 13 + c * (SW + 14)
#let row(r) = CY + 34 + r * (SH + 22)
#let stages = (
  ("ctrl", [词法分析、语法分析]),
  ("ctrl", [有类型的 SSA 图]),
  ("ctrl", [规范化、分解]),
  ("ctrl", [结构融合]),
  ("ctrl", [合法性检查、\ 选择数值模式]),
  ("ctrl", [分块、生存期分析、\ SRAM 和 DMA 规划]),
  ("ctrl", [任务编码、程序合成]),
  ("mem", [HWX 文件、绑定清单]),
)
#for (i, (k, name)) in stages.enumerate() {
  let (r, c) = (calc.quo(i, 4), calc.rem(i, 4))
  box-at(sx(c), row(r), SW, SH, k, width: 0.8pt, radius: 3pt)
  put(sx(c), row(r), box(width: SW * 1pt, height: SH * 1pt,
    align(center + horizon, text(size: 9.6pt, fill: cat.at(k).ink, name))))
  if c < 3 { arrow(sx(c) + SW + 1, row(r) + SH / 2, sx(c + 1) - 1, row(r) + SH / 2, c: cat.ctrl.fg) }
}
// 第 1 行的末尾接到第 2 行的开头
#let my = row(0) + SH + 11
#seg(sx(3) + SW / 2, row(0) + SH + 1, sx(3) + SW / 2, my, c: cat.ctrl.fg)
#seg(sx(3) + SW / 2, my, sx(0) + SW / 2, my, c: cat.ctrl.fg)
#arrow(sx(0) + SW / 2, my, sx(0) + SW / 2, row(1) - 1, c: cat.ctrl.fg)

// 右下:model.hwx 的段,从 model.hwx 展开
#let HY = 370
#let HH = 114
#panel(PX, HY, PW, HH, [#mono("model.hwx", weight: "bold") #note[· Mach-O 的变体 · magic 是 #mono("0xBEEFFACE", size: 9.6pt)]])
#zoom((BX + BW, by(3) + BH / 2), (EX, by(3) + BH / 2), (EX, HY + HH / 2), (PX, HY + HH / 2))
#let segs = (
  ("__TEXT", "ctrl", [TD 序列(程序)]),
  ("__KERN_0", "mem", [权重]),
  ("__FVMLIB", "host", [输入和输出的描述]),
)
#for (i, (name, k, what)) in segs.enumerate() {
  let y = HY + 34 + i * 25
  box-at(PX + 14, y, 106, 20, k, width: 0.8pt, radius: 2pt)
  txt(PX + 23, y + 10, mono(name, fill: cat.at(k).ink, size: 9.4pt), size: 9.4pt)
  txt(PX + 134, y + 10, what, size: 10pt)
}
