// 图 1:ANE 总览
//   右上 = DRAM(TD 序列、权重、特征图)
//   左   = 展开 TD1:header + packet,每个 packet 写配置地址空间的一个块
//   右下 = ANE:任务管理器 + 配置地址空间(控制面),NE、PE、L2、DMA(数据面)
//   数据面的模块和连线按专利 US12229657B2 的 Fig. 3–5 画,灰色数字是专利中的编号
// 配色:Ant Design 色板
// 编译: typst compile overview.typ overview.svg
#set page(width: 1240pt, height: 560pt, margin: 0pt, fill: white)
#set text(font: ("PingFang SC", "Helvetica Neue"), size: 9.5pt, fill: rgb("#262626"), lang: "zh")

#let ink = rgb("#262626")
#let dim = rgb("#595959")
#let line-c = rgb("#d9d9d9")
#let wire = rgb("#595959")
#let frame = rgb("#bfbfbf")
// Ant Design:bg = 1 号色,mid = 3 号色,fg = 7 号色,ink = 8 号色
#let cat = (
  ctrl: (bg: rgb("#e6f4ff"), mid: rgb("#91caff"), fg: rgb("#0958d9"), ink: rgb("#003eb3")),
  mem: (bg: rgb("#f9f0ff"), mid: rgb("#d3adf7"), fg: rgb("#531dab"), ink: rgb("#391085")),
  comp: (bg: rgb("#e6fffb"), mid: rgb("#87e8de"), fg: rgb("#08979c"), ink: rgb("#006d75")),
  dma: (bg: rgb("#fffbe6"), mid: rgb("#ffe58f"), fg: rgb("#d48806"), ink: rgb("#ad6800")),
)
#let mono(body, ..args) = text(font: ("Menlo", "PingFang SC"), ..args, body)
#let note(body) = text(fill: dim, body)

// ── 绘图原语(坐标单位 pt,原点在左上;文字以 y 为垂直中心) ─────
#let FS = 1.2 // 所有文字的放大倍数
#let put(x, y, body) = place(top + left, dx: x * 1pt, dy: y * 1pt, body)
#let txt(x, y, body, size: 9.5pt) = put(x, y - size * FS / 1pt * 0.42, text(size: size * FS, body))
#let rtxt(x, y, body, size: 9.5pt) = put(x - 400, y - size * FS / 1pt * 0.42,
  box(width: 400pt, align(right, text(size: size * FS, body))))
#let ctxt(x, y, body, size: 9.5pt) = put(x - 200, y - size * FS / 1pt * 0.42,
  box(width: 400pt, align(center, text(size: size * FS, body))))
#let seg(x1, y1, x2, y2, c: wire, dash: none, w: 1pt) = place(top + left,
  line(start: (x1 * 1pt, y1 * 1pt), end: (x2 * 1pt, y2 * 1pt), stroke: (paint: c, thickness: w, dash: dash)))
#let tip(x, y, d, c) = {
  let s = 6
  let p = if d == "r" { ((x, y), (x - s, y - s / 2), (x - s, y + s / 2)) } else if d == "l" {
    ((x, y), (x + s, y - s / 2), (x + s, y + s / 2))
  } else if d == "d" { ((x, y), (x - s / 2, y - s), (x + s / 2, y - s)) } else {
    ((x, y), (x - s / 2, y + s), (x + s / 2, y + s))
  }
  place(top + left, polygon(fill: c, stroke: none, ..p.map(q => (q.at(0) * 1pt, q.at(1) * 1pt))))
}
// 只画水平/竖直箭头;实线 = 数据流,虚线 = 配置流
#let arrow(x1, y1, x2, y2, c: wire, dash: none) = {
  seg(x1, y1, x2, y2, c: c, dash: dash)
  tip(x2, y2, if x2 > x1 { "r" } else if x2 < x1 { "l" } else if y2 > y1 { "d" } else { "u" }, c)
}
#let cfg-arrow(x1, y1, x2, y2, c: cat.ctrl.fg) = arrow(x1, y1, x2, y2, c: c, dash: "dashed")
#let box-at(x, y, w, h, fill, stroke: 0.6pt + line-c, radius: 0pt) = put(x, y,
  rect(width: w * 1pt, height: h * 1pt, fill: fill, stroke: stroke, radius: radius))
#let unit(x, y, w, h, k) = box-at(x, y, w, h, cat.at(k).bg, stroke: 1.1pt + cat.at(k).fg, radius: 5pt)
#let pill(x, y, w, label) = {
  box-at(x, y - 10, w, 20, cat.dma.bg, stroke: 1.1pt + cat.dma.fg, radius: 10pt)
  ctxt(x + w / 2, y, text(weight: "bold", fill: cat.dma.ink, label), size: 9pt)
}

// ── 行网格:左边 packet 行与右边配置地址空间的块逐行对齐 ───────────
#let RH = 22
#let ROW0 = 212 // 第一个块所在行的上沿
#let rowc(j) = ROW0 + RH * j + RH / 2
#let lx(l) = 14 + 18 * l

#let segs = (
  ("0x0000", "Common", "ctrl", [形状、窗口、tile、操作码 #mono("task_type")]),
  ("0x0500", "L2", "mem", [L2 的地址、格式、融合]),
  ("0x0900", "PE", "comp", [逐元素、池化、归约]),
  ("0x0d00", "NE", "comp", [MAC 运算、权重格式、激活]),
  ("0x1100", "TileDMA-Src", "dma", [输入:DRAM → L2]),
  ("0x1500", "TileDMA-Dst", "dma", [输出:L2 → DRAM]),
  ("0x1900", "KernelDMA", "dma", [权重:DRAM → NE]),
)
// 专利中的编号
#let num(n) = text(fill: dim, size: 0.82em, n)

// ── 右:ANE 外框与控制面外框(先画,底色不能盖住左边伸进来的箭头) ──
#let PX = 296
#let PW = 934
#box-at(PX, 100, PW, 450, rgb("#fafafa"), stroke: 1.3pt + frame, radius: 9pt)
#txt(PX + 12, 119, text(weight: "bold", fill: wire)[ANE], size: 10pt)
#let CX = 318
#let CW = 436
#let CY = 140   // 配置地址空间外框的上沿
// 表格左边留 16pt:左边伸进来的箭头头部完整落在外框和表格之间
#let TX = CX + 16
#let TW = CW - 24
#let C1 = 66
#let C2 = 108
#box-at(CX, CY, CW, 238, white, stroke: 1.1pt + cat.ctrl.fg, radius: 5pt)

// ── 右上:DRAM ───────────────────────────────────────────────
#let XK = 1128  // 权重 → KernelDMA
#let XS = 1166  // 特征图 → TileDMA-Src
#let XD = 1204  // TileDMA-Dst → 特征图
#box-at(PX, 8, PW, 78, cat.mem.bg, stroke: 1.1pt + cat.mem.fg, radius: 6pt)
#txt(PX + 12, 22, [#text(weight: "bold", fill: cat.mem.ink)[DRAM] #note[· 统一内存 · 区域大小不按比例]], size: 9.5pt)
#let region(x, w) = box-at(x, 32, w, 48, white, stroke: 0.8pt + cat.mem.mid, radius: 4pt)
#region(PX + 10, 426)
#txt(PX + 18, 43, [*TD 序列(程序)* #note[· 顺序存放,最后一条 TD 的 header 标记结尾]], size: 8.8pt)
#let TDW = 57
#let td-x(i) = PX + 18 + i * (TDW + 12)
#for (i, s) in ("TD0", "TD1", "TD2", "···", "TDn").enumerate() {
  let hit = s == "TD1"
  box-at(td-x(i), 53, TDW, 22, cat.ctrl.bg, stroke: (if hit { 1.8pt } else { 0.8pt }) + cat.ctrl.fg, radius: 3pt)
  ctxt(td-x(i) + TDW / 2, 64, text(fill: cat.ctrl.ink, weight: if hit { "bold" } else { "regular" }, s), size: 9pt)
  if i < 4 { arrow(td-x(i) + TDW + 1, 64, td-x(i + 1) - 1, 64, c: cat.ctrl.fg) }
}
#region(744, 402)
#ctxt(945, 48, [*权重*], size: 9.5pt)
#ctxt(945, 66, note[kernel], size: 8pt)
#region(1152, 72)
#ctxt(1188, 48, [*特征图*], size: 9.5pt)
#ctxt(1188, 66, note[输入、输出、中间], size: 7pt)

// ── 左:展开 TD1 ─────────────────────────────────────────────
#txt(14, 22, [*程序*], size: 10.5pt)
#txt(14, 42, note[程序 = DRAM 中顺序存放的 TD], size: 8.8pt)
#txt(14, 60, note[下面展开 TD1], size: 8.8pt)
#let tree = (
  (0, [*TD1* #note[· 1 个任务]]),
  (1, [*Header* #note[· TID 和依赖(DTID)]]),
  (1, [*Packets* #note[· 每个 packet 只写 1 个块]]),
  ..segs.map(s => (2, none)),
)
#let trow(i) = rowc(i - 3)
#seg(td-x(1) + TDW / 2, 76, td-x(1) + TDW / 2, 93, c: dim, dash: "dotted")
#seg(td-x(1) + TDW / 2, 93, lx(0) + 6, 93, c: dim, dash: "dotted")
#arrow(lx(0) + 6, 93, lx(0) + 6, trow(0) - 10, c: dim)
#txt(160, 83, note[展开], size: 8.4pt)
#for (i, (l, body)) in tree.enumerate() {
  if l > 0 {
    let p = range(i).rev().find(j => tree.at(j).at(0) == l - 1)
    let vx = lx(l - 1) + 5
    seg(vx, trow(p) + 8, vx, trow(i), c: line-c)
    seg(vx, trow(i), lx(l) - 3, trow(i), c: line-c)
  }
  if body != none { txt(lx(l), trow(i), body, size: 10pt) }
}
#let PKW = 138
#for (j, (addr, name, k, role)) in segs.enumerate() {
  let y = rowc(j)
  box-at(lx(2), y - 9, PKW, 18, cat.at(k).bg, stroke: 0.9pt + cat.at(k).fg, radius: 3pt)
  txt(lx(2) + 6, y, text(fill: cat.at(k).ink)[packet → #name], size: 9pt)
  cfg-arrow(lx(2) + PKW + 3, y, TX - 2, y, c: cat.at(k).fg)
}
#txt(lx(1), rowc(7) + 10, note[TD2、TD3 … 结构相同], size: 8.8pt)

// ── 控制面:任务管理器 ───────────────────────────────────────────
#let MX = 358
#box-at(MX, 108, CX + CW - MX, 22, cat.ctrl.bg, stroke: 1.1pt + cat.ctrl.fg, radius: 4pt)
#txt(MX + 10, 119, [#text(weight: "bold", fill: cat.ctrl.ink)[任务管理器] #num[310] #note[· 任务队列 → 选任务 → 把 packet 写到配置寄存器]], size: 8.8pt)
#cfg-arrow(700, 87, 700, 107)
#cfg-arrow(700, 131, 700, CY - 1)

// ── 控制面:配置地址空间 ──────────────────────────────────────────
#txt(TX, CY + 16, [*配置地址空间(MMIO)* #mono("0x0000–0x1a18")], size: 10pt)
#txt(TX, CY + 36, note[片上寄存器,不在 DRAM 中 · 共 186 个字 · 只存配置,不存数据], size: 8.6pt)
#{
  let y = ROW0 - RH
  box-at(TX, y, C1, RH, rgb("#f5f5f5"))
  box-at(TX + C1, y, C2, RH, rgb("#f5f5f5"))
  box-at(TX + C1 + C2, y, TW - C1 - C2, RH, rgb("#f5f5f5"))
  txt(TX + 8, y + RH / 2, [*地址*], size: 9pt)
  txt(TX + C1 + 8, y + RH / 2, [*块*], size: 9pt)
  txt(TX + C1 + C2 + 8, y + RH / 2, [*用途*], size: 9pt)
}
#for (j, (addr, name, k, role)) in segs.enumerate() {
  let y = ROW0 + RH * j
  let c = cat.at(k)
  box-at(TX, y, C1, RH, c.bg)
  box-at(TX + C1, y, C2, RH, c.bg)
  box-at(TX + C1 + C2, y, TW - C1 - C2, RH, white)
  txt(TX + 8, y + RH / 2, mono(addr, fill: c.ink), size: 9pt)
  txt(TX + C1 + 8, y + RH / 2, text(weight: "bold", fill: c.ink, name), size: 9.5pt)
  txt(TX + C1 + C2 + 8, y + RH / 2, role, size: 9.2pt)
}

// ── 数据面:按专利 US12229657B2 的 Fig. 3–5 ─────────────────────────
#let NX = 772   // NE 的左沿
#let NW = 246
#let NY = 128
#let DB = 462   // NE 和 PE 的下沿
#let L2Y = 484
#let UR = NX + NW   // 数据面单元的右沿
#txt(NX, 114, note[数据面 · 按专利 Fig. 3–5 画 · 灰色数字是专利编号(和 H14 的对应是推断)], size: 8.4pt)
// 控制面 → 数据面
#cfg-arrow(CX + CW + 1, 259, NX - 1, 259)
#cfg-arrow(440, CY + 239, 440, 397)
#txt(450, 388, note[每个块配置同色的单元], size: 8.4pt)
// 单元里的子模块:白底,第 1 行是名字和专利编号,第 2 行是说明
#let sub(x, y, w, h, name, n, extra: none, k: "comp") = {
  box-at(x, y, w, h, white, stroke: 0.8pt + cat.at(k).mid, radius: 3pt)
  if extra == none {
    ctxt(x + w / 2, y + h / 2, [#text(fill: cat.at(k).ink, name) #num(n)], size: 9pt)
  } else {
    ctxt(x + w / 2, y + h / 2 - 7, [#text(fill: cat.at(k).ink, name) #num(n)], size: 9pt)
    ctxt(x + w / 2, y + h / 2 + 8, note(extra), size: 7.4pt)
  }
}

// NE:输入缓冲 → MAC(MAD ⇄ 累加器)→ 后处理 → 输出电路;核提取把权重送进 MAD
#box-at(NX + 5, NY + 5, NW, DB - NY, cat.comp.bg, stroke: 0.9pt + cat.comp.mid, radius: 5pt)
#unit(NX, NY, NW, DB - NY, "comp")
#let NL = NX + 10   // 左列
#let NR = NX + 128  // 右列
#let SW = 108
#txt(NL, NY + 15, [#text(weight: "bold", fill: cat.comp.ink)[NE] #mono("0x0d00", fill: cat.comp.ink) #note[· neural engine] #num[314A…N]], size: 9.5pt)
#txt(NL, NY + 34, note[NE 控制 418 · rasterizer 430], size: 8pt)
#txt(NL, NY + 68, note[多个 NE 并行], size: 8pt)
#txt(NL, NY + 84, note[公开规格:16 个核], size: 8pt)
#sub(NR, 186, SW, 36, [核提取], [432], extra: [LUT + mask 解压])
#let MY = 250
#box-at(NL, MY, NW - 20, 70, white, stroke: 0.8pt + cat.comp.mid, radius: 3pt)
#txt(NL + 8, MY + 13, [#text(fill: cat.comp.ink)[MAC] #num[404] #note[· 乘加]], size: 9pt)
#txt(NL + 8, MY + 46, text(fill: cat.comp.ink)[MAD], size: 9pt)
#for (i, s) in ("0", "1", "…", "N").enumerate() {
  let x = NL + 40 + i * 27
  box-at(x, MY + 35, 24, 22, cat.comp.bg, stroke: 0.8pt + cat.comp.mid, radius: 3pt)
  ctxt(x + 12, MY + 46, text(fill: cat.comp.ink, s), size: 8pt)
}
#arrow(NL + 149, MY + 42, NL + 161, MY + 42, c: cat.comp.fg)
#arrow(NL + 161, MY + 50, NL + 149, MY + 50, c: cat.comp.fg)
#box-at(NL + 163, MY + 35, 56, 22, cat.comp.bg, stroke: 0.8pt + cat.comp.mid, radius: 3pt)
#ctxt(NL + 191, MY + 46, [#text(fill: cat.comp.ink)[累加器] #num[414]], size: 8pt)
#sub(NL, 350, SW, 36, [输入缓冲], [402], extra: [shifter 410])
#sub(NR, 350, SW, 36, [后处理], [428], extra: [ReLU 等,可旁路])
#sub(NR, 412, SW, 24, [输出电路], [424])
#let NI = NL + SW / 2   // 输入的 x
#let NO = NR + SW / 2   // 输出的 x
#arrow(NO, 223, NO, MY - 1)        // 核提取 → MAD
#arrow(NI, 349, NI, MY + 71)       // 输入缓冲 → MAC
#arrow(NO, MY + 71, NO, 349)       // MAC → 后处理
#arrow(NO, 387, NO, 411)           // 后处理 → 输出电路
#arrow(NI, L2Y - 1, NI, 387)       // L2 → 输入缓冲
#arrow(NO, 437, NO, L2Y - 1)       // 输出电路 → L2
#txt(NI + 5, 473, note[输入], size: 8pt)
#txt(NO + 5, 473, note[结果], size: 8pt)

// PE:格式转换 → 水平滤波 → 行缓冲 → 垂直滤波 → 后处理 → 格式转换
#let PY = 398
#unit(CX, PY, CW, DB - PY, "comp")
#txt(CX + 10, PY + 13, [#text(weight: "bold", fill: cat.comp.ink)[PE] #mono("0x0900", fill: cat.comp.ink) #note[· planar engine] #num[340]], size: 9.5pt)
#rtxt(CX + CW - 10, PY + 13, note[PE 控制 530 · rasterizer 540], size: 8pt)
#let stages = (([格式转换], [502]), ([水平滤波], [506]), ([行缓冲], [510]), ([垂直滤波], [514]), ([后处理], [518]), ([格式转换], [522]))
#let QW = 62
#let qx(i) = CX + 7 + i * (QW + 10)
#for (i, (name, n)) in stages.enumerate() {
  box-at(qx(i), PY + 26, QW, 32, white, stroke: 0.8pt + cat.comp.mid, radius: 3pt)
  ctxt(qx(i) + QW / 2, PY + 36, text(fill: cat.comp.ink, name), size: 8.6pt)
  ctxt(qx(i) + QW / 2, PY + 50, num(n), size: 8.6pt)
  if i < 5 { arrow(qx(i) + QW + 1, PY + 42, qx(i + 1) - 1, PY + 42, c: cat.comp.fg) }
}
#let PI = qx(0) + QW / 2
#let PO = qx(5) + QW / 2
#arrow(PI, L2Y - 1, PI, DB + 1)
#arrow(PO, DB + 1, PO, L2Y - 1)
#txt(PI + 5, 473, note[输入], size: 8pt)
#txt(PO + 5, 473, note[结果], size: 8pt)

// L2:数据处理电路 318 = buffer 334 + 流量控制 332
#unit(CX, L2Y, UR - CX, 56, "mem")
#txt(CX + 10, L2Y + 16, [#text(weight: "bold", fill: cat.mem.ink)[L2] #mono("0x0500", fill: cat.mem.ink) #note[· 片上 SRAM · 数据处理电路 318 的 buffer] #num[334]], size: 9.5pt)
#txt(CX + 10, L2Y + 39, note[Alias:相邻 TD 的中间结果留在 L2,不写回 DRAM], size: 8.6pt)
#sub(UR - 256, L2Y + 10, 118, 36, [流量控制], [332], extra: [广播、单播], k: "mem")

// DMA:DRAM ⇄ 片上
#let DX = 1030
#let KY = 204
#let SY = L2Y + 16
#let TY = L2Y + 40
#pill(DX, KY, 82, [KernelDMA])
#pill(DX, SY, 82, [TileDMA-Src])
#pill(DX, TY, 82, [TileDMA-Dst])
#txt(DX, KY - 19, note[kernel DMA #num[324]], size: 7.6pt)
#txt(DX, SY - 21, note[data processor DMA #num[320]], size: 7.6pt)
// 权重 → KernelDMA → NE 的核提取
#seg(XK, 81, XK, KY)
#seg(XK, KY, DX + 83, KY)
#arrow(DX - 1, KY, NR + SW + 1, KY)
// 特征图 → TileDMA-Src → L2
#seg(XS, 81, XS, SY)
#seg(XS, SY, DX + 83, SY)
#arrow(DX - 1, SY, UR + 1, SY)
// L2 → TileDMA-Dst → 特征图
#seg(UR + 1, TY, DX - 1, TY)
#seg(DX + 83, TY, XD, TY)
#arrow(XD, TY, XD, 81)

// ── 左下:图例 ───────────────────────────────────────────────────
#let LY = 412
#txt(14, LY, [*图例*], size: 10pt)
#for (i, (k, s)) in (
  ("ctrl", [控制:TD、Common]),
  ("mem", [存储:DRAM、L2]),
  ("comp", [计算:NE、PE]),
  ("dma", [搬运:3 个 DMA]),
).enumerate() {
  let y = LY + 22 + i * 19
  box-at(14, y - 6, 18, 12, cat.at(k).bg, stroke: 1pt + cat.at(k).fg, radius: 2pt)
  txt(40, y, s, size: 9pt)
}
#for (i, (c, dash, s)) in (
  (wire, none, [数据流]),
  (cat.ctrl.fg, "dashed", [配置流:读 TD、写寄存器]),
  (dim, "dotted", [展开]),
).enumerate() {
  let y = LY + 22 + (4 + i) * 19
  seg(14, y, 32, y, c: c, dash: dash, w: 1.3pt)
  txt(40, y, s, size: 9pt)
}
