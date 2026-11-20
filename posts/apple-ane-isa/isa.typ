// 图 2:配置地址空间(左)→ PE 放大到位(左)→ TD class(中)→ TD 二进制,按地址一行一个字(右)
// 数据:layer_norm/model.hwx 的 TD1(类型 6);字段与 ane_isa.py 的声明一致
// 配色:Ant Design 色板;代码高亮:GitHub 浅色主题
// 编译: typst compile isa.typ isa.svg
#set page(width: 1110pt, height: 656pt, margin: 0pt, fill: white)
#set text(font: ("PingFang SC", "Helvetica Neue"), size: 8.5pt, fill: rgb("#262626"), lang: "zh")

// Ant Design:bg = 1 号色,fill = 2 号色,mid = 3 号色,fg = 7 号色,ink = 8 号色
#let ant = (
  blue: (bg: rgb("#e6f4ff"), fill: rgb("#bae0ff"), mid: rgb("#91caff"), fg: rgb("#0958d9"), ink: rgb("#003eb3")),
  purple: (bg: rgb("#f9f0ff"), fill: rgb("#efdbff"), mid: rgb("#d3adf7"), fg: rgb("#531dab"), ink: rgb("#391085")),
  cyan: (bg: rgb("#e6fffb"), fill: rgb("#b5f5ec"), mid: rgb("#87e8de"), fg: rgb("#08979c"), ink: rgb("#006d75")),
  geekblue: (bg: rgb("#f0f5ff"), fill: rgb("#d6e4ff"), mid: rgb("#adc6ff"), fg: rgb("#1d39c4"), ink: rgb("#10239e")),
  orange: (bg: rgb("#fff7e6"), fill: rgb("#ffe7ba"), mid: rgb("#ffd591"), fg: rgb("#d46b08"), ink: rgb("#ad4e00")),
  green: (bg: rgb("#f6ffed"), fill: rgb("#d9f7be"), mid: rgb("#b7eb8f"), fg: rgb("#389e0d"), ink: rgb("#237804")),
  magenta: (bg: rgb("#fff0f6"), fill: rgb("#ffd6e7"), mid: rgb("#ffadd2"), fg: rgb("#c41d7f"), ink: rgb("#9e1068")),
  gray: (bg: rgb("#fafafa"), fill: rgb("#f0f0f0"), mid: rgb("#d9d9d9"), fg: rgb("#8c8c8c"), ink: rgb("#595959")),
)
// 字段角色 → 颜色
#let role = (subop: ant.orange, mod: ant.green, src: ant.magenta, raw: ant.gray)
#let dim = rgb("#595959")
#let grid-c = rgb("#e8e8e8")
// GitHub 浅色主题
#let gh = (
  text: rgb("#24292f"), keyword: rgb("#cf222e"), entity: rgb("#8250df"), string: rgb("#0a3069"),
  constant: rgb("#0550ae"), variable: rgb("#953800"), comment: rgb("#6e7781"),
  bg: rgb("#f6f8fa"), border: rgb("#d0d7de"),
)
#let mono(body, ..args) = text(font: ("Menlo", "PingFang SC"), ..args, body)

// ── 绘图原语(坐标单位 pt,原点在左上;文字以 y 为垂直中心) ─────
#let FS = 1.2 // 所有文字的放大倍数
#let put(x, y, body) = place(top + left, dx: x * 1pt, dy: y * 1pt, body)
#let txt(x, y, body, size: 8.5pt) = put(x, y - size * FS / 1pt * 0.42, text(size: size * FS, body))
#let ctxt(x, y, body, size: 8.5pt) = put(x - 150, y - size * FS / 1pt * 0.42,
  box(width: 300pt, align(center, text(size: size * FS, body))))
#let rtxt(x, y, body, size: 8.5pt) = put(x - 300, y - size * FS / 1pt * 0.42,
  box(width: 300pt, align(right, text(size: size * FS, body))))
#let cell(x, y, w, h, fill, stroke: 0.5pt + grid-c, radius: 0pt) = put(x, y,
  rect(width: w * 1pt, height: h * 1pt, fill: fill, stroke: stroke, radius: radius))
#let seg(x1, y1, x2, y2, c: dim, w: 0.8pt) = place(top + left,
  line(start: (x1 * 1pt, y1 * 1pt), end: (x2 * 1pt, y2 * 1pt), stroke: w + c))
#let fit(label, w, max: 8) = calc.min(max, (w - 1.5) / (label.len() * 0.62) / FS) * 1pt
// 框上的标签:压在边框上,像 fieldset 的 legend
#let tag(x, y, body, c, size: 8.2pt, anchor: left) = {
  let b = box(fill: white, stroke: 0.9pt + c.fg, radius: 3pt, inset: (x: 4pt, y: 2pt),
    text(size: size * FS, weight: "bold", fill: c.ink, body))
  if anchor == left { put(x, y - 7, b) } else { put(x - 300, y - 7, box(width: 300pt, align(right, b))) }
}

// ── PE 的 5 个字:(地址, 字段=(msb, lsb, 名字, 角色), 代码行=(名字, 角色, 类型, msb, lsb, RAW?)) ──
#let LS = 15.5
#let pe-words = (
  ("0x0900", ((19, 18, "src2", "src"), (16, 16, "s1", "src"), (13, 12, "nl", "subop"), (6, 6, "c", "subop"),
    (5, 5, "r", "raw"), (4, 2, "op", "subop"), (1, 0, "pool", "subop")), (
    ("pool_mode", "subop", "PoolMode", "1, 0", false),
    ("op", "subop", "PEOp", "4, 2", false),
    ("pe_output_re_lu", "raw", "int", "5", true),
    ("condition", "subop", "PECondition", "6, 6", false),
    ("nl_mode", "subop", "PENonlinear", "13, 12", false),
    ("first_source", "src", "PEFirstSource", "16, 16", false),
    ("second_source", "src", "PESecondSource", "19, 18", false),
  )),
  ("0x0904", ((31, 16, "scale · fp16", "mod"), (15, 0, "bias · fp16", "mod")), (
    ("bias", "mod", "float", "15, 0", false),
    ("scale", "mod", "float", "31, 16", false),
  )),
  ("0x0908", ((31, 16, "pre_scale · fp16", "mod"), (15, 0, "epsilon · fp16", "mod")), (
    ("reduction_epsilon", "mod", "float", "15, 0", false),
    ("pre_scale", "mod", "float", "31, 16", false),
  )),
  ("0x090c", ((31, 0, "final_scale · fp32", "mod"),), (
    ("final_scale", "mod", "float", "31, 0", false),
  )),
  ("0x0910", ((31, 16, "clamp_high · fp16", "mod"), (15, 0, "clamp_low · fp16", "mod")), (
    ("clamp_low", "mod", "float", "15, 0", false),
    ("clamp_high", "mod", "float", "31, 16", false),
  )),
)
#let row-h(n) = calc.max(26, n * LS + 8)

// ── 左 1:配置地址空间(块按字数画高,空洞压缩) ──────────────────────
#let SX = 14
#let SW = 106
#let ST = 60
#txt(SX, 20, [*配置地址空间*], size: 10.5pt)
#txt(SX, 37, mono("0x0000–0x1a18", fill: dim), size: 7.8pt)
#let space = (
  ("Common", "0x0000–0x004c", 40, false),
  ("L2", "0x0500–0x0564", 44, false),
  ("PE", "0x0900–0x0914", 34, true),
  ("NE", "0x0d00–0x0d14", 34, false),
  ("TileDMA-Src", "0x1100–0x11d4", 56, false),
  ("TileDMA-Dst", "0x1500–0x1524", 36, false),
  ("KernelDMA", "0x1900–0x1a18", 62, false),
)
#let y = ST
#let pe-band = (0, 0)
#for (i, (name, range-label, h, hit)) in space.enumerate() {
  let c = if hit { ant.cyan } else { ant.gray }
  if i > 0 {
    cell(SX, y, SW, 11, white, stroke: (left: 0.5pt + ant.gray.mid, right: 0.5pt + ant.gray.mid))
    for k in range(11) { seg(SX + 3 + k * 9.6, y + 9.5, SX + 7.5 + k * 9.6, y + 1.5, c: ant.gray.mid, w: 0.5pt) }
    y += 11
  }
  cell(SX, y, SW, h, c.bg, stroke: if hit { 1.8pt + c.fg } else { 0.8pt + c.mid }, radius: 2pt)
  txt(SX + 8, y + h / 2 - 7, text(weight: "bold", fill: c.ink, name), size: 9.2pt)
  txt(SX + 8, y + h / 2 + 8, mono(range-label, fill: c.ink), size: 7.2pt)
  if hit { pe-band = (y, y + h) }
  y += h
}
#txt(SX, y + 14, text(fill: dim)[斜线:未使用的地址], size: 8pt)

// ── 左 2:PE 放大到位,一行一个 32 位字 ─────────────────────────────
#let AX = SX + SW + 26
#let AW = 54
#let BW = 5.8
#let BX = AX + AW
#let RX = BX + 32 * BW
#let bitx(b) = BX + (31 - b) * BW
#let PT = 172
#txt(AX, PT - 30, [*PE 块* #text(fill: dim)[· 0x0900–0x0914 · 每行 1 个字(32 位)]], size: 9pt)
#cell(AX, PT - 14, AW, 14, ant.gray.fill)
#for b in range(32) {
  cell(bitx(b), PT - 14, BW, 14, ant.gray.fill)
  if calc.rem(b, 4) == 0 or b == 31 { ctxt(bitx(b) + BW / 2, PT - 7, text(fill: dim, str(b)), size: 6pt) }
}
#let y = PT
#let pe-rows = ()
#for (addr, fields, code) in pe-words {
  let h = row-h(code.len())
  cell(AX, y, AW, h, ant.cyan.bg, stroke: 0.6pt + ant.cyan.mid)
  txt(AX + 6, y + h / 2, mono(addr, fill: ant.cyan.ink, weight: "bold"), size: 8.2pt)
  for b in range(32) { cell(bitx(b), y, BW, h, white) }
  for (msb, lsb, label, k) in fields {
    let w = (msb - lsb + 1) * BW
    let c = role.at(k)
    cell(bitx(msb), y, w, h, c.fill, stroke: 0.7pt + c.fg)
    ctxt(bitx(msb) + w / 2, y + h / 2, mono(label, fill: c.ink, weight: "bold"), size: fit(label, w))
  }
  pe-rows.push((y, h))
  y += h
}
#let PB = y
#seg(SX + SW, pe-band.at(0), AX, PT - 14, c: ant.cyan.fg, w: 0.8pt)
#seg(SX + SW, pe-band.at(1), AX, PB, c: ant.cyan.fg, w: 0.8pt)

// ── 中:TD class(GitHub 配色),每组和左边那一行对齐 ───────────────
#let KX = RX + 18
#let KW = 380
#let code-top = PT - 56
#put(KX, code-top, rect(width: KW * 1pt, height: (PB - code-top + 10) * 1pt, fill: white,
  stroke: 0.8pt + gh.border, radius: 5pt))
#put(KX, code-top, rect(width: KW * 1pt, height: 20pt, fill: gh.bg, stroke: 0.8pt + gh.border,
  radius: (top: 5pt)))
#txt(KX + 10, code-top + 10, mono[ane_isa.py #text(fill: gh.comment)[· 节选]], size: 8pt)
#let code-x = KX + 10
// 代码高亮:(片段, 颜色) 依次拼接
#let hl(..parts) = text(font: ("Menlo", "PingFang SC"), size: 8pt * FS,
  parts.pos().map(((s, c)) => if type(c) == color { text(fill: c, s) } else { text(fill: c.at(0), weight: "bold", s) }).join())
#let T = gh.text
#txt(code-x, PT - 20, hl(("class", gh.keyword), (" ", T), ("PE", gh.entity), ("(", T), ("Block", gh.entity), (", ", T),
  ("base", gh.variable), ("=", T), ("0x0900", gh.constant), (", ", T), ("end", gh.variable), ("=", T),
  ("0x0914", gh.constant), (", ", T), ("kind", gh.variable), ("=", T), ("\"comp\"", gh.string), ("):", T)), size: 8pt)
#let decl(name, ty, bits, raw, addr) = {
  let builtin = ty == "int" or ty == "float"
  let parts = ((name, (T,)), (": BitField[", T), (ty, if builtin { gh.constant } else { T }), (", ", T),
    ("at", gh.entity), ("(", T), (addr, gh.constant), (", ", T), (bits, gh.constant), (")", T))
  if raw { parts += ((", ", T), ("RAW", gh.constant)) }
  hl(..parts, ("]", T))
}
#let groups = ()
#for (i, (addr, fields, code)) in pe-words.enumerate() {
  let (ry, rh) = pe-rows.at(i)
  let n = code.len()
  let y0 = ry + rh / 2 - (n - 1) * LS / 2
  for (j, (name, k, ty, bits, raw)) in code.enumerate() {
    let yy = y0 + j * LS
    cell(code-x + 6, yy - 3.5, 7, 7, role.at(k).fill, stroke: 0.6pt + role.at(k).fg, radius: 1pt)
    txt(code-x + 18, yy, decl(name, ty, bits, raw, if addr == "0x090c" { "0x090C" } else { addr }), size: 8pt)
  }
  seg(RX + 1, ry + rh / 2, KX - 1, ry + rh / 2, c: ant.cyan.mid, w: 0.9pt)
  groups.push(ry + rh / 2)
}
#txt(KX + 2, PB + 24, text(fill: dim)[TD1 不写 0x0904 和 0x0910。这两个字取默认值:], size: 8pt)
#txt(KX + 2, PB + 40, text(fill: dim)[scale = 1.0、bias = 0、clamp_low = 0、clamp_high = +inf。], size: 8pt)

// ── 右:TD 二进制,按地址一行一个 32 位字;嵌套框表示包装关系 ─────────
#let DX = KX + KW + 58
#let DA = 40
#let DW = 64
#let DN = 142
#let DR = 14
#let GAP = 16
#let DWID = DA + DW + DN
#txt(DX - 4, 20, [*#mono("layer_norm/model.hwx")*], size: 9.5pt)
#txt(DX - 4, 37, mono("__TEXT,__text", fill: dim), size: 7.8pt)
#let collapsed(y, addr, label) = {
  cell(DX, y, DA, 17, ant.gray.fill)
  txt(DX + 5, y + 8.5, mono(addr, fill: dim), size: 7.4pt)
  cell(DX + DA, y, DW + DN, 17, ant.gray.bg, stroke: 0.5pt + ant.gray.mid)
  txt(DX + DA + 6, y + 8.5, text(fill: dim, label), size: 8pt)
}
#collapsed(50, "4000", [preamble · 16 字节])
#collapsed(67, "4010", [TD0 · #mono("PE_EW_REDUCE_UNARY") · 39 字])

// (起始行, 行数, 名字, 颜色, 值的底色)
#let parts = (
  (0, 8, [header], ant.geekblue),
  (8, 12, [packet → Common], ant.blue),
  (20, 6, [packet → L2], ant.purple),
  (26, 4, [packet → PE], ant.cyan),
)
#let words = (
  ("001e0001", [task_size=30 · tid=1]), ("00000000", []), ("00000000", []), ("00000000", []),
  ("00fff800", []), ("00000000", []), ("00000001", []), ("00000001", [tde = 0:不等待其他 TD]),
  ("be1e8000", [头:起始 0x0000,另写 10 个字]), ("00200010", []), ("00000010", []), ("00000001", []),
  ("00000001", []), ("00200010", []), ("00000010", []), ("00000020", []), ("00000044", []), ("00000007", []),
  ("00000061", [task_type=6]), ("00000000", []),
  ("84d00141", [头:起始 0x0504,另写 4 个字]), ("00400100", []), ("00000040", []), ("00000210", []),
  ("00000020", []), ("00000020", []),
  ("80030240", [头:起始 0x0900,另写 +2、+3]), ("00092010", [→ 0x0900]), ("3c001419", [→ 0x0908]),
  ("39000000", [→ 0x090c]),
)
#let pe-value-role = ("subop", "mod", "mod")
#let D0 = 124
#let part-of(i) = parts.enumerate().find(((j, p)) => p.at(0) <= i and i < p.at(0) + p.at(1)).at(0)
#let drow(i) = D0 + i * DR + part-of(i) * GAP

// 外框:TD1
#let TT = D0 - 22
#let TB = drow(29) + DR + 8
#cell(DX - 9, TT, DWID + 18, TB - TT, white, stroke: 1.4pt + rgb("#434343"), radius: 6pt)
#put(DX - 2, TT - 8, box(fill: rgb("#434343"), radius: 3pt, inset: (x: 5pt, y: 2.5pt),
  text(size: 8.6pt * FS, weight: "bold", fill: white)[TD1 · tid 1 · 30 字 = 120 字节]))

#for (pi, (first, n, label, c)) in parts.enumerate() {
  let top = drow(first) - 5
  let bot = drow(first + n - 1) + DR + 5
  cell(DX - 4, top, DWID + 8, bot - top, c.bg, stroke: 1.2pt + c.fg, radius: 4pt)
  tag(DX + DWID + 4, top, label, c, anchor: right)
  for i in range(first, first + n) {
    let (w, note) = words.at(i)
    let y = drow(i)
    let head = pi > 0 and i == first
    cell(DX, y, DA, DR, white, stroke: 0.4pt + c.mid)
    txt(DX + 5, y + DR / 2, mono(str(0x40b0 + 4 * i, base: 16), fill: dim), size: 7.4pt)
    if head {
      cell(DX + DA, y, DW, DR, white, stroke: 1.1pt + c.ink)
      txt(DX + DA + 6, y + DR / 2, mono(w, weight: "bold", fill: c.ink), size: 8pt)
    } else if pi == 3 {
      let r = role.at(pe-value-role.at(i - first - 1))
      cell(DX + DA, y, DW, DR, r.fill, stroke: 0.6pt + r.fg)
      txt(DX + DA + 6, y + DR / 2, mono(w, fill: r.ink, weight: "bold"), size: 8pt)
    } else {
      cell(DX + DA, y, DW, DR, white, stroke: 0.4pt + c.mid)
      txt(DX + DA + 6, y + DR / 2, mono(w, fill: c.ink), size: 8pt)
    }
    txt(DX + DA + DW + 6, y + DR / 2, text(fill: if head or pi == 3 { c.ink } else { dim }, note), size: 7.6pt)
  }
}
#collapsed(TB + 10, "4130", [TD2 · #mono("PE_GOC") · 54 字])

// ── 中 → 右:写进 PE 的三个字,各连到 packet 里的那个值 ──────────────
#let GX = KX + KW
#for (i, (gi, di, k)) in ((0, 27, "subop"), (2, 28, "mod"), (3, 29, "mod")).enumerate() {
  let gy = groups.at(gi)
  let dy = drow(di) + DR / 2
  let vx = GX + 36 - i * 11
  let c = role.at(k).fg
  seg(GX, gy, vx, gy, c: c, w: 0.9pt)
  seg(vx, gy, vx, dy, c: c, w: 0.9pt)
  seg(vx, dy, DX - 10, dy, c: c, w: 0.9pt)
  place(top + left, polygon(fill: c, stroke: none, ((DX - 10) * 1pt, dy * 1pt), ((DX - 15) * 1pt, (dy - 2.5) * 1pt),
    ((DX - 15) * 1pt, (dy + 2.5) * 1pt)))
}

// ── 图例 ─────────────────────────────────────────────────────────
#let LY = 640
#for (i, (k, label)) in (("subop", "子操作"), ("mod", "修饰"), ("src", "源选择(值的含义未知)"), ("raw", "RAW(含义未知)")).enumerate() {
  let x = AX + (0, 74, 138, 290).at(i)
  cell(x, LY - 6, 14, 12, role.at(k).fill, stroke: 0.7pt + role.at(k).fg, radius: 1.5pt)
  txt(x + 19, LY, label, size: 8.6pt)
}
#cell(AX + 420, LY - 6, 14, 12, white, stroke: 1.1pt + rgb("#434343"), radius: 1.5pt)
#txt(AX + 439, LY, [packet 头], size: 8.6pt)
