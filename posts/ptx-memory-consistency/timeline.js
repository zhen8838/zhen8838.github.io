/*
 * 竖版时间线：时间向下，每个线程一组列，每个操作一列。
 *
 * Timeline.mount(容器, spec) 在容器末尾加一张卡片。spec 的格式：
 *
 *   {
 *     title: '2.2 写操作重叠',
 *     ticks: 9,                                   // 刻度 t1..t9，事件都放在整数刻度上
 *     lanes: [                                    // 通道：每个线程一组列
 *       { name: 'P',  ops: [{ id: 'Wd', code: 'data=42' }, { id: 'Wf', code: 'full=1' }] },
 *       { name: 'C1', ops: [{ id: 'Rf', code: 'r1=full' }, { id: 'Rd', code: 'r2=data' }] },
 *     ],
 *     fences: [                                   // 画在列头两列之间，标出程序中的位置
 *       { between: ['Wd', 'Wf'], label: '.sc', on: [false, true], at: [null, 3] },
 *     ],                                          // on：各状态是否有这条 fence；at：▲ 的刻度（fence.sc）
 *     check: 'Rd',                                // 决定结果的读（可为数组），◆ 与值按结果着色
 *     states: [                                   // 一个或两个状态；两个时出现切换按钮
 *       {
 *         at: {                                   // 每个操作一项，两种写法：
 *           Wd: [1, 8, 9],                        //   [发出, 完成, 可延后到]；发出 = 完成时只画 ◆
 *           Rd: { marks: [[1, 'dia', '0'], [2, 'reg']], span: [1, 2] },
 *         },                                      //   记号：iss ●、dia ◆、stale ◇、reg ·；span 画进行中
 *         val: { Rd: '0' },                       // 读取得的值，标在 ◆ 旁边
 *         result: { text: 'r2 = 0', ok: false, why: 'full=1 先于 data=42 完成' },  // ok 省略时不着色
 *       },
 *       { ... },
 *     ],
 *     edges: [                                    // kind 可以是一个值，或 [状态 0, 状态 1]
 *       { from: 'Wf', to: 'Rf', kind: 'rf' },
 *       { from: 'Wd', to: 'Wf', kind: ['rev', 'keep'] },
 *     ],
 *     toggle: ['P 加 fence.sc', 'P 去掉 fence.sc'],  // 按钮在状态 0、状态 1 时的文字
 *   }
 *
 * 边的 kind：rf 读到；sync 同步；keep 保序；kept 已保序（淡）；rev 反序；free 不受约束；none 不画。
 * op.code 也可以是 [状态 0, 状态 1]。
 */
(function (global) {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const G = { axis: 30, colW: 58, laneGap: 16, right: 34, row: 28 };
  const LEGEND = [
    ['iss', '发出'], ['fly', '进行中'], ['dia', '完成'], ['tail', '可延后'],
    ['reg', '只读寄存器'], ['stale', '读到旧副本'], ['sc', 'fence.sc 的位置'],
    ['rf', '读到'], ['sync', '同步'], ['keep', '保序'], ['rev', '反序'], ['free', '不受约束'],
  ];
  const ARROW = new Set(['rf', 'sync', 'keep', 'kept']);
  let uid = 0;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, anchor) {
    const t = el('text', { x, y, class: cls, 'text-anchor': anchor || 'start' }, parent);
    t.textContent = s;
    return t;
  }
  const lerp = (a, b, p) => a + (b - a) * p;
  const pick = (v, side) => (Array.isArray(v) ? v[side] : v);

  // 两点之间的线段，两端各让出 r1、r2 像素
  function seg(a, b, r1, r2) {
    const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
    return [a.x + dx / L * r1, a.y + dy / L * r1, b.x - dx / L * r2, b.y - dy / L * r2];
  }

  function layout(spec) {
    const x = {}, seps = [], heads = [];
    let cx = G.axis + G.colW / 2;
    spec.lanes.forEach((ln, li) => {
      if (li) {
        seps.push(cx - G.colW / 2 + G.laneGap / 2);
        cx += G.laneGap;
      }
      const first = cx;
      for (const op of ln.ops) {
        x[op.id] = cx;
        cx += G.colW;
      }
      heads.push({ name: ln.name, x: (first + cx - G.colW) / 2 });
    });
    const top = spec.fences && spec.fences.length ? 78 : 64;
    return { x, seps, heads, top, W: cx - G.colW / 2 + G.right, H: top + (spec.ticks - 1) * G.row + 14 };
  }

  function glyph(g, k, x, y, cls) {
    if (k === 'iss') return el('circle', { cx: x, cy: y, r: 4.5, class: 'tl-iss' }, g);
    if (k === 'reg') return el('circle', { cx: x, cy: y, r: 2.5, class: 'tl-reg' }, g);
    if (k === 'sc') return el('path', { d: `M${x},${y - 6.5} L${x + 6.5},${y + 5} L${x - 6.5},${y + 5} z`, class: 'tl-sc' }, g);
    const d = `M${x},${y - 6.5} L${x + 6.5},${y} L${x},${y + 6.5} L${x - 6.5},${y} z`;
    return el('path', { d, class: k === 'stale' ? 'tl-stale' : 'tl-dia' + (cls || '') }, g);
  }

  // 一个操作在一个状态下的画法；opacity 用于两种写法之间的切换
  function drawOp(svg, L, Y, x, v, val, cls, opacity, layer) {
    const g = el('g', { opacity }, svg);
    if (Array.isArray(v)) {
      const [i, d, tail] = v;
      if (layer === 'line') {
        if (tail != null && tail > d + 0.05) el('line', { x1: x, y1: Y(d) + 9, x2: x, y2: Y(tail), class: 'tl-tail' }, g);
        if (Y(d) - Y(i) > 12) el('line', { x1: x, y1: Y(i) + 6, x2: x, y2: Y(d) - 8, class: 'tl-fly' }, g);
        return;
      }
      if (d - i > 0.05) glyph(g, 'iss', x, Y(i));
      glyph(g, 'dia', x, Y(d), cls);
      if (val != null) txt(g, x + 10, Y(d) + 4, val, 'tl-val' + (cls || ''));
      return;
    }
    if (layer === 'line') {
      if (v.span) el('line', { x1: x, y1: Y(v.span[0]) + 6, x2: x, y2: Y(v.span[1]) - 8, class: 'tl-fly' }, g);
      return;
    }
    for (const [t, k, lab] of v.marks) {
      glyph(g, k, x, Y(t), cls);
      if (lab != null) txt(g, x + 10, Y(t) + 4, lab, 'tl-val' + (k === 'dia' ? cls || '' : ''));
    }
  }

  function draw(svg, spec, L, p, mid) {
    svg.replaceChildren();
    const Y = t => L.top + (t - 1) * G.row;
    const n = spec.states.length, side = n > 1 && p >= 0.5 ? 1 : 0;
    const s0 = spec.states[0], s1 = spec.states[n - 1], q = side ? s1 : s0;
    const r = q.result, checks = [].concat(spec.check || []);
    const ccls = id => (checks.includes(id) && r && r.ok !== undefined ? (r.ok ? ' ok' : ' bad') : '');

    const defs = el('defs', {}, svg);
    for (const k of ARROW) {
      const m = el('marker', { id: `${mid}-${k}`, viewBox: '0 0 8 8', refX: 7, refY: 4, markerWidth: 8, markerHeight: 8,
        markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
      el('path', { d: 'M0,0 L8,4 L0,8 z', class: 'tl-mk-' + k }, m);
    }

    for (let t = 1; t <= spec.ticks; t++) {
      el('line', { x1: G.axis, y1: Y(t), x2: L.W - 4, y2: Y(t), class: 'tl-grid' }, svg);
      txt(svg, 4, Y(t) + 3.5, 't' + t, 'tl-tick');
    }
    for (const sx of L.seps) el('line', { x1: sx, y1: 6, x2: sx, y2: L.H - 6, class: 'tl-sep' }, svg);
    for (const h of L.heads) txt(svg, h.x, 18, h.name, 'tl-lane', 'middle');
    for (const ln of spec.lanes) for (const op of ln.ops) txt(svg, L.x[op.id], 40, pick(op.code, side), 'tl-code', 'middle');

    // fence：列头两列之间的竖条与名称；▲ 留到最后画
    const tris = [];
    for (const f of spec.fences || []) {
      const x = (L.x[f.between[0]] + L.x[f.between[1]]) / 2;
      const on = f.on || [true, true], w = [n > 1 ? 1 - p : 1, n > 1 ? p : 0];
      const op = (on[0] ? w[0] : 0) + (on[1] ? w[1] : 0);
      if (op > 0.01) {
        const g = el('g', { opacity: op }, svg);
        el('line', { x1: x, y1: 28, x2: x, y2: 46, class: 'tl-fence' }, g);
        txt(g, x, 60, f.label, 'tl-fence-t', 'middle');
      }
      if (f.at) {
        const a0 = pick(f.at, 0), a1 = n > 1 ? pick(f.at, 1) : a0;
        if (a0 != null && a0 === a1) tris.push([x, a0, op]);
        else {
          if (a0 != null) tris.push([x, a0, w[0]]);
          if (a1 != null) tris.push([x, a1, w[1]]);
        }
      }
    }

    // 各操作：两个状态都是区间时插值，否则淡入淡出
    const ids = Object.keys(Object.assign({}, s0.at, s1.at)), pos = {}, parts = [];
    for (const id of ids) {
      const a = s0.at[id], b = s1.at[id], x = L.x[id];
      if (Array.isArray(a) && Array.isArray(b)) {
        const ta = a[2] != null ? a[2] : null, tb = b[2] != null ? b[2] : null;
        const v = [lerp(a[0], b[0], p), lerp(a[1], b[1], p),
          ta == null && tb == null ? null : lerp(ta != null ? ta : a[1], tb != null ? tb : b[1], p)];
        parts.push([x, v, (q.val || {})[id], ccls(id), 1]);
        pos[id] = { x, y: Y(v[1]) };
      } else {
        if (a) parts.push([x, a, (s0.val || {})[id], side ? '' : ccls(id), n > 1 && a !== b ? 1 - p : 1]);
        if (b && b !== a) parts.push([x, b, (s1.val || {})[id], side ? ccls(id) : '', p]);
        const v = side ? b : a;
        if (v) pos[id] = { x, y: Y(Array.isArray(v) ? v[1] : v.marks[v.marks.length - 1][0]) };
      }
    }
    for (const [x, v, , , op] of parts) if (op > 0.01) drawOp(svg, L, Y, x, v, null, '', op, 'line');

    for (const e of spec.edges || []) {
      const a = pos[e.from], b = pos[e.to];
      if (!a || !b) continue;
      const [x1, y1, x2, y2] = seg(a, b, 10, 11);
      const k0 = pick(e.kind, 0), k1 = n > 1 ? pick(e.kind, 1) : k0;
      const layers = k0 === k1 ? [[k0, 1]] : [[k0, 1 - p], [k1, p]];
      for (const [k, op] of layers) {
        if (k === 'none' || op <= 0.01) continue;
        const attrs = { x1, y1, x2, y2, class: 'tl-e-' + k, opacity: op };
        if (ARROW.has(k)) attrs['marker-end'] = `url(#${mid}-${k})`;
        el('line', attrs, svg);
      }
    }

    for (const [x, v, val, cls, op] of parts) if (op > 0.01) drawOp(svg, L, Y, x, v, val, cls, op, 'mark');
    for (const [x, t, op] of tris) if (op > 0.01) glyph(el('g', { opacity: op }, svg), 'sc', x, Y(t));
  }

  function arrow(svg, k) {
    el('line', { x1: 1, y1: 7, x2: 16, y2: 7, class: 'tl-e-' + k }, svg);
    el('path', { d: 'M15,3 L21,7 L15,11 z', class: 'tl-mk-' + k }, svg);
  }
  const ICON = {
    iss: s => glyph(s, 'iss', 11, 7),
    fly: s => el('line', { x1: 11, y1: 1, x2: 11, y2: 13, class: 'tl-fly' }, s),
    dia: s => glyph(s, 'dia', 11, 7),
    tail: s => el('line', { x1: 11, y1: 0, x2: 11, y2: 14, class: 'tl-tail' }, s),
    reg: s => glyph(s, 'reg', 11, 7),
    stale: s => glyph(s, 'stale', 11, 7),
    sc: s => glyph(s, 'sc', 11, 7.5),
    rf: s => arrow(s, 'rf'),
    sync: s => arrow(s, 'sync'),
    keep: s => arrow(s, 'keep'),
    rev: s => el('line', { x1: 1, y1: 7, x2: 21, y2: 7, class: 'tl-e-rev' }, s),
    free: s => el('line', { x1: 1, y1: 7, x2: 21, y2: 7, class: 'tl-e-free' }, s),
  };

  // 图例只列出这张图用到的符号
  function legend(spec) {
    const used = new Set(['dia']);
    for (const st of spec.states) {
      for (const id in st.at) {
        const v = st.at[id];
        if (Array.isArray(v)) {
          if (v[1] > v[0]) used.add('iss').add('fly');
          if (v[2] != null) used.add('tail');
        } else {
          if (v.span) used.add('fly');
          for (const m of v.marks) used.add(m[1]);
        }
      }
    }
    for (const f of spec.fences || []) if (f.at) used.add('sc');
    for (const e of spec.edges || []) {
      for (const k of [].concat(e.kind)) if (k !== 'none') used.add(k === 'kept' ? 'keep' : k);
    }
    const box = document.createElement('div');
    box.className = 'tl-legend';
    for (const [k, name] of LEGEND) {
      if (!used.has(k)) continue;
      const item = document.createElement('span');
      const ic = el('svg', { viewBox: '0 0 22 14', class: 'tl-ic', 'aria-hidden': 'true' });
      ICON[k](ic);
      item.append(ic, name);
      box.append(item);
    }
    return box;
  }

  function mount(container, spec) {
    const L = layout(spec), mid = 'tl' + (++uid);
    const card = document.createElement('figure');
    card.className = 'tl-card';
    card.style.width = Math.max(L.W + 30, 300) + 'px';
    const head = document.createElement('div');
    head.className = 'tl-head';
    const title = document.createElement('figcaption');
    title.className = 'tl-title';
    title.textContent = spec.title;
    head.append(title);
    const svg = el('svg', { viewBox: `0 0 ${L.W} ${L.H}`, class: 'tl-svg', role: 'img', 'aria-label': spec.title });
    svg.style.maxWidth = L.W + 'px';
    const res = document.createElement('p');
    res.className = 'tl-res';
    card.append(head, svg, res, legend(spec));

    let p = 0, target = 0, raf = 0, btn = null;
    function update() {
      draw(svg, spec, L, p, mid);
      const r = spec.states[p >= 0.5 ? spec.states.length - 1 : 0].result;
      res.replaceChildren();
      res.hidden = !r;
      if (r) {
        const code = document.createElement('code');
        code.textContent = r.text;
        res.className = 'tl-res' + (r.ok === undefined ? '' : r.ok ? ' ok' : ' bad');
        res.append(code);
        if (r.ok !== undefined) {
          const mark = document.createElement('span');
          mark.className = 'tl-mark';
          mark.textContent = r.ok ? '✓' : '✗';
          res.append(mark);
        }
        const why = document.createElement('span');
        why.className = 'tl-why';
        why.textContent = r.why;
        res.append(why);
      }
      if (btn) {
        btn.textContent = spec.toggle[target];
        btn.setAttribute('aria-pressed', String(!!target));
      }
    }
    if (spec.states.length > 1) {
      btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tl-btn';
      head.append(btn);
      btn.addEventListener('click', () => {
        target = 1 - target;
        cancelAnimationFrame(raf);
        if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
          p = target;
          update();
          return;
        }
        const t0 = performance.now(), from = p;
        const step = now => {
          const k = Math.min(1, (now - t0) / 450);
          p = lerp(from, target, k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2);
          update();
          if (k < 1) raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
      });
    }
    update();
    container.append(card);
    return card;
  }

  global.Timeline = { mount };
})(window);
