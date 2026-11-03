/*
 * 代码 + 时间线合成一张图：每条通道的上方是它自己的代码行，下方是它自己的时间轴。
 * 点「运行」代码逐行高亮、时间点逐个打上去，跑完给出结果。
 * 记号沿用时间线那一套：● 发出、◆ 完成（写生效或读到值）、后面接长虚线表示还能更晚；
 * 跨通道的数据转移画箭头，橙＝读到、蓝＝保序、红色虚线＝反序。
 */
(function (global) {
'use strict';

const NS = 'http://www.w3.org/2000/svg';
const ROW = 30;
const STEP_MS = 600;

const SHAPES = {
  iss: '<circle cx="6" cy="6" r="3.8"/>',
  dia: '<path d="M6,1.4 L10.6,6 L6,10.6 L1.4,6 Z"/>',
  sc: '<path d="M6,1.8 L11,10.2 L1,10.2 Z"/>',
};

function div(cls, text) {
  const node = document.createElement('div');
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function svgEl(tag, attrs) {
  const node = document.createElementNS(NS, tag);
  for (const key in attrs) node.setAttribute(key, attrs[key]);
  return node;
}

function icon(kind) {
  const svg = svgEl('svg', { viewBox: '0 0 12 12', class: 'cr-ic' });
  svg.innerHTML = SHAPES[kind];
  return svg;
}

function arrowIcon(kind) {
  const svg = svgEl('svg', { viewBox: '0 0 16 10', class: 'cr-ic cr-ic-arrow' });
  svg.innerHTML = '<line x1="1" y1="5" x2="12" y2="5" class="cr-e-' + kind + '"/>' +
    '<path d="M15,5 L10,2.4 L10,7.6 Z" class="cr-h-' + kind + '"/>';
  return svg;
}

function tailIcon() {
  const svg = svgEl('svg', { viewBox: '0 0 12 12', class: 'cr-ic' });
  svg.innerHTML = '<rect x="5" y="1" width="2" height="3.5"/>' +
    '<rect x="5" y="7" width="2" height="4.5"/>';
  return svg;
}

const LEGEND = [
  ['shape', 'iss'], ['shape', 'dia'], ['tail', null], ['shape', 'sc'],
  ['arrow', 'rf'], ['arrow', 'keep'], ['arrow', 'rev'], ['arrow', 'free'], ['arrow', 'order'], ['arrow', 'sync'],
];
const LEGEND_TEXT = {
  iss: '发出', dia: '完成', tail: '还能更晚', sc: 'fence',
  rf: '读到', keep: '保序', rev: '反序', free: '不受约束', order: 'fence.sc 的先后', sync: 'barrier 同步',
};
// 只在用到时才进图例
const OPTIONAL = new Set(['free', 'order', 'sync']);

function legendNode(spec) {
  const used = new Set();
  for (const st of Object.values(spec.states)) for (const e of st.edges || []) used.add(e.kind);
  const legend = div('tl-legend');
  for (const [type, kind] of LEGEND) {
    if (OPTIONAL.has(kind) && !used.has(kind)) continue;
    const item = document.createElement('span');
    const glyph = type === 'tail' ? tailIcon() : type === 'arrow' ? arrowIcon(kind) : icon(kind);
    item.append(glyph, document.createTextNode(LEGEND_TEXT[kind || 'tail']));
    legend.append(item);
  }
  return legend;
}

function mount(host, spec) {
  const rows = spec.ticks;
  const yOf = tick => (tick + 0.5) * ROW;

  // ---- 骨架 ----
  const card = div('tl-card cr-card' + (spec.dense ? ' is-dense' : ''));
  const head = div('tl-head');
  head.append(div('tl-title', spec.title));

  const controls = div('cr-controls');
  const runBtn = document.createElement('button');
  runBtn.type = 'button';
  runBtn.className = 'tl-btn cr-run';
  runBtn.textContent = '▶ 运行';
  controls.append(runBtn);

  const state = {};
  const switches = spec.switches.map(s => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'cr-switch';
    btn.setAttribute('role', 'switch');
    btn.append(div('cr-switch-dot'));
    const label = document.createElement('span');
    label.textContent = s.label;
    btn.append(label);
    btn.addEventListener('click', () => flip(s));
    controls.append(btn);
    state[s.key] = !!spec.defaults[s.key];
    return { spec: s, btn };
  });
  head.append(controls);
  card.append(head);

  const hint = div('cr-hint');
  card.append(hint);

  const body = div('cr-body');
  // 窄屏时代码放不下，代码和时间轴这一块单独横向滚动，标题、按钮、结果不动
  const scroller = div('cr-scroll');
  scroller.append(body);
  card.append(scroller);

  const res = div('tl-res');
  res.hidden = true;
  card.append(res);
  card.append(legendNode(spec));
  host.append(card);

  // 每次 render 重建：列 = 每条通道的各条指令
  let columns = [];       // [{ op, lane, rail, cell }]
  let plot = null;
  let canvas = null;
  let resizeObserver = null;
  let sizing = null;      // { codes: [{ code, span, weight }], perLane, total }

  // ---- 回放 ----
  const timers = new Set();
  let stamped = new Set();
  let drawn = new Map();
  let current = null;
  let focused = null;

  function stop() {
    for (const id of timers) clearTimeout(id);
    timers.clear();
  }
  function later(fn, ms) {
    const id = setTimeout(() => { timers.delete(id); fn(); }, ms);
    timers.add(id);
  }

  const stateKey = () => spec.switches.map(s => (state[s.key] ? '1' : '0')).join('');
  const railOf = opId => columns.find(c => c.op.id === opId);

  function codeOf(lane, index) {
    // 开关打开时整段换成 swap.code
    const lines = (lane.swap && state[lane.swap.key] ? lane.swap.code : lane.code).slice();
    for (const f of spec.fences) {
      if (f.show !== true && !state[f.show || 'fence']) continue;
      if (f.lane != null && f.lane !== index) continue;
      lines.splice(f.after + 1, 0, f.swap && state[f.swap.key] ? f.swap.code : f.code);
    }
    // 插进来的和本来就写在代码里的 fence 行都标出来
    const fences = new Set();
    lines.forEach((text, i) => { if (/^\s*fence\./.test(text)) fences.add(i); });
    return { lines, fences };
  }

  function syncSwitches() {
    for (const { spec: s, btn } of switches) {
      const on = !!state[s.key];
      btn.setAttribute('aria-checked', String(on));
      btn.classList.toggle('is-on', on);
      const blocked = s.needs && !state[s.needs];
      btn.disabled = !!blocked;
      const dep = blocked && spec.switches.find(x => x.key === s.needs);
      btn.title = blocked && dep ? '先打开' + dep.label : '';
    }
  }

  // ---- 一个 op 在哪些刻度上留记号 ----
  function marksOf(op) {
    const label = op.label;
    if (op.fence != null) return [{ tick: op.fence, kind: 'sc', label }];
    if (op.read != null) return [{ tick: op.read, kind: 'dia', val: op.val, bad: op.bad, label }];
    if (op.iss != null) return [
      { tick: op.iss, kind: 'iss' },
      { tick: op.land, kind: 'dia', tail: op.tail, val: op.val, bad: op.bad, label },
    ];
    return [{ tick: op.tick, kind: 'dia', label }];
  }

  // 箭头的落点：写取生效刻度，读取读取刻度，fence 取自己的位置
  function anchorOf(op) {
    const col = railOf(op.id);
    if (!col) return null;
    const tick = op.fence != null ? op.fence
      : op.read != null ? op.read
      : op.land != null ? op.land
      : op.tick;
    if (tick == null) return null;
    const r = col.rail.getBoundingClientRect();
    const p = plot.getBoundingClientRect();
    return { x: r.left - p.left + r.width / 2, y: r.top - p.top + yOf(tick) };
  }

  function drawArrow(from, to, kind) {
    const dx = to.x - from.x, dy = to.y - from.y;
    const len = Math.hypot(dx, dy);
    if (len < 14) return;
    const ux = dx / len, uy = dy / len;
    const x1 = from.x + ux * 7, y1 = from.y + uy * 7;
    const x2 = to.x - ux * 12, y2 = to.y - uy * 12;
    // 交叉的两条线分别往两边弓，避免重叠成一团；
    // 同一条轴上的两点往左弓出去，不压住轴，也避开轴右边的标签
    const bow = Math.abs(dx) < 1 ? 18 * uy : len * 0.11 * (dy > 0 ? 1 : -1);
    const cx = (x1 + x2) / 2 - uy * bow, cy = (y1 + y2) / 2 + ux * bow;
    canvas.append(svgEl('path', {
      d: `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`,
      class: 'cr-edge cr-e-' + kind,
    }));
    // 箭头方向取终点处的切线
    const tx = x2 - cx, ty = y2 - cy;
    const tl = Math.hypot(tx, ty) || 1;
    const vx = tx / tl, vy = ty / tl;
    const back = 5.5, side = 3.6;
    const bx = x2 - vx * back, by = y2 - vy * back;
    canvas.append(svgEl('path', {
      d: `M${x2},${y2} L${bx - vy * side},${by + vx * side} L${bx + vy * side},${by - vx * side} Z`,
      class: 'cr-head cr-h-' + kind,
    }));
  }

  function flushEdges() {
    if (!current) return;
    const ops = new Map(current.ops.map(op => [op.id, op]));
    for (const edge of current.edges || []) {
      const key = edge.from + '>' + edge.to;
      if (drawn.has(key)) continue;
      if (!stamped.has(edge.from) || !stamped.has(edge.to)) continue;
      const a = anchorOf(ops.get(edge.from));
      const b = anchorOf(ops.get(edge.to));
      if (!a || !b) continue;
      drawn.set(key, { from: edge.from, to: edge.to, kind: edge.kind });
      drawArrow(a, b, edge.kind);
    }
  }

  function redrawEdges() {
    if (!canvas) return;
    canvas.replaceChildren();
    const ops = new Map((current ? current.ops : []).map(op => [op.id, op]));
    for (const { from, to, kind } of drawn.values()) {
      const a = anchorOf(ops.get(from));
      const b = anchorOf(ops.get(to));
      if (a && b) drawArrow(a, b, kind);
    }
  }

  function setFocused(key) {
    if (focused === key) return;
    if (focused) {
      const prev = card.querySelector('[data-key="' + focused + '"]');
      if (prev) prev.classList.remove('is-active');
    }
    focused = key;
    if (!key) return;
    const node = card.querySelector('[data-key="' + key + '"]');
    if (node) node.classList.add('is-active');
  }

  function stamp(op, mark) {
    const col = railOf(op.id);
    if (!col) return;
    const y = yOf(mark.tick);

    // 先画线，再把记号盖上去
    if (mark.kind === 'iss' && op.land != null && op.land > mark.tick) {
      const fly = div('cr-fly');
      fly.style.top = y + 7 + 'px';
      fly.style.height = (op.land - mark.tick) * ROW - 16 + 'px';
      col.rail.append(fly);
    }
    if (mark.tail != null && mark.tail > mark.tick) {
      const tail = div('cr-tail');
      tail.style.top = y + 9 + 'px';
      tail.style.height = (mark.tail - mark.tick) * ROW - 14 + 'px';
      col.rail.append(tail);
    }

    const node = div('cr-mark');
    node.dataset.kind = mark.kind;
    node.style.top = y + 'px';
    node.append(icon(mark.kind));
    col.rail.append(node);

    const text = mark.label != null ? mark.label : mark.val;
    if (text != null) {
      const value = document.createElement('span');
      value.className = 'cr-val' + (mark.bad ? ' bad' : '');
      value.textContent = text;
      value.style.top = y + 'px';
      col.rail.append(value);
    }

    setFocused('op:' + op.id);
    stamped.add(op.id);
    flushEdges();
  }

  function finish(st) {
    runBtn.textContent = '▶ 重新运行';
    runBtn.classList.remove('is-running');
    res.replaceChildren();
    const code = document.createElement('code');
    code.textContent = st.result.text;
    const mark = document.createElement('span');
    mark.className = 'tl-mark';
    mark.textContent = st.result.ok ? '✓' : '✗';
    const why = document.createElement('span');
    why.className = 'tl-why';
    why.textContent = st.result.why;
    res.className = 'tl-res ' + (st.result.ok ? 'ok' : 'bad');
    res.append(code, mark, why);
    res.hidden = false;
  }

  function run() {
    stop();
    const st = current;
    if (!st) return;

    for (const col of columns) {
      for (const node of col.rail.querySelectorAll('.cr-mark, .cr-fly, .cr-tail, .cr-val')) node.remove();
    }
    canvas.replaceChildren();
    stamped = new Set();
    drawn = new Map();
    res.hidden = true;
    setFocused(null);

    const events = [];
    for (const op of st.ops) for (const mark of marksOf(op)) events.push({ op, mark });
    events.sort((a, b) => a.mark.tick - b.mark.tick);

    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) {
      for (const ev of events) stamp(ev.op, ev.mark);
      finish(st);
      return;
    }
    runBtn.classList.add('is-running');
    runBtn.textContent = '运行中…';
    events.forEach((ev, i) => later(() => stamp(ev.op, ev.mark), i * STEP_MS));
    later(() => finish(st), events.length * STEP_MS);
  }

  // 一列时间轴：列头写行号（按线程一条轴时不写）
  function railCell(head, perLane) {
    const cell = div('cr-col');
    const rail = div(perLane ? 'cr-rail cr-rail-lane' : 'cr-rail');
    rail.style.setProperty('--rows', rows);
    for (let t = 0; t < rows; t++) {
      const guide = div('cr-guide');
      guide.style.top = yOf(t) + 'px';
      rail.append(guide);
    }
    cell.append(div('cr-col-no', head), rail);
    plot.append(cell);
    return { cell, rail };
  }

  // ---- 布局：每条通道一列，列里按指令再分通道 ----
  function render() {
    stop();
    current = spec.states[stateKey()];
    runBtn.textContent = '▶ 运行';
    runBtn.classList.remove('is-running');

    body.replaceChildren();
    columns = [];
    plot = null;
    canvas = null;

    // 每个通道的指令，按代码行排序
    const lanes = spec.lanes.map((lane, li) => {
      const ops = (current ? current.ops : []).filter(op => op.lane === li)
        .sort((a, b) => a.line - b.line);
      return { lane, li, ops };
    });
    // rails: 'lane' 时每条通道只有一条时间轴，否则每条指令一条
    const perLane = spec.rails === 'lane';
    const total = (perLane ? lanes.length : lanes.reduce((n, l) => n + l.ops.length, 0)) || 1;

    // 按线程一条轴时，各通道可以用 width 调宽窄（默认 1）
    const tracks = perLane
      ? lanes.map(({ lane }) => `minmax(0, ${lane.width || 1}fr)`).join(' ')
      : `repeat(${total}, minmax(0, 1fr))`;
    // 紧凑的卡片列间距小一点，长代码多一点位置
    const gap = spec.dense ? 8 : 12;
    body.style.gridTemplateColumns = `26px ${tracks}`;
    body.style.columnGap = gap + 'px';

    const axis = div('cr-axis');
    axis.style.setProperty('--rows', rows);
    axis.style.gridColumn = '1';
    axis.style.gridRow = '3';
    for (let t = 0; t < rows; t++) axis.append(div(null, 't' + (t + 1)));
    body.append(axis);

    plot = div('cr-plot');
    plot.style.gridColumn = `2 / -1`;
    plot.style.gridRow = '3';
    plot.style.gridTemplateColumns = tracks;
    plot.style.columnGap = gap + 'px';
    body.append(plot);

    const codes = [];
    let cursor = 2;
    for (const { lane, li, ops } of lanes) {
      const span = perLane ? 1 : ops.length || 1;
      const start = cursor;
      cursor += span;

      const name = div('cr-lane-name', lane.name);
      name.style.gridColumn = `${start} / ${start + span}`;
      name.style.gridRow = '1';
      body.append(name);

      const { lines, fences } = codeOf(lane, li);
      const code = div('cr-code');
      code.style.gridColumn = `${start} / ${start + span}`;
      code.style.gridRow = '2';
      lines.forEach((text, i) => {
        const line = div('cr-line');
        line.dataset.key = li + ':' + i;
        if (fences.has(i)) line.classList.add('is-fence');
        const no = document.createElement('span');
        no.className = 'cr-no';
        no.textContent = String(i + 1);
        const src = document.createElement('span');
        src.className = 'cr-src';
        src.textContent = text;
        line.append(no, src);
        // 点一下代码行，高亮它自己的通道
        const op = ops.find(o => o.line === i);
        if (op) {
          line.dataset.op = op.id;
          line.dataset.key = 'op:' + op.id;
          line.addEventListener('click', () => setFocused('op:' + op.id));
        }
        code.append(line);
      });
      body.append(code);
      codes.push({ code, span, weight: perLane ? lane.width || 1 : 1 });

      if (perLane) {
        const { cell, rail } = railCell('', true);
        for (const op of ops) columns.push({ op, lane: li, rail, cell });
      } else {
        for (const op of ops) {
          const { cell, rail } = railCell(String(op.line + 1));
          columns.push({ op, lane: li, rail, cell });
        }
      }
    }

    canvas = svgEl('svg', { class: 'cr-canvas' });
    plot.append(canvas);

    hint.textContent = current ? current.hint : '';
    res.hidden = true;
    res.replaceChildren();
    syncSwitches();
    if (global.PTXVariables) global.PTXVariables.enhance(card);

    sizing = { codes, perLane, total, gap };
    fitWidth();
    if (typeof ResizeObserver === 'function') {
      if (!resizeObserver) resizeObserver = new ResizeObserver(() => { fitWidth(); if (drawn.size) redrawEdges(); });
      resizeObserver.disconnect();
      // 卡片跟着屏幕变宽变窄；代码字号在窄屏会变小，所以每次都重新量
      resizeObserver.observe(card);
      if (plot) resizeObserver.observe(plot);
    }
  }

  // 代码不能挤在一起：量出每段代码最长一行的实际宽度，算出整块至少要多宽
  function fitWidth() {
    if (!sizing) return;
    const { codes, perLane, total, gap } = sizing;
    const px = v => parseFloat(v) || 0;
    const weightSum = codes.reduce((n, c) => n + c.weight, 0);
    let plotNeed = 0;
    for (const { code, span, weight } of codes) {
      const cs = getComputedStyle(code);
      let widest = 0;
      for (const line of code.querySelectorAll('.cr-line')) {
        const ls = getComputedStyle(line);
        widest = Math.max(widest, line.querySelector('.cr-no').offsetWidth + line.querySelector('.cr-src').offsetWidth
          + px(ls.columnGap) + px(ls.paddingLeft) + px(ls.paddingRight));
      }
      const need = widest + px(cs.paddingLeft) + px(cs.paddingRight) + px(cs.borderLeftWidth) + px(cs.borderRightWidth) + 2;
      // 按线程一条轴时列宽按 weight 分；否则各列一样宽，这段代码横跨 span 列（中间夹着 span - 1 个间隔）
      plotNeed = Math.max(plotNeed, perLane ? need * weightSum / weight : (need - gap * (span - 1)) / span * total);
    }
    // 26px 时间刻度列 + 每列前面一个间隔
    const min = Math.ceil(26 + gap * total + plotNeed) + 'px';
    if (body.style.minWidth !== min) body.style.minWidth = min;
  }

  function flip(s) {
    if (s.needs && !state[s.needs]) return;
    state[s.key] = !state[s.key];
    for (const other of switches) {
      if (other.spec.needs && !state[other.spec.needs]) state[other.spec.key] = false;
    }
    render();
    run();
  }

  runBtn.addEventListener('click', run);
  render();

  return card;
}

function autoMount() {
  for (const host of document.querySelectorAll('[data-code-run]')) {
    if (host.dataset.mounted) continue;
    const spec = (global.PTX_CODE_CASES || {})[host.dataset.codeRun];
    if (!spec) continue;
    host.dataset.mounted = 'true';
    mount(host, spec);
  }
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', autoMount);
else autoMount();

global.PTXCodeRun = { mount };
})(window);
