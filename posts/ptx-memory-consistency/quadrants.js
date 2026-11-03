/* 四象限：把一个线程的代码按 fence 分成上下，每条访问再分读写，落进四个格子。
 * 箭头只连同一个线程里、fence 两侧的两条访问：蓝色实线＝保序，灰色虚线＝不受约束。
 * 每节只放一个例子：格子里写这个例子的代码，没有访问的格子画「·」。 */
(function (global) {
'use strict';

const NS = 'http://www.w3.org/2000/svg';
const VW = 280, VH = 172;
const BOX = { w: 118, h: 38 };
const COL = { r: 24, w: 156 };   // 读、写两列的左边
const ROW = { u: 26, d: 128 };   // fence 上、下两行的顶边
const FENCE_Y = 96;

// 格子的键：第一个字母 u/d＝fence 上/下，第二个 r/w＝读/写
const CELLS = ['ur', 'uw', 'dr', 'dw'];
const at = key => ({ x: COL[key[1]], y: ROW[key[0]] });

// 每节一个例子
const STATE = {
  '3.2': {
    title: 'P 的代码按 fence 分上下',
    fence: 'fence.release',
    cells: { uw: 'data = 42', dw: 'full = 1', dr: 'while empty != 1' },
    edges: [['uw', 'dw', 'keep'], ['uw', 'dr', 'free']],
    note: 'release 只拦下面的写：full 要等 data，读 empty 不用等。',
  },
  '3.3': {
    title: 'C1 的代码按 fence 分上下',
    fence: 'fence.acquire',
    cells: { ur: 'while full != 1', dr: 'r1 = data' },
    edges: [['ur', 'dr', 'keep']],
    note: 'acquire 拦住下面的读：读 data 要等读 full 完成。C1 上面没有写。',
  },
  '3.6': {
    title: 'C1 的代码按 fence 分上下',
    fence: 'fence.sc',
    cells: { uw: 'done1 = 1', dr: 'r1 = done2' },
    edges: [['uw', 'dr', 'keep']],
    note: 'fence.sc 比 acq_rel 多保证的就是这一对 W→R 的顺序。',
  },
};

const LEGEND = { keep: '保序', free: '不受约束' };

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function txt(parent, x, y, s, cls) {
  const t = el('text', { x, y, class: cls, 'text-anchor': 'middle' }, parent);
  t.textContent = s;
  return t;
}

function drawCell(svg, key, code, on) {
  const p = at(key);
  const g = el('g', { class: 'qd-node' + (code ? '' : ' is-empty') + (on ? ' is-on' : '') }, svg);
  el('rect', { x: p.x, y: p.y, width: BOX.w, height: BOX.h, rx: 7, class: 'qd-box' }, g);
  txt(g, p.x + BOX.w / 2, p.y + BOX.h / 2 + 4, code || '·', code ? 'tl-code qd-code' : 'qd-dot');
}

// 斜着的箭头两端各往对方偏一点，免得和竖着的箭头叠在一起
function drawEdge(svg, from, to, kind) {
  const a = at(from), b = at(to);
  const lean = Math.sign(b.x - a.x) * 22;
  const x1 = a.x + BOX.w / 2 + lean, y1 = a.y + BOX.h + 3;
  const x2 = b.x + BOX.w / 2 - lean, y2 = b.y - 3;
  const len = Math.hypot(x2 - x1, y2 - y1);
  const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
  const bx = x2 - ux * 8, by = y2 - uy * 8;
  el('line', { x1, y1, x2: bx, y2: by, class: 'qd-edge qd-e-' + kind }, svg);
  el('path', {
    d: `M${x2},${y2} L${bx - uy * 4.5},${by + ux * 4.5} L${bx + uy * 4.5},${by - ux * 4.5} Z`,
    class: 'qd-head qd-h-' + kind,
  }, svg);
}

function legendNode(conf) {
  const legend = document.createElement('div');
  legend.className = 'tl-legend';
  for (const kind of Object.keys(LEGEND)) {
    if (!conf.edges.some(e => e[2] === kind)) continue;
    const item = document.createElement('span');
    const icon = el('svg', { viewBox: '0 0 22 10', class: 'tl-ic' });
    el('line', { x1: 1, y1: 5, x2: 15, y2: 5, class: 'qd-edge qd-e-' + kind }, icon);
    el('path', { d: 'M21,5 L14,1.5 L14,8.5 Z', class: 'qd-head qd-h-' + kind }, icon);
    item.append(icon, document.createTextNode(LEGEND[kind]));
    legend.append(item);
  }
  return legend;
}

function mount(host, section) {
  const conf = STATE[section];
  if (!conf) return null;
  const card = document.createElement('div');
  card.className = 'tl-card qd-card';

  const head = document.createElement('div');
  head.className = 'tl-head';
  const title = document.createElement('span');
  title.className = 'tl-title';
  title.textContent = conf.title;
  const tag = document.createElement('span');
  tag.className = 'qd-tag';
  tag.textContent = conf.fence;
  head.append(title, tag);
  card.append(head);

  const svg = el('svg', {
    viewBox: `0 0 ${VW} ${VH}`, class: 'qd-svg', role: 'img',
    'aria-label': conf.title + '：fence 上下两侧的读和写',
  }, card);
  txt(svg, COL.r + BOX.w / 2, 16, '读', 'qd-axis');
  txt(svg, COL.w + BOX.w / 2, 16, '写', 'qd-axis');
  txt(svg, 10, ROW.u + BOX.h / 2 + 4, '上', 'qd-axis');
  txt(svg, 10, ROW.d + BOX.h / 2 + 4, '下', 'qd-axis');
  el('line', { x1: COL.r - 4, y1: FENCE_Y, x2: COL.w + BOX.w + 4, y2: FENCE_Y, class: 'qd-fence' }, svg);

  // 保序箭头两端的格子描蓝边
  const on = new Set();
  for (const [a, b, kind] of conf.edges) if (kind === 'keep') { on.add(a); on.add(b); }
  for (const key of CELLS) drawCell(svg, key, conf.cells[key], on.has(key));
  for (const [a, b, kind] of conf.edges) drawEdge(svg, a, b, kind);

  const note = document.createElement('div');
  note.className = 'qd-note';
  note.textContent = conf.note;
  card.append(note, legendNode(conf));

  host.append(card);
  if (global.PTXVariables) global.PTXVariables.enhance(card);
  return card;
}

function autoMount() {
  document.querySelectorAll('[data-quadrant]').forEach(host => {
    if (host.dataset.mounted) return;
    host.dataset.mounted = 'true';
    mount(host, host.dataset.quadrant);
  });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', autoMount);
else autoMount();

global.PTXQuadrant = { mount, has: section => section in STATE };
})(window);
