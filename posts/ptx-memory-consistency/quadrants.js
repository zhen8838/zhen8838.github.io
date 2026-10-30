/* 四象限坐标系：上面是程序顺序里前面那条访问，下面是后面那条；
 * 每条访问可以是读或写，于是四个位置。release / acquire 用保序箭头画在上面。 */
(function (global) {
'use strict';

const NS = 'http://www.w3.org/2000/svg';
const VW = 460, VH = 264;
const BOX = { w: 152, h: 54 };
const POS = {
  RR: { x: 58,  y: 40  },   // 上·读
  RW: { x: 250, y: 40  },   // 上·写
  WR: { x: 58,  y: 176 },   // 下·读
  WW: { x: 250, y: 176 },   // 下·写
};
const CODE = {
  RR: ['x = data', '读'],
  RW: ['data = 42', '写'],
  WR: ['r = full', '读'],
  WW: ['full = 1', '写'],
};

// 每一节画出哪些保序箭头
const STATE = {
  '3.1': { edges: [], tag: '', note: '上面是前一条访问，下面是后一条；两条各可能是读或写。' },
  '3.2': { edges: [['RR', 'WW', 'release'], ['RW', 'WW', 'release']], tag: 'release',
           note: 'release 夹在中间，箭头终点都在「下·写」：上面的读和写都要先完成。' },
  '3.3': { edges: [['RR', 'WR', 'acquire'], ['RR', 'WW', 'acquire']], tag: 'acquire',
           note: 'acquire 夹在中间，箭头起点都是「上·读」：上面那条读完成之后，下面的才能开始。' },
  '3.4': { edges: [['RR', 'WR', 'acquire'], ['RR', 'WW', 'acquire'], ['RW', 'WW', 'release']],
           tag: 'acq_rel', note: '两种合起来，只剩「上·写 → 下·读」没有箭头。' },
  '3.6': { edges: [['RR', 'WR', 'sc'], ['RR', 'WW', 'sc'], ['RW', 'WW', 'sc'], ['RW', 'WR', 'sc']],
           tag: 'fence.sc', note: 'fence.sc 四条都保，回到第 1 节的顺序。' },
};

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function txt(parent, x, y, s, cls, anchor) {
  const t = el('text', { x, y, class: cls, 'text-anchor': anchor || 'middle' }, parent);
  t.textContent = s;
  return t;
}

function drawNode(svg, key, on) {
  const p = POS[key], c = CODE[key];
  const g = el('g', { class: 'qd-node' + (on ? ' is-on' : '') }, svg);
  el('rect', { x: p.x, y: p.y, width: BOX.w, height: BOX.h, rx: 8, class: 'qd-box' }, g);
  txt(g, p.x + BOX.w / 2, p.y + 24, c[0], 'qd-code');
  txt(g, p.x + BOX.w / 2, p.y + 42, c[1], 'qd-kind');
}

function anchorPoint(from, to) {
  const a = POS[from], b = POS[to];
  const ax = a.x + BOX.w / 2, bx = b.x + BOX.w / 2;
  const ay = a.y + BOX.h, by = b.y;
  return { ax, ay, bx, by };
}

function drawEdge(svg, [from, to, kind], i) {
  const { ax, ay, bx, by } = anchorPoint(from, to);
  const x1 = ax, y1 = ay + 4;
  const x2 = bx, y2 = by - 8;
  const mid = (y1 + y2) / 2;
  const d = `M${x1},${y1} C${x1},${mid} ${x2},${mid} ${x2},${y2}`;
  el('path', { d, class: 'qd-edge qd-e-' + kind }, svg);
  // 箭头
  el('path', { d: `M${x2},${y2} L${x2 - 5},${y2 - 8} L${x2 + 5},${y2 - 8} Z`, class: 'qd-head qd-e-' + kind }, svg);
  if (i === 0) {
    txt(svg, (x1 + x2) / 2, mid - 2, kind, 'qd-edge-label');
  }
}

function mount(host, section) {
  const conf = STATE[section] || STATE['3.1'];
  const card = document.createElement('div');
  card.className = 'tl-card qd-card';

  const head = document.createElement('div');
  head.className = 'tl-head';
  const title = document.createElement('span');
  title.className = 'tl-title';
  title.textContent = '程序顺序上的两条访问';
  head.append(title);
  if (conf.tag) {
    const tag = document.createElement('span');
    tag.className = 'qd-tag';
    tag.textContent = conf.tag;
    head.append(tag);
  }
  card.append(head);

  const svg = el('svg', { viewBox: `0 0 ${VW} ${VH}`, class: 'qd-svg' }, card);

  txt(svg, 134, 20, '读', 'qd-axis');
  txt(svg, 326, 20, '写', 'qd-axis');
  txt(svg, 20, 70, '上', 'qd-axis');
  txt(svg, 20, 206, '下', 'qd-axis');

  const on = new Set();
  for (const [a, b] of conf.edges) { on.add(a); on.add(b); }
  for (const key of Object.keys(POS)) drawNode(svg, key, on.has(key));
  conf.edges.forEach((e, i) => drawEdge(svg, e, i));

  const note = document.createElement('div');
  note.className = 'qd-note';
  note.textContent = conf.note;
  card.append(note);

  host.append(card);
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

global.PTXQuadrant = { mount };
})(window);
