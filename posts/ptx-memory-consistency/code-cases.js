/* 代码 + 时间线的联合视图：代码在上，时间轴在下，点「运行」逐行执行。
 *
 * CodeRun.mount(容器, spec)
 *
 *   spec = {
 *     title, ticks,
 *     switches: [{ key, label, needs? }],
 *     lanes: [{ name, code: [...] }],
 *     fences: [{ after, code }],
 *     defaults: { key: bool },
 *     states: { '00': { ops, edges, result, hint } },   // 键是各开关的 0/1 位
 *   }
 *
 *   op 有四种写法，都指到自己通道里的某个刻度：
 *     写（发出即生效）  { id, lane, line, tick }
 *     写（进缓冲）      { id, lane, line, iss, land, tail? }   tail = 还可以拖到的刻度
 *     读                { id, lane, line, read, val, bad? }
 *     fence             { id, lane, line, fence }
 *
 *   edge = { from, to, kind }
 *     rf    读到：橙，从写的落地点指向读；跨线程的数据转移
 *     keep  保序：蓝，同一线程内被压住的先后
 *     rev   反序：红虚线，本该读到却来得太晚
 */
(function () {
'use strict';

const cases = {};
window.PTX_CODE_CASES = cases;

cases['2.1'] = {
  title: '2.1 写缓冲与读旁路',
  ticks: 7,
  switches: [
    { key: 'writeBuffer', label: '写缓冲' },
    { key: 'fence', label: 'fence.sc.gpu', needs: 'writeBuffer' },
  ],
  lanes: [
    { name: 'C1', code: ['done1 = 1', 'r1 = done2'] },
    { name: 'C2', code: ['done2 = 1', 'r2 = done1'] },
  ],
  // 插在第 0 行之后，所以读的行号从 1 变成 2
  fences: [{ after: 0, code: 'fence.sc.gpu' }],
  defaults: { writeBuffer: false, fence: false },
  states: {
    // 不开优化：写一发出就落地，读排在自己的写之后
    '00': {
      hint: '不开优化：写一发出就落地。',
      ops: [
        { id: 'D1', lane: 0, line: 0, tick: 0 },
        { id: 'D2', lane: 1, line: 0, tick: 1 },
        { id: 'R1', lane: 0, line: 1, read: 2, val: '1' },
        { id: 'R2', lane: 1, line: 1, read: 3, val: '1' },
      ],
      edges: [
        { from: 'D2', to: 'R1', kind: 'rf' },
        { from: 'D1', to: 'R2', kind: 'rf' },
      ],
      result: { text: 'r1 = 1, r2 = 1', ok: true, why: '两个读都在对方写落地之后，交接完成' },
    },

    // 写缓冲：写先进队列，发出与落地被拉开，两个读都越过了还在路上的写
    '10': {
      hint: '写缓冲开着：写的发出与落地被拉开，中间的读可以越过。',
      ops: [
        { id: 'D1', lane: 0, line: 0, iss: 0, land: 4, tail: 5 },
        { id: 'D2', lane: 1, line: 0, iss: 1, land: 5, tail: 6 },
        { id: 'R1', lane: 0, line: 1, read: 2, val: '0', bad: true },
        { id: 'R2', lane: 1, line: 1, read: 3, val: '0', bad: true },
      ],
      edges: [
        { from: 'D2', to: 'R1', kind: 'rev' },
        { from: 'D1', to: 'R2', kind: 'rev' },
      ],
      result: { text: 'r1 = 0, r2 = 0', ok: false, why: '两个读都取到旧值，没人写 empty，P 一直等' },
    },

    // 插一行 fence：写必须先落地，读才轮到
    '11': {
      hint: 'fence 插在写和读之间：写先落地，读才走。',
      ops: [
        { id: 'F1', lane: 0, line: 1, fence: 2 },
        { id: 'F2', lane: 1, line: 1, fence: 3 },
        { id: 'D1', lane: 0, line: 0, iss: 0, land: 2 },
        { id: 'D2', lane: 1, line: 0, iss: 1, land: 3 },
        { id: 'R1', lane: 0, line: 2, read: 4, val: '1' },
        { id: 'R2', lane: 1, line: 2, read: 5, val: '1' },
      ],
      edges: [
        { from: 'F1', to: 'R1', kind: 'keep' },
        { from: 'F2', to: 'R2', kind: 'keep' },
        { from: 'D2', to: 'R1', kind: 'rf' },
        { from: 'D1', to: 'R2', kind: 'rf' },
      ],
      result: { text: 'r1 = 1, r2 = 1', ok: true, why: 'fence 把写压回读之前，两个读都取到 1' },
    },
  },
};

cases['2.2'] = {
  title: '2.2 写操作重叠',
  ticks: 7,
  switches: [
    { key: 'overlap', label: '写重叠' },
    { key: 'fence', label: 'fence.sc.gpu', needs: 'overlap' },
  ],
  lanes: [
    { name: 'P', code: ['data = 42', 'full = 1'] },
    { name: 'C1', code: ['while full != 1', 'r1 = data'] },
  ],
  // fence 只插在 P 的两个写之间
  fences: [{ lane: 0, after: 0, code: 'fence.sc.gpu' }],
  defaults: { overlap: false, fence: false },
  states: {
    // 不开优化：两个写依次落地
    '00': {
      hint: '不开优化：两个写依次落地。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0 },
        { id: 'Wf', lane: 0, line: 1, tick: 1 },
        { id: 'Rf', lane: 1, line: 0, read: 2, val: '1' },
        { id: 'Rd', lane: 1, line: 1, read: 3, val: '42' },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: '两个写依次落地，C1 读到那份数据' },
    },

    // 写重叠：两个写同时在途，full 先落地
    '10': {
      hint: '写重叠开着：两个写同时在途，落地的先后可以反过来。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, iss: 0, land: 5, tail: 6 },
        { id: 'Wf', lane: 0, line: 1, iss: 1, land: 2 },
        { id: 'Rf', lane: 1, line: 0, read: 3, val: '1' },
        { id: 'Rd', lane: 1, line: 1, read: 4, val: '0', bad: true },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Wf', kind: 'rev' },
      ],
      result: { text: 'r1 = 0', ok: false, why: 'full 先落地，C1 被叫醒时 data 还没到' },
    },

    // 插一行 fence：data 先落地，full 才能走
    '11': {
      hint: 'fence 插在两个写之间：data 先落地，full 才能走。',
      ops: [
        { id: 'F', lane: 0, line: 1, fence: 2 },
        { id: 'Wd', lane: 0, line: 0, iss: 0, land: 2 },
        { id: 'Wf', lane: 0, line: 2, iss: 1, land: 3 },
        { id: 'Rf', lane: 1, line: 0, read: 4, val: '1' },
        { id: 'Rd', lane: 1, line: 1, read: 5, val: '42' },
      ],
      edges: [
        { from: 'F', to: 'Wf', kind: 'keep' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence 把 data 压回 full 之前，C1 读到数据' },
    },
  },
};

cases['2.3'] = {
  title: '2.3 非阻塞读',
  ticks: 7,
  switches: [
    { key: 'nonBlocking', label: '非阻塞读' },
    { key: 'fence', label: 'fence.sc.gpu', needs: 'nonBlocking' },
  ],
  lanes: [
    { name: 'P', code: ['data = 42', 'full = 1'] },
    { name: 'C1', code: ['while full != 1', 'r1 = data'] },
  ],
  // fence 插在 C1 的两次读之间
  fences: [{ lane: 1, after: 0, code: 'fence.sc.gpu' }],
  defaults: { nonBlocking: false, fence: false },
  states: {
    // 不开优化：读等返回，第二次读排在第一次之后
    '00': {
      hint: '不开优化：一次读要等值返回才继续。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0 },
        { id: 'Wf', lane: 0, line: 1, tick: 1 },
        { id: 'Rf', lane: 1, line: 0, read: 2, val: '1' },
        { id: 'Rd', lane: 1, line: 1, read: 3, val: '42' },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: '读 full 返回后才发读 data，取到那份数据' },
    },

    // 非阻塞读：读 data 抢先发出，等 full 变成 1 时它带回的是旧值
    '10': {
      hint: '非阻塞读开着：读一发出就走，后面的读可以先取得值。',
      ops: [
        { id: 'Rd', lane: 1, line: 1, iss: 0, land: 4, val: '0', bad: true },
        { id: 'Wd', lane: 0, line: 0, tick: 1 },
        { id: 'Wf', lane: 0, line: 1, tick: 2 },
        { id: 'Rf', lane: 1, line: 0, read: 3, val: '1' },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r1 = 0', ok: false, why: '读 data 在 t1 就走了，full 到 1 时带回的还是 0' },
    },

    // 插一行 fence：第二次读必须等第一次读完成
    '11': {
      hint: 'fence 插在两次读之间：第二次读要等第一次读完成。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0 },
        { id: 'Wf', lane: 0, line: 1, tick: 1 },
        { id: 'Rf', lane: 1, line: 0, read: 2, val: '1' },
        { id: 'F', lane: 1, line: 1, fence: 3 },
        { id: 'Rd', lane: 1, line: 2, read: 4, val: '42' },
      ],
      edges: [
        { from: 'F', to: 'Rd', kind: 'keep' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence 把第二次读压到第一次读之后' },
    },
  },
};

cases['2.4'] = {
  title: '2.4 读与后续写重叠',
  ticks: 7,
  switches: [
    { key: 'readWrite', label: '读与后续写重叠' },
    { key: 'fence', label: 'fence.sc.gpu', needs: 'readWrite' },
  ],
  lanes: [
    { name: 'C1', code: ['r1 = data', 'done1 = 1'] },
    { name: 'P', code: ['data = 43'] },
  ],
  // fence 插在 C1 的读和写之间
  fences: [{ lane: 0, after: 0, code: 'fence.sc.gpu' }],
  defaults: { readWrite: false, fence: false },
  states: {
    // 不开优化：读取得值之后才写 done1
    '00': {
      hint: '不开优化：读取得值之后，才轮到后面的写。',
      ops: [
        { id: 'Rd', lane: 0, line: 0, read: 0, val: '42' },
        { id: 'Wd1', lane: 0, line: 1, tick: 1 },
        { id: 'W43', lane: 1, line: 0, tick: 2 },
      ],
      edges: [],
      result: { text: 'r1 = 42', ok: true, why: '读在写 done1 之前就取到了这一份数据' },
    },

    // 读与后续写重叠：写先落地，链条跑完一圈，读最后带回的是 43
    '10': {
      hint: '读与后续写重叠开着：读还没返回，后面的写已经生效。',
      ops: [
        { id: 'Rd', lane: 0, line: 0, iss: 0, land: 4, val: '43', bad: true },
        { id: 'Wd1', lane: 0, line: 1, tick: 1 },
        { id: 'W43', lane: 1, line: 0, tick: 2 },
      ],
      edges: [
        { from: 'W43', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r1 = 43', ok: false, why: '读 data 在 t1 发出，P 写下 43 之后它才返回' },
    },

    // 插一行 fence：读必须取到值才写 done1
    '11': {
      hint: 'fence 插在读和写之间：读取到值，才轮到写。',
      ops: [
        { id: 'Rd', lane: 0, line: 0, read: 0, val: '42' },
        { id: 'F', lane: 0, line: 1, fence: 1 },
        { id: 'Wd1', lane: 0, line: 2, tick: 2 },
        { id: 'W43', lane: 1, line: 0, tick: 3 },
      ],
      edges: [
        { from: 'F', to: 'Wd1', kind: 'keep' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence 把写压到读取到值之后' },
    },
  },
};
})();
