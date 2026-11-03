/* 代码 + 时间线的联合视图：代码在上，时间轴在下，点「运行」逐行执行。
 *
 * CodeRun.mount(容器, spec)
 *
 *   spec = {
 *     title, ticks,
 *     rails?: 'lane',   // 每条通道一条时间轴；不写时每条指令一条
 *     dense?: true,     // 代码字号小一号，三条通道放长代码时用
 *     switches: [{ key, label, needs? }],
 *     lanes: [{ name, code: [...], swap?, width? }],
 *       swap   { key, code: [...] }：开关 key 打开时，整段代码换成 code
 *       width  按线程一条轴时这条通道的宽窄，默认 1
 *     fences: [{ lane?, after, code, show?, swap? }],
 *       show  哪个开关打开时插这一行，默认 'fence'；true＝一直插
 *       swap  { key, code }：开关 key 打开时，这一行换成 code
 *     defaults: { key: bool },
 *     states: { '00': { ops, edges, result, hint } },   // 键是各开关的 0/1 位；没有开关时键是 ''
 *   }
 *
 *   op 有四种写法，都指到自己通道里的某个刻度：
 *     写（发出即生效）  { id, lane, line, tick }
 *     写（进缓冲）      { id, lane, line, iss, land, tail? }   tail = 还可以拖到的刻度
 *     读                { id, lane, line, read, val, bad? }
 *     fence             { id, lane, line, fence }
 *   任何 op 都可以带 label：记号旁的短标签（如 'done2=1'），不写时显示 val
 *
 *   edge = { from, to, kind }
 *     rf    读到：橙，从写的生效点指向读；跨线程的数据转移
 *     keep  保序：蓝，同一线程内被压住的先后
 *     rev   反序：红虚线，本该读到却来得太晚
 *     free  不受约束：灰虚线，fence 不管的那一对
 *     order fence.sc 的先后：紫，两个 fence.sc 排出的先后（不是时间）
 *     sync  barrier 同步：紫虚线，bar.arrive 到 bar.sync 这一类不用读到值的同步
 */
(function () {
'use strict';

const cases = {};
window.PTX_CODE_CASES = cases;

cases['2.1'] = {
  title: '2.1 写缓冲与读旁路',
  ticks: 7,
  rails: 'lane',
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
    // 不开优化：写一发出就生效，读排在自己的写之后
    '00': {
      hint: '不开优化：写一发出就生效。',
      ops: [
        { id: 'D1', lane: 0, line: 0, tick: 0, label: 'done1=1' },
        { id: 'D2', lane: 1, line: 0, tick: 1, label: 'done2=1' },
        { id: 'R1', lane: 0, line: 1, read: 2, label: 'done2→1' },
        { id: 'R2', lane: 1, line: 1, read: 3, label: 'done1→1' },
      ],
      edges: [
        { from: 'D2', to: 'R1', kind: 'rf' },
        { from: 'D1', to: 'R2', kind: 'rf' },
      ],
      result: { text: 'r1 = 1, r2 = 1', ok: true, why: '两个读都在对方的写生效之后，交接完成' },
    },

    // 写缓冲：写先进队列，发出与生效被拉开，两个读都越过了还在路上的写
    '10': {
      hint: '写缓冲开着：写的发出与生效被拉开，中间的读可以越过。',
      ops: [
        { id: 'D1', lane: 0, line: 0, iss: 0, land: 4, tail: 5, label: 'done1=1' },
        { id: 'D2', lane: 1, line: 0, iss: 1, land: 5, tail: 6, label: 'done2=1' },
        { id: 'R1', lane: 0, line: 1, read: 2, label: 'done2→0', bad: true },
        { id: 'R2', lane: 1, line: 1, read: 3, label: 'done1→0', bad: true },
      ],
      edges: [
        { from: 'D2', to: 'R1', kind: 'rev' },
        { from: 'D1', to: 'R2', kind: 'rev' },
      ],
      result: { text: 'r1 = 0, r2 = 0', ok: false, why: '两个读都读到旧值，没有消费者写 empty，P 一直等' },
    },

    // 插一行 fence：写先生效，读才发出
    '11': {
      hint: 'fence 插在写和读之间：写先生效，读才发出。',
      ops: [
        { id: 'D1', lane: 0, line: 0, iss: 0, land: 2, label: 'done1=1' },
        { id: 'D2', lane: 1, line: 0, iss: 1, land: 3, label: 'done2=1' },
        { id: 'F1', lane: 0, line: 1, fence: 3, label: 'sc' },
        { id: 'F2', lane: 1, line: 1, fence: 4, label: 'sc' },
        { id: 'R1', lane: 0, line: 2, read: 4, label: 'done2→1' },
        { id: 'R2', lane: 1, line: 2, read: 5, label: 'done1→1' },
      ],
      edges: [
        { from: 'D1', to: 'F1', kind: 'keep' },
        { from: 'F1', to: 'R1', kind: 'keep' },
        { from: 'D2', to: 'F2', kind: 'keep' },
        { from: 'F2', to: 'R2', kind: 'keep' },
        { from: 'D2', to: 'R1', kind: 'rf' },
        { from: 'D1', to: 'R2', kind: 'rf' },
      ],
      result: { text: 'r1 = 1, r2 = 1', ok: true, why: 'fence 把写压回读之前，两个读都读到 1' },
    },
  },
};

cases['2.2'] = {
  title: '2.2 写操作重叠',
  ticks: 7,
  rails: 'lane',
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
    // 不开优化：两个写依次生效
    '00': {
      hint: '不开优化：两个写依次生效。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0, label: 'data=42' },
        { id: 'Wf', lane: 0, line: 1, tick: 1, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 2, label: 'full→1' },
        { id: 'Rd', lane: 1, line: 1, read: 3, label: 'data→42' },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: '两个写依次生效，C1 读到那份数据' },
    },

    // 写重叠：两个写同时在途，full 先生效
    '10': {
      hint: '写重叠开着：两个写同时在途，生效的先后可以反过来。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, iss: 0, land: 5, tail: 6, label: 'data=42' },
        { id: 'Wf', lane: 0, line: 1, iss: 1, land: 2, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 3, label: 'full→1' },
        { id: 'Rd', lane: 1, line: 1, read: 4, label: 'data→0', bad: true },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Wf', kind: 'rev' },
      ],
      result: { text: 'r1 = 0', ok: false, why: 'full 先生效，C1 被叫醒时 data 还没到' },
    },

    // 插一行 fence：data 先生效，full 才能发出
    '11': {
      hint: 'fence 插在两个写之间：data 先生效，full 才能发出。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, iss: 0, land: 1, label: 'data=42' },
        { id: 'F', lane: 0, line: 1, fence: 2, label: 'sc' },
        { id: 'Wf', lane: 0, line: 2, iss: 3, land: 4, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 5, label: 'full→1' },
        { id: 'Rd', lane: 1, line: 1, read: 6, label: 'data→42' },
      ],
      edges: [
        { from: 'Wd', to: 'F', kind: 'keep' },
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
  rails: 'lane',
  switches: [
    { key: 'nonBlocking', label: '非阻塞读' },
    { key: 'fence', label: 'fence.sc.gpu', needs: 'nonBlocking' },
  ],
  lanes: [
    { name: 'P', code: ['data = 42', 'full = 1'] },
    { name: 'C1', code: ['while full != 1', 'r1 = data'] },
  ],
  // fence 只插在 C1 的两次读之间
  fences: [{ lane: 1, after: 0, code: 'fence.sc.gpu' }],
  defaults: { nonBlocking: false, fence: false },
  states: {
    // 不开优化：一次读要等值返回才继续
    '00': {
      hint: '不开优化：一次读要等值返回才继续。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0, label: 'data=42' },
        { id: 'Wf', lane: 0, line: 1, tick: 1, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 2, label: 'full→1' },
        { id: 'Rd', lane: 1, line: 1, read: 3, label: 'data→42' },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: '读 full 完成以后才发出读 data，读到那份数据' },
    },

    // 非阻塞读：读 data 提前发出，带回的是旧值
    '10': {
      hint: '非阻塞读开着：读一发出就走，后面的读可以先读到值。',
      ops: [
        { id: 'Rd', lane: 1, line: 1, iss: 0, land: 4, label: 'data→0', bad: true },
        { id: 'Wd', lane: 0, line: 0, tick: 1, label: 'data=42' },
        { id: 'Wf', lane: 0, line: 1, tick: 2, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 3, label: 'full→1' },
      ],
      edges: [
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r1 = 0', ok: false, why: '读 data 在 t1 就走了，full 到 1 时带回的还是 0' },
    },

    // 插一行 fence：第二次读要等第一次读完成
    '11': {
      hint: 'fence 插在两次读之间：第二次读要等第一次读完成。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0, label: 'data=42' },
        { id: 'Wf', lane: 0, line: 1, tick: 1, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 2, label: 'full→1' },
        { id: 'F', lane: 1, line: 1, fence: 3, label: 'sc' },
        { id: 'Rd', lane: 1, line: 2, read: 4, label: 'data→42' },
      ],
      edges: [
        { from: 'Rf', to: 'F', kind: 'keep' },
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
  rails: 'lane',
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
    // 不开优化：读完成以后，才发出后面的写
    '00': {
      hint: '不开优化：读完成以后，才发出后面的写。',
      ops: [
        { id: 'Rd', lane: 0, line: 0, read: 0, label: 'data→42' },
        { id: 'Wd1', lane: 0, line: 1, tick: 1, label: 'done1=1' },
        { id: 'W43', lane: 1, line: 0, tick: 2, label: 'data=43' },
      ],
      edges: [],
      result: { text: 'r1 = 42', ok: true, why: '读在写 done1 之前就读到了这一份数据' },
    },

    // 读与后续写重叠：读还没返回，done1 已经生效，P 写下 43
    '10': {
      hint: '读与后续写重叠开着：读还没返回，后面的写已经生效。',
      ops: [
        { id: 'Rd', lane: 0, line: 0, iss: 0, land: 4, label: 'data→43', bad: true },
        { id: 'Wd1', lane: 0, line: 1, tick: 1, label: 'done1=1' },
        { id: 'W43', lane: 1, line: 0, tick: 2, label: 'data=43' },
      ],
      edges: [
        { from: 'W43', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r1 = 43', ok: false, why: '读 data 在 t1 发出，P 写下 43 之后它才返回' },
    },

    // 插一行 fence：读完成以后，才发出写
    '11': {
      hint: 'fence 插在读和写之间：读完成以后，才发出写。',
      ops: [
        { id: 'Rd', lane: 0, line: 0, read: 0, label: 'data→42' },
        { id: 'F', lane: 0, line: 1, fence: 1, label: 'sc' },
        { id: 'Wd1', lane: 0, line: 2, tick: 2, label: 'done1=1' },
        { id: 'W43', lane: 1, line: 0, tick: 3, label: 'data=43' },
      ],
      edges: [
        { from: 'Rd', to: 'F', kind: 'keep' },
        { from: 'F', to: 'Wd1', kind: 'keep' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence 把写排到读完成之后' },
    },
  },
};

cases['3.2'] = {
  title: '3.2 release 只拦下面的写',
  ticks: 10,
  rails: 'lane',
  switches: [
    { key: 'release', label: 'P 换成 fence.release' },
  ],
  lanes: [
    { name: 'P', code: ['data = 42', 'full = 1', 'while empty != 1'] },
    { name: 'C1', code: ['while full != 1', 'r1 = data'] },
  ],
  // 两边一直各有一行 fence；开关只换 P 那一行
  fences: [
    { lane: 0, after: 0, code: 'fence.sc.gpu', show: true, swap: { key: 'release', code: 'fence.release.gpu' } },
    { lane: 1, after: 0, code: 'fence.sc.gpu', show: true },
  ],
  defaults: { release: false },
  states: {
    // fence.sc：下面的写和读都等 data 生效
    '0': {
      hint: '第 2 节的优化都开着。P、C1 各有一行 fence.sc。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, iss: 0, land: 3, label: 'data=42' },
        { id: 'F', lane: 0, line: 1, fence: 4, label: 'sc' },
        { id: 'Wf', lane: 0, line: 2, iss: 5, land: 6, label: 'full=1' },
        { id: 'Re', lane: 0, line: 3, read: 7, label: 'empty→0' },
        { id: 'Rf', lane: 1, line: 0, read: 7, label: 'full→1' },
        { id: 'F2', lane: 1, line: 1, fence: 8, label: 'sc' },
        { id: 'Rd', lane: 1, line: 2, read: 9, label: 'data→42' },
      ],
      edges: [
        { from: 'Wd', to: 'F', kind: 'keep' },
        { from: 'F', to: 'Wf', kind: 'keep' },
        { from: 'F', to: 'Re', kind: 'keep' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Rf', to: 'F2', kind: 'keep' },
        { from: 'F2', to: 'Rd', kind: 'keep' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence.sc 拦住下面的写和读，读 empty 也等 data 生效' },
    },

    // fence.release：full 还要等 data；读 empty 不受约束，可以先走
    '1': {
      hint: 'P 换成 fence.release：full 还要等 data，读 empty 不用等。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, iss: 0, land: 3, label: 'data=42' },
        { id: 'Re', lane: 0, line: 3, read: 1, label: 'empty→0' },
        { id: 'F', lane: 0, line: 1, fence: 4, label: 'release' },
        { id: 'Wf', lane: 0, line: 2, iss: 5, land: 6, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 7, label: 'full→1' },
        { id: 'F2', lane: 1, line: 1, fence: 8, label: 'sc' },
        { id: 'Rd', lane: 1, line: 2, read: 9, label: 'data→42' },
      ],
      edges: [
        { from: 'Wd', to: 'F', kind: 'keep' },
        { from: 'F', to: 'Wf', kind: 'keep' },
        { from: 'Wd', to: 'Re', kind: 'free' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Rf', to: 'F2', kind: 'keep' },
        { from: 'F2', to: 'Rd', kind: 'keep' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence.release 只拦写：full 等 data，读 empty 先读到值' },
    },
  },
};

cases['3.3'] = {
  title: '3.3 acquire 拦住下面的读',
  ticks: 7,
  rails: 'lane',
  switches: [
    { key: 'fence', label: 'C1 加 fence.sc' },
    { key: 'acquire', label: '换成 fence.acquire', needs: 'fence' },
  ],
  lanes: [
    { name: 'P', code: ['data = 42', 'full = 1'] },
    { name: 'C1', code: ['while full != 1', 'r1 = data'] },
  ],
  // P 一直用 3.2 的 fence.release；C1 的 fence 由开关控制
  fences: [
    { lane: 0, after: 0, code: 'fence.release.gpu', show: true },
    { lane: 1, after: 0, code: 'fence.sc.gpu', swap: { key: 'acquire', code: 'fence.acquire.gpu' } },
  ],
  defaults: { fence: false, acquire: false },
  states: {
    // C1 没有 fence：读 data 提前发出，带回旧值
    '00': {
      hint: '第 2 节的优化都开着。P 有 fence.release，C1 没有 fence。',
      ops: [
        { id: 'Rd', lane: 1, line: 1, iss: 0, land: 5, label: 'data→0', bad: true },
        { id: 'Wd', lane: 0, line: 0, tick: 1, label: 'data=42' },
        { id: 'F', lane: 0, line: 1, fence: 2, label: 'release' },
        { id: 'Wf', lane: 0, line: 2, tick: 3, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 4, label: 'full→1' },
      ],
      edges: [
        { from: 'Wd', to: 'F', kind: 'keep' },
        { from: 'F', to: 'Wf', kind: 'keep' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Wd', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r1 = 0', ok: false, why: 'P 的 release 对 C1 没有作用：读 data 在 t1 就发出了' },
    },

    // C1 加 fence.sc：读 data 要等读 full 完成
    '10': {
      hint: 'C1 加 fence.sc：读 data 要等读 full 完成。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 1, label: 'data=42' },
        { id: 'F', lane: 0, line: 1, fence: 2, label: 'release' },
        { id: 'Wf', lane: 0, line: 2, tick: 3, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 4, label: 'full→1' },
        { id: 'F2', lane: 1, line: 1, fence: 5, label: 'sc' },
        { id: 'Rd', lane: 1, line: 2, read: 6, label: 'data→42' },
      ],
      edges: [
        { from: 'Wd', to: 'F', kind: 'keep' },
        { from: 'F', to: 'Wf', kind: 'keep' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Rf', to: 'F2', kind: 'keep' },
        { from: 'F2', to: 'Rd', kind: 'keep' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence.sc 把读 data 压到读 full 之后' },
    },

    // 换成 fence.acquire：时间线和 fence.sc 一样
    '11': {
      hint: 'C1 换成 fence.acquire：时间线和 fence.sc 一样。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 1, label: 'data=42' },
        { id: 'F', lane: 0, line: 1, fence: 2, label: 'release' },
        { id: 'Wf', lane: 0, line: 2, tick: 3, label: 'full=1' },
        { id: 'Rf', lane: 1, line: 0, read: 4, label: 'full→1' },
        { id: 'F2', lane: 1, line: 1, fence: 5, label: 'acquire' },
        { id: 'Rd', lane: 1, line: 2, read: 6, label: 'data→42' },
      ],
      edges: [
        { from: 'Wd', to: 'F', kind: 'keep' },
        { from: 'F', to: 'Wf', kind: 'keep' },
        { from: 'Wf', to: 'Rf', kind: 'rf' },
        { from: 'Rf', to: 'F2', kind: 'keep' },
        { from: 'F2', to: 'Rd', kind: 'keep' },
        { from: 'Wd', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r1 = 42', ok: true, why: 'fence.acquire 拦住下面的读：读 data 等读 full 完成' },
    },
  },
};

cases['3.4'] = {
  title: '3.4 C2 读完 data 怎样传到 P',
  ticks: 18,
  rails: 'lane',
  switches: [
    { key: 'fence', label: 'C2 的 fence.release' },
  ],
  lanes: [
    { name: 'P', code: ['data = 42', 'fence.release.gpu', 'full = 1', 'while empty != 1:', '    pass',
                        'fence.acquire.gpu', 'data = 43'] },
    { name: 'C1', code: ['while full != 1:', '    pass', 'fence.acquire.gpu', 'value1 = data',
                         'fence.release.gpu', 'done1 = 1', 'fence.sc.gpu', 'if done2 == 1:',
                         '    fence.acq_rel.gpu', '    empty = 1'] },
    { name: 'C2', code: ['while full != 1:', '    pass', 'fence.acquire.gpu', 'value2 = data',
                         'done2 = 1', 'fence.sc.gpu', 'if done1 == 1:',
                         '    fence.acq_rel.gpu', '    empty = 1'] },
  ],
  // 只有 C2 读 data 与写 done2 之间那一行由开关控制
  fences: [{ lane: 2, after: 3, code: 'fence.release.gpu' }],
  defaults: { fence: true },
  states: {
    // 全部 fence：C2 读 data → 写 done2 → C1 读到 → 写 empty → P 读到 → 写 43
    '1': {
      hint: '只画接力的五条边。C2 先读完、先写 done2；C1 最后一个，由它写 empty。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0, label: 'data=42' },
        { id: 'Fr', lane: 0, line: 1, fence: 1, label: 'release' },
        { id: 'Wf', lane: 0, line: 2, tick: 2, label: 'full=1' },
        { id: 'Re', lane: 0, line: 3, read: 15, label: 'empty→1' },
        { id: 'Fa', lane: 0, line: 5, fence: 16, label: 'acquire' },
        { id: 'W43', lane: 0, line: 6, tick: 17, label: 'data=43' },

        { id: 'Rf1', lane: 1, line: 0, read: 4, label: 'full→1' },
        { id: 'Fa1', lane: 1, line: 2, fence: 5, label: 'acquire' },
        { id: 'Rd1', lane: 1, line: 3, read: 6, label: 'data→42' },
        { id: 'Fr1', lane: 1, line: 4, fence: 7, label: 'release' },
        { id: 'Wn1', lane: 1, line: 5, iss: 8, land: 10, label: 'done1=1' },
        { id: 'Fs1', lane: 1, line: 6, fence: 11, label: 'sc' },
        { id: 'Rn2', lane: 1, line: 7, read: 12, label: 'done2→1' },
        { id: 'Far1', lane: 1, line: 8, fence: 13, label: 'acq_rel' },
        { id: 'We1', lane: 1, line: 9, tick: 14, label: 'empty=1' },

        { id: 'Rf2', lane: 2, line: 0, read: 3, label: 'full→1' },
        { id: 'Fa2', lane: 2, line: 2, fence: 4, label: 'acquire' },
        { id: 'Rd2', lane: 2, line: 3, read: 5, label: 'data→42' },
        { id: 'Fr2', lane: 2, line: 4, fence: 6, label: 'release' },
        { id: 'Wn2', lane: 2, line: 5, tick: 7, label: 'done2=1' },
        { id: 'Fs2', lane: 2, line: 6, fence: 8, label: 'sc' },
        { id: 'Rn1', lane: 2, line: 7, read: 9, label: 'done1→0' },
      ],
      edges: [
        { from: 'Rd2', to: 'Wn2', kind: 'keep' },
        { from: 'Wn2', to: 'Rn2', kind: 'rf' },
        { from: 'Rn2', to: 'We1', kind: 'keep' },
        { from: 'We1', to: 'Re', kind: 'rf' },
        { from: 'Re', to: 'W43', kind: 'keep' },
      ],
      result: { text: 'value2 = 42', ok: true, why: 'C2 读 data 排在 P 写 43 之前' },
    },

    // 去掉 C2 的 fence.release：done2 在读 data 返回之前就生效，链的第一环断开
    '0': {
      hint: '去掉 C2 的 fence.release：读 data 还没返回，done2 已经生效。',
      ops: [
        { id: 'Wd', lane: 0, line: 0, tick: 0, label: 'data=42' },
        { id: 'Fr', lane: 0, line: 1, fence: 1, label: 'release' },
        { id: 'Wf', lane: 0, line: 2, tick: 2, label: 'full=1' },
        { id: 'Re', lane: 0, line: 3, read: 14, label: 'empty→1' },
        { id: 'Fa', lane: 0, line: 5, fence: 15, label: 'acquire' },
        { id: 'W43', lane: 0, line: 6, tick: 16, label: 'data=43' },

        { id: 'Rf1', lane: 1, line: 0, read: 4, label: 'full→1' },
        { id: 'Fa1', lane: 1, line: 2, fence: 5, label: 'acquire' },
        { id: 'Rd1', lane: 1, line: 3, read: 6, label: 'data→42' },
        { id: 'Fr1', lane: 1, line: 4, fence: 7, label: 'release' },
        { id: 'Wn1', lane: 1, line: 5, iss: 8, land: 9, label: 'done1=1' },
        { id: 'Fs1', lane: 1, line: 6, fence: 10, label: 'sc' },
        { id: 'Rn2', lane: 1, line: 7, read: 11, label: 'done2→1' },
        { id: 'Far1', lane: 1, line: 8, fence: 12, label: 'acq_rel' },
        { id: 'We1', lane: 1, line: 9, tick: 13, label: 'empty=1' },

        { id: 'Rf2', lane: 2, line: 0, read: 3, label: 'full→1' },
        { id: 'Fa2', lane: 2, line: 2, fence: 4, label: 'acquire' },
        { id: 'Rd2', lane: 2, line: 3, iss: 5, land: 17, label: 'data→43', bad: true },
        { id: 'Wn2', lane: 2, line: 4, tick: 6, label: 'done2=1' },
        { id: 'Fs2', lane: 2, line: 5, fence: 7, label: 'sc' },
        { id: 'Rn1', lane: 2, line: 6, read: 8, label: 'done1→0' },
      ],
      edges: [
        { from: 'Rd2', to: 'Wn2', kind: 'rev' },
        { from: 'Wn2', to: 'Rn2', kind: 'rf' },
        { from: 'Rn2', to: 'We1', kind: 'keep' },
        { from: 'We1', to: 'Re', kind: 'rf' },
        { from: 'Re', to: 'W43', kind: 'keep' },
        { from: 'W43', to: 'Rd2', kind: 'rf' },
      ],
      result: { text: 'value2 = 43', ok: false, why: 'done2 先生效，P 写下 43 时 C2 还没读完' },
    },
  },
};

cases['3.6'] = {
  title: '3.6 C1、C2 谁最后完成',
  ticks: 8,
  rails: 'lane',
  switches: [
    { key: 'sc', label: '换成 fence.sc' },
  ],
  lanes: [
    { name: 'C1', code: ['done1 = 1', 'r1 = done2'] },
    { name: 'C2', code: ['done2 = 1', 'r2 = done1'] },
  ],
  // 两边各一行 fence，开关把 acq_rel 换成 sc
  fences: [{ after: 0, code: 'fence.acq_rel.gpu', show: true, swap: { key: 'sc', code: 'fence.sc.gpu' } }],
  defaults: { sc: false },
  states: {
    // 只用 fence.acq_rel：谁都没读到对方的写，交接接不上
    '0': {
      hint: '第 2 节的优化全部打开。C1、C2 写完自己的 done，隔一行 fence.acq_rel，再读对方的 done。',
      ops: [
        { id: 'D1', lane: 0, line: 0, iss: 0, land: 4, tail: 5, label: 'done1=1' },
        { id: 'F1', lane: 0, line: 1, fence: 1, label: 'acq_rel' },
        { id: 'R1', lane: 0, line: 2, read: 2, label: 'done2→0', bad: true },
        { id: 'D2', lane: 1, line: 0, iss: 1, land: 5, tail: 6, label: 'done2=1' },
        { id: 'F2', lane: 1, line: 1, fence: 2, label: 'acq_rel' },
        { id: 'R2', lane: 1, line: 2, read: 3, label: 'done1→0', bad: true },
      ],
      edges: [
        { from: 'D1', to: 'R1', kind: 'free' },
        { from: 'D2', to: 'R2', kind: 'free' },
      ],
      result: { text: 'r1 = 0, r2 = 0', ok: false, why: '两个读都没有读到对方的写，接不上，没有消费者写 empty' },
    },

    // fence.sc：两个 fence.sc 排成一队，C1 的在前，接上 D1 → F1 → F2 → R2
    '1': {
      hint: '换成 fence.sc：两个 fence.sc 排成一队，这里 C1 的在前，接上一条边。',
      ops: [
        { id: 'D1', lane: 0, line: 0, iss: 0, land: 2, label: 'done1=1' },
        { id: 'F1', lane: 0, line: 1, fence: 3, label: 'sc' },
        { id: 'R1', lane: 0, line: 2, read: 4, label: 'done2→0' },
        { id: 'D2', lane: 1, line: 0, iss: 1, land: 5, label: 'done2=1' },
        { id: 'F2', lane: 1, line: 1, fence: 6, label: 'sc' },
        { id: 'R2', lane: 1, line: 2, read: 7, label: 'done1→1' },
      ],
      edges: [
        { from: 'D1', to: 'F1', kind: 'keep' },
        { from: 'F1', to: 'F2', kind: 'order' },
        { from: 'F2', to: 'R2', kind: 'keep' },
        { from: 'D1', to: 'R2', kind: 'rf' },
      ],
      result: { text: 'r1 = 0, r2 = 1', ok: true, why: 'C1 写 done1 排在 C2 读 done1 之前，C2 读到 1，由它写 empty' },
    },
  },
};

cases['4.1'] = {
  title: '4.1 两人各给 count 加 1',
  ticks: 5,
  rails: 'lane',
  switches: [
    { key: 'atom', label: '读写换成 atom.add' },
  ],
  lanes: [
    { name: 'C1', code: ['value1 = data', 'fence.release.gpu', 'old1 = count', 'count = old1 + 1',
                         'if old1 == 1:', '    fence.acq_rel.gpu', '    empty = 1'],
      swap: { key: 'atom', code: ['value1 = data', 'fence.release.gpu', 'old1 = atom.add(count, 1)',
                                  'if old1 == 1:', '    fence.acq_rel.gpu', '    empty = 1'] } },
    { name: 'C2', code: ['value2 = data', 'fence.release.gpu', 'old2 = count', 'count = old2 + 1',
                         'if old2 == 1:', '    fence.acq_rel.gpu', '    empty = 1'],
      swap: { key: 'atom', code: ['value2 = data', 'fence.release.gpu', 'old2 = atom.add(count, 1)',
                                  'if old2 == 1:', '    fence.acq_rel.gpu', '    empty = 1'] } },
  ],
  fences: [],
  defaults: { atom: false },
  states: {
    // 普通读写：理想执行里也会交错，C2 的读插在 C1 的读和写之间；fence 都在，也管不了
    '0': {
      hint: '按 1.2 的理想执行，没有任何优化，fence 都在。只画 count 的读和写。',
      ops: [
        { id: 'R1', lane: 0, line: 2, read: 0, label: 'count→0', bad: true },
        { id: 'R2', lane: 1, line: 2, read: 1, label: 'count→0', bad: true },
        { id: 'W1', lane: 0, line: 3, tick: 2, label: 'count=1' },
        { id: 'W2', lane: 1, line: 3, tick: 3, label: 'count=1' },
      ],
      edges: [],
      result: { text: 'old1 = 0, old2 = 0, count = 1', ok: false, why: '两个消费者都读到 0，没有消费者写 empty' },
    },

    // atom.add：读、加 1、写回一步完成；fence 不动
    '1': {
      hint: '读写换成 atom.add，fence 不动：读、加 1、写回一步完成，中间插不进别人的写。',
      ops: [
        { id: 'A1', lane: 0, line: 2, tick: 0, label: 'count 0→1' },
        { id: 'A2', lane: 1, line: 2, tick: 2, label: 'count 1→2' },
        { id: 'E2', lane: 1, line: 5, tick: 3, label: 'empty=1' },
      ],
      edges: [
        { from: 'A1', to: 'A2', kind: 'rf' },
      ],
      result: { text: 'old1 = 0, old2 = 1, count = 2', ok: true, why: 'C2 读到旧值 1，是最后一个，由它写 empty' },
    },
  },
};

cases['4.2'] = {
  title: '4.2 C1 写成 atom.cta',
  ticks: 4,
  rails: 'lane',
  switches: [
    { key: 'gpu', label: 'C1 改用 .gpu' },
  ],
  lanes: [
    { name: 'C1（CTA 0）', code: ['old1 = atom.cta.add(count, 1)', 'if old1 == 1:', '    empty = 1'],
      swap: { key: 'gpu', code: ['old1 = atom.gpu.add(count, 1)', 'if old1 == 1:', '    empty = 1'] } },
    { name: 'C2（CTA 1）', code: ['old2 = atom.gpu.add(count, 1)', 'if old2 == 1:', '    empty = 1'] },
  ],
  fences: [],
  defaults: { gpu: false },
  states: {
    // atom.cta 不挡别的 CTA：在 C2 看来，C1 的读和写之间插进了自己的读改写
    '0': {
      hint: 'C1、C2 在不同 CTA。atom.cta 只挡本 CTA 的线程，C2 的读改写插了进来。',
      ops: [
        { id: 'R1', lane: 0, line: 0, read: 0, label: '读 0', bad: true },
        { id: 'A2', lane: 1, line: 0, tick: 1, label: 'count 0→1' },
        { id: 'W1', lane: 0, line: 0, tick: 2, label: '写 1' },
      ],
      edges: [],
      result: { text: 'old1 = 0, old2 = 0, count = 1', ok: false, why: '两个消费者都读到 0，没有消费者写 empty' },
    },

    // 两端 .gpu：互相挡住
    '1': {
      hint: '两端都写 .gpu，互相插不进来。',
      ops: [
        { id: 'A1', lane: 0, line: 0, tick: 0, label: 'count 0→1' },
        { id: 'A2', lane: 1, line: 0, tick: 1, label: 'count 1→2' },
        { id: 'E2', lane: 1, line: 2, tick: 2, label: 'empty=1' },
      ],
      edges: [
        { from: 'A1', to: 'A2', kind: 'rf' },
      ],
      result: { text: 'old1 = 0, old2 = 1, count = 2', ok: true, why: 'C2 读到旧值 1，由它写 empty' },
    },
  },
};

cases['4.4'] = {
  title: '4.4 P 等 count 变成 2',
  ticks: 6,
  rails: 'lane',
  dense: true,
  switches: [
    { key: 'swap', label: 'C2 先加' },
  ],
  lanes: [
    { name: 'C1', width: 1.25, code: ['value1 = data', 'atom.release.add(count, 1)'] },
    { name: 'C2', width: 1.25, code: ['value2 = data', 'atom.release.add(count, 1)'] },
    { name: 'P', code: ['while count != 2:', '    pass', 'data = 43'] },
  ],
  fences: [],
  defaults: { swap: false },
  states: {
    // C1 先加：P 读到的 2 是 C2 写的；C1 的 1 经过 C2 的读改写也算被 P 读到
    '0': {
      hint: 'C1 先加、C2 后加。P 用 ld.acquire 读 count，读到 2 才写 43。',
      ops: [
        { id: 'Rd1', lane: 0, line: 0, read: 0, label: 'data→42' },
        { id: 'A1', lane: 0, line: 1, tick: 1, label: 'count 0→1' },
        { id: 'Rd2', lane: 1, line: 0, read: 1, label: 'data→42' },
        { id: 'A2', lane: 1, line: 1, tick: 2, label: 'count 1→2' },
        { id: 'Rc', lane: 2, line: 0, read: 3, label: 'count→2' },
        { id: 'W43', lane: 2, line: 2, tick: 4, label: 'data=43' },
      ],
      edges: [
        { from: 'Rd1', to: 'A1', kind: 'keep' },
        { from: 'Rd2', to: 'A2', kind: 'keep' },
        { from: 'A1', to: 'A2', kind: 'rf' },
        { from: 'A2', to: 'Rc', kind: 'rf' },
        { from: 'A1', to: 'Rc', kind: 'rf' },
        { from: 'Rc', to: 'W43', kind: 'keep' },
      ],
      result: { text: 'value1 = 42, value2 = 42', ok: true, why: '2 是在 1 的基础上加出来的，P 读到 2，就算读到了 C1 写的 1' },
    },

    // C2 先加：两人对调，P 的代码和结果都不变
    '1': {
      hint: 'C2 先加、C1 后加。P 的代码不变，还是等到 2 才写 43。',
      ops: [
        { id: 'Rd2', lane: 1, line: 0, read: 0, label: 'data→42' },
        { id: 'A2', lane: 1, line: 1, tick: 1, label: 'count 0→1' },
        { id: 'Rd1', lane: 0, line: 0, read: 1, label: 'data→42' },
        { id: 'A1', lane: 0, line: 1, tick: 2, label: 'count 1→2' },
        { id: 'Rc', lane: 2, line: 0, read: 3, label: 'count→2' },
        { id: 'W43', lane: 2, line: 2, tick: 4, label: 'data=43' },
      ],
      edges: [
        { from: 'Rd2', to: 'A2', kind: 'keep' },
        { from: 'Rd1', to: 'A1', kind: 'keep' },
        { from: 'A2', to: 'A1', kind: 'rf' },
        { from: 'A1', to: 'Rc', kind: 'rf' },
        { from: 'A2', to: 'Rc', kind: 'rf' },
        { from: 'Rc', to: 'W43', kind: 'keep' },
      ],
      result: { text: 'value1 = 42, value2 = 42', ok: true, why: '2 是在 1 的基础上加出来的，P 读到 2，就算读到了 C2 写的 1' },
    },
  },
};


cases['5.1'] = {
  title: '5.1 cp.async 搬 data，bar 通知 C',
  ticks: 8,
  rails: 'lane',
  switches: [
    { key: 'wait', label: '加 cp.async.wait_all' },
  ],
  lanes: [
    { name: 'P', code: ['cp.async [data], [src]', 'bar.arrive full'] },
    { name: 'C', code: ['bar.sync full', 'r = data'] },
  ],
  fences: [{ lane: 0, after: 0, code: 'cp.async.wait_all', show: 'wait' }],
  defaults: { wait: false },
  states: {
    // 拷贝不在 P 的程序顺序里：bar.arrive 不等它，C 读早了
    '0': {
      hint: 'cp.async 发出以后，P 不等它做完就往下走。',
      ops: [
        { id: 'Cp', lane: 0, line: 0, iss: 0, land: 5, label: 'data=tile' },
        { id: 'Ar', lane: 0, line: 1, tick: 1, label: 'arrive' },
        { id: 'Sy', lane: 1, line: 0, tick: 2, label: 'sync' },
        { id: 'Rd', lane: 1, line: 1, read: 3, label: 'data→旧值', bad: true },
      ],
      edges: [
        { from: 'Ar', to: 'Sy', kind: 'sync' },
        { from: 'Sy', to: 'Rd', kind: 'keep' },
        { from: 'Cp', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r = 旧值', ok: false, why: '拷贝还没做完，bar 就把 C 放行了' },
    },

    // wait 等拷贝做完，再用 bar 交给 C
    '1': {
      hint: '加 cp.async.wait_all：P 先等自己的拷贝做完，再 bar.arrive。',
      ops: [
        { id: 'Cp', lane: 0, line: 0, iss: 0, land: 2, label: 'data=tile' },
        { id: 'Wt', lane: 0, line: 1, tick: 3, label: 'wait' },
        { id: 'Ar', lane: 0, line: 2, tick: 4, label: 'arrive' },
        { id: 'Sy', lane: 1, line: 0, tick: 5, label: 'sync' },
        { id: 'Rd', lane: 1, line: 1, read: 6, label: 'data→tile' },
      ],
      edges: [
        { from: 'Cp', to: 'Wt', kind: 'keep' },
        { from: 'Wt', to: 'Ar', kind: 'keep' },
        { from: 'Ar', to: 'Sy', kind: 'sync' },
        { from: 'Sy', to: 'Rd', kind: 'keep' },
        { from: 'Cp', to: 'Rd', kind: 'rf' },
      ],
      result: { text: 'r = tile', ok: true, why: 'wait 把拷贝接回 P 的顺序，bar 再交给 C' },
    },
  },
};

cases['5.2'] = {
  title: '5.2 TMA 搬 data',
  ticks: 9,
  rails: 'lane',
  dense: true,
  switches: [
    { key: 'mbar', label: '完成记到 full 上' },
  ],
  lanes: [
    { name: 'P', width: 1.4,
      code: ['cp.async.bulk [data]', 'bar.arrive full'],
      swap: { key: 'mbar', code: ['arrive.expect_tx [full]', 'cp.async.bulk [data], [full]'] } },
    { name: 'TMA', width: 0.7, code: ['（硬件）'] },
    { name: 'C',
      code: ['bar.sync full', 'r = data'],
      swap: { key: 'mbar', code: ['try_wait [full]', 'r = data'] } },
  ],
  fences: [],
  defaults: { mbar: false },
  states: {
    // 照 5.1 写：TMA 没有 wait 可用，P 发出就 arrive，C 读早了
    '0': {
      hint: '照 5.1 写。拷贝由 TMA 去做，P 发出就走，没有 wait 能等它。',
      ops: [
        { id: 'Cb', lane: 0, line: 0, tick: 0, label: '发出' },
        { id: 'Ar', lane: 0, line: 1, tick: 1, label: 'arrive' },
        { id: 'Tw', lane: 1, line: 0, iss: 1, land: 5, label: 'data=tile' },
        { id: 'Sy', lane: 2, line: 0, tick: 2, label: 'sync' },
        { id: 'Rd', lane: 2, line: 1, read: 3, label: 'data→旧值', bad: true },
      ],
      edges: [
        { from: 'Ar', to: 'Sy', kind: 'sync' },
        { from: 'Sy', to: 'Rd', kind: 'keep' },
        { from: 'Tw', to: 'Rd', kind: 'rev' },
      ],
      result: { text: 'r = 旧值', ok: false, why: 'TMA 还没写完，C 已经读了' },
    },

    // TMA 写完把完成记到 full 上；C 用 try_wait 等 full
    '1': {
      hint: 'P 先告诉 full 这一轮要等多少字节，再发出 TMA。TMA 写完，把完成记到 full 上。',
      ops: [
        { id: 'Ex', lane: 0, line: 0, tick: 0, label: 'expect_tx' },
        { id: 'Cb', lane: 0, line: 1, tick: 1, label: '发出' },
        { id: 'Tw', lane: 1, line: 0, iss: 2, land: 5, label: 'data=tile' },
        { id: 'Dn', lane: 1, line: 0, tick: 6, label: 'full 完成' },
        { id: 'Tr', lane: 2, line: 0, read: 7, label: 'full 完成' },
        { id: 'Rd', lane: 2, line: 1, read: 8, label: 'data→tile' },
      ],
      edges: [
        { from: 'Tw', to: 'Dn', kind: 'keep' },
        { from: 'Dn', to: 'Tr', kind: 'rf' },
        { from: 'Tr', to: 'Rd', kind: 'keep' },
      ],
      result: { text: 'r = tile', ok: true, why: 'C 读到 full 这一轮完成以后，才读 data' },
    },
  },
};

cases['5.3'] = {
  title: '5.3 C 读完，P 用 TMA 写下一块',
  ticks: 8,
  rails: 'lane',
  dense: true,
  switches: [
    { key: 'proxy', label: 'C 加 fence.proxy.async' },
  ],
  lanes: [
    // 列宽按实测分：代码最长一行 C 165、P 241；TMA 一栏要放下记号右边的标签，约 100
    { name: 'C', width: 1.0, code: ['try_wait [full]', 'r = data', 'arrive [empty]'] },
    { name: 'P', width: 1.4, code: ['try_wait [empty]', 'cp.async.bulk [data], [full]'] },
    { name: 'TMA', width: 0.6, code: ['（硬件）'] },
  ],
  fences: [{ lane: 0, after: 1, code: 'fence.proxy.async', show: 'proxy' }],
  defaults: { proxy: false },
  states: {
    // 按第 3 节已经排好，但 C 的 ld 和 TMA 的写使用两个通路，排好的先后无效
    '0': {
      hint: 'C 已经等到 full，开始读第一块。按第 3 节的规则，arrive、try_wait 已经排好了先后。',
      ops: [
        { id: 'Ld', lane: 0, line: 1, iss: 0, land: 6, label: 'data→tile1', bad: true },
        { id: 'Ar', lane: 0, line: 2, tick: 1, label: 'arrive' },
        { id: 'Pw', lane: 1, line: 0, read: 2, label: 'empty 完成' },
        { id: 'Cb', lane: 1, line: 1, tick: 3, label: '发出' },
        { id: 'Tw', lane: 2, line: 0, iss: 3, land: 5, label: 'tile1' },
      ],
      edges: [
        { from: 'Ar', to: 'Pw', kind: 'rf' },
        { from: 'Pw', to: 'Cb', kind: 'keep' },
        { from: 'Tw', to: 'Ld', kind: 'rf' },
      ],
      result: { text: 'r = tile1', ok: false, why: 'C 的读和 TMA 的写使用两个通路，排好的先后对它们无效' },
    },

    // proxy fence 切换通路：C 的读排在 arrive 之前，对 TMA 也有效
    '1': {
      hint: 'C 读完 data，先用 fence.proxy.async 切换通路，再 arrive empty。',
      ops: [
        { id: 'Ld', lane: 0, line: 1, iss: 0, land: 1, label: 'data→tile0' },
        { id: 'Fp', lane: 0, line: 2, fence: 2, label: 'proxy' },
        { id: 'Ar', lane: 0, line: 3, tick: 3, label: 'arrive' },
        { id: 'Pw', lane: 1, line: 0, read: 4, label: 'empty 完成' },
        { id: 'Cb', lane: 1, line: 1, tick: 5, label: '发出' },
        { id: 'Tw', lane: 2, line: 0, iss: 5, land: 7, label: 'tile1' },
      ],
      edges: [
        { from: 'Ld', to: 'Fp', kind: 'keep' },
        { from: 'Fp', to: 'Ar', kind: 'keep' },
        { from: 'Ar', to: 'Pw', kind: 'rf' },
        { from: 'Pw', to: 'Cb', kind: 'keep' },
      ],
      result: { text: 'r = tile0', ok: true, why: '切换通路以后，C 的读排在 TMA 的写之前' },
    },
  },
};

cases['5.4'] = {
  title: '5.4 C2 在另一个 CTA',
  ticks: 8,
  rails: 'lane',
  dense: true,
  switches: [
    { key: 'cluster', label: '两端写 .cluster' },
  ],
  lanes: [
    // empty 只有 A 里的一份，代码里不再写「A 的」
    { name: 'C2（CTA B）', width: 1.25,
      code: ['try_wait [full]', 'r = data', 'fence.proxy.async', 'arrive [empty]'],
      swap: { key: 'cluster', code: ['try_wait [full]', 'r = data', 'fence.proxy.async',
                                     'arrive.release.cluster [empty]'] } },
    { name: 'P（CTA A）', width: 1.3,
      code: ['try_wait [empty]', 'cp.async.bulk …'],
      swap: { key: 'cluster', code: ['try_wait.acquire.cluster [empty]', 'cp.async.bulk …'] } },
    { name: 'TMA', width: 0.45, code: ['（硬件）'] },
  ],
  fences: [],
  defaults: { cluster: false },
  states: {
    // 默认 scope 是 .cta：C2 的 arrive 只对 B，P 的 try_wait 只对 A，接不上
    '0': {
      hint: 'C2 照抄 5.3，arrive 的是 A 里的 empty。arrive、try_wait 不写 scope 时都是 .cta。',
      ops: [
        { id: 'Ld', lane: 0, line: 1, iss: 0, land: 6, label: 'data→tile1', bad: true },
        { id: 'Fp', lane: 0, line: 2, fence: 1, label: 'proxy' },
        { id: 'Ar', lane: 0, line: 3, tick: 2, label: 'arrive' },
        { id: 'Pw', lane: 1, line: 0, read: 3, label: 'empty 完成' },
        { id: 'Cb', lane: 1, line: 1, tick: 4, label: '发出' },
        { id: 'Tw', lane: 2, line: 0, iss: 4, land: 5, label: 'tile1' },
      ],
      edges: [
        { from: 'Ar', to: 'Pw', kind: 'free' },
        { from: 'Pw', to: 'Cb', kind: 'keep' },
        { from: 'Tw', to: 'Ld', kind: 'rf' },
      ],
      result: { text: 'r = tile1', ok: false, why: '两端的 scope 都不包含对方，arrive 和 try_wait 接不上' },
    },

    // 两端 .cluster：接上了
    '1': {
      hint: '两端都写 .cluster，scope 包含对方的线程。',
      ops: [
        { id: 'Ld', lane: 0, line: 1, iss: 0, land: 1, label: 'data→tile0' },
        { id: 'Fp', lane: 0, line: 2, fence: 2, label: 'proxy' },
        { id: 'Ar', lane: 0, line: 3, tick: 3, label: 'arrive' },
        { id: 'Pw', lane: 1, line: 0, read: 4, label: 'empty 完成' },
        { id: 'Cb', lane: 1, line: 1, tick: 5, label: '发出' },
        { id: 'Tw', lane: 2, line: 0, iss: 5, land: 7, label: 'tile1' },
      ],
      edges: [
        { from: 'Ld', to: 'Fp', kind: 'keep' },
        { from: 'Fp', to: 'Ar', kind: 'keep' },
        { from: 'Ar', to: 'Pw', kind: 'rf' },
        { from: 'Pw', to: 'Cb', kind: 'keep' },
      ],
      result: { text: 'r = tile0', ok: true, why: 'C2 的 release 和 P 的 acquire 接上了' },
    },
  },
};

})();
