(function () {
'use strict';

const scenarios = {};
function register(spec) { scenarios[spec.title.match(/^\d+\.\d+/)[0]] = spec; }
const SC = (a, b, at) => ({ between: [a, b], label: '.sc', on: [false, true], at });

register({
  title: '2.1 写缓冲与读旁路（加 fence 后即 3.6）',
  ticks: 7,
  lanes: [
    { name: 'C1', ops: [{ id: 'D1', code: 'done1=1' }, { id: 'R1', code: 'r1=done2' }] },
    { name: 'C2', ops: [{ id: 'D2', code: 'done2=1' }, { id: 'R2', code: 'r2=done1' }] },
  ],
  fences: [SC('D1', 'R1', [null, 3]), SC('D2', 'R2', [null, 6])],
  check: ['R1', 'R2'],
  states: [
    { at: { D1: [1, 6], R1: [1, 3], D2: [2, 7], R2: [2, 4] }, val: { R1: '0', R2: '0' },
      result: { text: 'r1 = 0, r2 = 0', ok: false, why: '两个读都越过了自己的写' } },
    { at: { D1: [1, 2], R1: [1, 4], D2: [2, 5], R2: [2, 7] }, val: { R1: '0', R2: '1' },
      result: { text: 'r1 = 0, r2 = 1', ok: true, why: 'fence.sc 保住 W→R，两个 ▲ 总有先后' } },
  ],
  edges: [
    { from: 'D1', to: 'R2', kind: ['none', 'rf'] },
    { from: 'D1', to: 'R1', kind: ['rev', 'keep'] },
    { from: 'D2', to: 'R2', kind: ['rev', 'keep'] },
  ],
  toggle: ['两边加 fence.sc', '去掉 fence.sc'],
});

register({
  title: '2.2 写操作重叠',
  ticks: 9,
  lanes: [
    { name: 'P', ops: [{ id: 'Wd', code: 'data=42' }, { id: 'Wf', code: 'full=1' }] },
    { name: 'C1', ops: [{ id: 'Rf', code: 'r1=full' }, { id: 'Rd', code: 'r2=data' }] },
  ],
  fences: [SC('Wd', 'Wf')],
  check: 'Rd',
  states: [
    { at: { Wd: [1, 8, 9], Wf: [2, 3, 9], Rf: [4, 5], Rd: [6, 7] }, val: { Rf: '1', Rd: '0' },
      result: { text: 'r2 = 0', ok: false, why: 'full=1 先于 data=42 完成' } },
    { at: { Wd: [1, 4, 5], Wf: [2, 5, 9], Rf: [4, 6], Rd: [7, 8] }, val: { Rf: '1', Rd: '42' },
      result: { text: 'r2 = 42', ok: true, why: 'fence.sc 保住 W→W' } },
  ],
  edges: [
    { from: 'Wf', to: 'Rf', kind: 'rf' },
    { from: 'Wd', to: 'Wf', kind: ['rev', 'keep'] },
  ],
  toggle: ['P 加 fence.sc', 'P 去掉 fence.sc'],
});

register({
  title: '2.3 非阻塞读',
  ticks: 9,
  lanes: [
    { name: 'P', ops: [{ id: 'Wd', code: 'data=42' }, { id: 'Wf', code: 'full=1' }] },
    { name: 'C1', ops: [{ id: 'Rf', code: 'r1=full' }, { id: 'Rd', code: 'r2=data' }] },
  ],
  fences: [{ between: ['Wd', 'Wf'], label: '.sc' }, SC('Rf', 'Rd')],
  check: 'Rd',
  states: [
    { at: { Wd: [1, 4, 5], Wf: [2, 5, 9], Rf: [1, 6], Rd: [2, 3] }, val: { Rf: '1', Rd: '0' },
      result: { text: 'r2 = 0', ok: false, why: 'r2 先于 r1 完成' } },
    { at: { Wd: [1, 4, 5], Wf: [2, 5, 9], Rf: [1, 6], Rd: [2, 7] }, val: { Rf: '1', Rd: '42' },
      result: { text: 'r2 = 42', ok: true, why: 'fence.sc 保住 R→R' } },
  ],
  edges: [
    { from: 'Wf', to: 'Rf', kind: 'rf' },
    { from: 'Wd', to: 'Wf', kind: 'kept' },
    { from: 'Rf', to: 'Rd', kind: ['rev', 'keep'] },
  ],
  toggle: ['C1 加 fence.sc', 'C1 去掉 fence.sc'],
});

register({
  title: '2.4 读与后续写重叠',
  ticks: 8,
  lanes: [
    { name: 'C1', ops: [{ id: 'R1', code: 'r1=data' }, { id: 'D1', code: 'done1=1' }] },
    { name: 'C2', ops: [{ id: 'R2', code: 'r2=done1' }, { id: 'E', code: 'empty=1' }] },
    { name: 'P', ops: [{ id: 'R3', code: 'r3=empty' }, { id: 'W43', code: 'data=43' }] },
  ],
  fences: [SC('R1', 'D1'), SC('R2', 'E'), SC('R3', 'W43')],
  check: 'R1',
  states: [
    { at: { R1: [1, 8], D1: [2, 3], R2: [3, 4], E: [4, 5], R3: [5, 6], W43: [6, 7] },
      val: { R1: '43', R2: '1', R3: '1' },
      result: { text: 'r1 = 43', ok: false, why: 'done1=1 先于 r1 完成' } },
    { at: { R1: [1, 2], D1: [2, 3], R2: [3, 4], E: [4, 5], R3: [5, 6], W43: [6, 7] },
      val: { R1: '42', R2: '1', R3: '1' },
      result: { text: 'r1 = 42', ok: true, why: 'fence.sc 保住三处 R→W' } },
  ],
  edges: [
    { from: 'D1', to: 'R2', kind: 'rf' },
    { from: 'E', to: 'R3', kind: 'rf' },
    { from: 'W43', to: 'R1', kind: ['rf', 'none'] },
    { from: 'R1', to: 'D1', kind: ['rev', 'keep'] },
    { from: 'R2', to: 'E', kind: ['none', 'keep'] },
    { from: 'R3', to: 'W43', kind: ['none', 'keep'] },
  ],
  toggle: ['三个线程加 fence.sc', '去掉 fence.sc'],
});

register({
  title: '3.2 release 不管下面的读',
  ticks: 8,
  lanes: [
    { name: 'P', ops: [{ id: 'Wd', code: 'data=42' }, { id: 'Wf', code: 'full=1' }, { id: 'Re', code: 'r=empty' }] },
  ],
  fences: [{ between: ['Wd', 'Wf'], label: '.release' }],
  states: [
    { at: { Wd: [1, 5, 6], Wf: [2, 6, 8], Re: [3, 4] }, val: { Re: '0' },
      result: { text: 'r = 0', why: 'fence.release 保住 W→W；读 empty 不受约束' } },
  ],
  edges: [
    { from: 'Wd', to: 'Wf', kind: 'keep' },
    { from: 'Wd', to: 'Re', kind: 'free' },
  ],
});

register({
  title: '3.5 等待循环的三种情况',
  ticks: 6,
  lanes: [
    { name: 'P', ops: [{ id: 'Wf', code: 'full=1' }] },
    { name: 'C1 期望', ops: [{ id: 'A', code: 'r=full' }] },
    { name: 'C1 读一次', ops: [{ id: 'B', code: 'r=full' }] },
    { name: 'C1 旧副本', ops: [{ id: 'C', code: 'r=full' }] },
  ],
  states: [
    { at: {
        Wf: [2, 3],
        A: { marks: [[1, 'dia', '0'], [2, 'dia', '0'], [4, 'dia', '1']] },
        B: { marks: [[1, 'dia', '0'], [2, 'reg'], [3, 'reg'], [4, 'reg'], [5, 'reg'], [6, 'reg']] },
        C: { marks: [[1, 'dia', '0'], [2, 'stale', '0'], [3, 'stale', '0'], [4, 'stale', '0'], [5, 'stale', '0'], [6, 'stale', '0']] },
      },
      result: { text: 'C1 读一次 / 旧副本', ok: false, why: '可能一直等下去' } },
  ],
  edges: [{ from: 'Wf', to: 'A', kind: 'rf' }],
});

register({
  title: '4.2 scope 不同，读改写被拆开',
  ticks: 4,
  lanes: [
    { name: 'C1', ops: [{ id: 'A1', code: ['atom.cta', 'atom.gpu'] }] },
    { name: 'C2', ops: [{ id: 'A2', code: 'atom.gpu' }] },
  ],
  states: [
    { at: { A1: { marks: [[1, 'dia', '读0'], [3, 'dia', '写1']], span: [1, 3] }, A2: { marks: [[2, 'dia', '0→1']] } },
      result: { text: 'count = 1', ok: false, why: 'C2 插进了 C1 的读改写' } },
    { at: { A1: { marks: [[1, 'dia', '0→1']] }, A2: { marks: [[2, 'dia', '1→2']] } },
      result: { text: 'count = 2', ok: true, why: '两端 .gpu，读改写不可分割' } },
  ],
  toggle: ['C1 改用 .gpu', 'C1 改回 .cta'],
});

register({
  title: '5.3 proxy：两条访问路径',
  ticks: 7,
  lanes: [
    { name: 'C', ops: [{ id: 'L', code: 'r=buf' }, { id: 'Ar', code: 'arrive' }] },
    { name: 'P', ops: [{ id: 'Tw', code: 'try_wait' }, { id: 'T', code: 'buf=tile1' }] },
  ],
  fences: [{ between: ['L', 'Ar'], label: '.proxy.async', on: [false, true] }],
  check: 'L',
  states: [
    { at: { L: [1, 7], Ar: [3, 3], Tw: [4, 4], T: [5, 6] }, val: { L: 'tile1' },
      result: { text: 'r = tile1', ok: false, why: 'ld 在 TMA 写之后取值' } },
    { at: { L: [1, 2], Ar: [3, 3], Tw: [4, 4], T: [5, 6] }, val: { L: 'tile0' },
      result: { text: 'r = tile0', ok: true, why: 'fence.proxy.async 换路，再由 arrive 交出' } },
  ],
  edges: [
    { from: 'T', to: 'L', kind: ['rf', 'none'] },
    { from: 'L', to: 'Ar', kind: ['rev', 'keep'] },
    { from: 'Ar', to: 'Tw', kind: 'sync' },
    { from: 'Tw', to: 'T', kind: 'kept' },
  ],
  toggle: ['C 加 proxy fence', 'C 去掉 proxy fence'],
});

register({
  title: '6.2 写不是同时对所有线程可见',
  ticks: 4,
  lanes: [
    { name: 'P1', ops: [{ id: 'W', code: 'A=1' }] },
    { name: 'P2', ops: [{ id: 'R2', code: 'r=A' }] },
    { name: 'P3', ops: [{ id: 'R3', code: 'r=A' }] },
  ],
  states: [
    { at: { W: { marks: [[1, 'iss'], [2, 'dia', '→P2'], [4, 'dia', '→P3']], span: [1, 4] }, R2: [3, 3], R3: [3, 3] },
      val: { R2: '1', R3: '0' },
      result: { text: 'P2: 1, P3: 0', why: '同一次写，对 P2、P3 可见的时刻不同' } },
  ],
});

window.PTX_SCENARIOS = scenarios;
})();
