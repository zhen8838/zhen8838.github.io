/* One continuous t1–t13 figure, using the supplied Timeline renderer. */
(function () {
  'use strict';
  const SVG = 'http://www.w3.org/2000/svg';
  function mount(host) {
    const lanes = ['P', 'C1', 'C2'].map(name => ({ name, ops: [{ id: name, code: '' }] }));
    const card = Timeline.mount(host, {
      title: '1 理想的交接', ticks: 13, lanes,
      states: [{
        at: {
          P: { marks: [[1, 'dia', 'data=42'], [2, 'dia', 'full=1'], [12, 'dia', 'empty→1'], [13, 'dia', 'data=43']] },
          C1: { marks: [[3, 'dia', 'full→1'], [4, 'dia', 'data→42'], [9, 'dia', 'done1=1'], [10, 'dia', 'done2→1'], [11, 'dia', 'empty=1']] },
          C2: { marks: [[5, 'dia', 'full→1'], [6, 'dia', 'data→42'], [7, 'dia', 'done2=1'], [8, 'dia', 'done1→0']] },
        },
        result: { text: 'value1 = value2 = 42', why: '两人读完后，P 才写入下一份数据 43' },
      }],
    });
    card.classList.add('ptx-intro-card');
    const svg = card.querySelector('.tl-svg');
    const ticks = [...svg.querySelectorAll('.tl-tick')];
    function element(tag, attrs, text) {
      const node = document.createElementNS(SVG, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text) node.textContent = text;
      svg.append(node);
      return node;
    }
    const width = svg.viewBox.baseVal.width;
    const divider = (Number(ticks[5].getAttribute('y')) + Number(ticks[6].getAttribute('y'))) / 2 - 3.5;
    element('text', { x: width / 2, y: 46, class: 'ptx-phase-label', 'text-anchor': 'middle' }, '交接数据');
    element('line', { x1: 0, x2: width, y1: divider, y2: divider, class: 'ptx-phase-divider' });
    element('rect', { x: width / 2 - 46, y: divider - 9, width: 92, height: 18, class: 'ptx-phase-label-bg' });
    element('text', { x: width / 2, y: divider + 4, class: 'ptx-phase-label', 'text-anchor': 'middle' }, '归还缓冲区');

    const legend = document.createElement('div');
    legend.className = 'ptx-variable-legend';
    for (const name of ['data', 'full', 'done1', 'done2', 'empty']) {
      const label = document.createElement('span');
      label.className = 'ptx-var ptx-v-' + name;
      label.textContent = name;
      legend.append(label);
    }
    const key = document.createElement('div');
    key.className = 'ptx-phase-key';
    key.textContent = '= 写入；→ 读到的值';
    card.append(legend, key);
    window.PTXVariables.enhance(card);
    return card;
  }
  window.PTXIntro = { mount };
})();
