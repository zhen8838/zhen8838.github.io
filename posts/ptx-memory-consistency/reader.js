(function () {
  'use strict';
  function init() {
    const lab = document.getElementById('ptx-lab');
    const article = document.getElementById('quarto-document-content');
    const grid = document.getElementById('quarto-content');
    if (!lab || !article || !grid || !window.Timeline) return;
    grid.append(lab);
    const figure = document.getElementById('ptx-lab-figure');
    const header = document.getElementById('quarto-header');
    const headings = [...article.querySelectorAll('section[id] > h2, section[id] > h3')];
    const panels = new Map();
    const mobile = matchMedia('(max-width: 1085.98px)');
    let active = null, scheduled = false;
    // 侧栏只放两种图：第 1 节的理想交接图，以及 3.2、3.3、3.6 的四象限图。
    // 其余小节的图都在正文里（code-run、表格），侧栏不显示。
    function scenarioKey(section) {
      if (section === '1' || section === '1.1' || section === '1.2') return 'intro';
      if (window.PTXQuadrant && window.PTXQuadrant.has(section)) return 'qd:' + section;
      return null;
    }
    function mountFigure(host, key) {
      if (key === 'intro') window.PTXIntro.mount(host);
      else window.PTXQuadrant.mount(host, key.slice(3));
    }
    function show(section) {
      if (section === active) return;
      active = section;
      const key = scenarioKey(section);
      for (const panel of panels.values()) panel.hidden = true;
      lab.dataset.section = section;
      lab.hidden = !key;
      if (!key) return;
      if (!panels.has(key)) {
        const panel = document.createElement('div');
        panel.dataset.scenario = key;
        mountFigure(panel, key);
        figure.append(panel);
        panels.set(key, panel);
      }
      panels.get(key).hidden = false;
    }
    function sync() {
      scheduled = false;
      const visibleHeader = header ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
      // 放在根元素上：侧栏的图按它躲开顶栏
      document.documentElement.style.setProperty('--ptx-header-bottom', Math.ceil(visibleHeader) + 'px');
      const readingLine = Math.min(innerHeight * 0.45, Math.max(180, visibleHeader + 120));
      let current = headings[0];
      for (const heading of headings) {
        if (heading.getBoundingClientRect().top > readingLine) break;
        current = heading;
      }
      if (!current) { show(''); return; }
      const section = current.parentElement.id.replace(/^sec-/, '').replace('-', '.');
      show(section);
    }
    function schedule() {
      if (!scheduled) { scheduled = true; requestAnimationFrame(sync); }
    }
    // 窄屏没有侧栏时，把侧栏的图放进正文里对应的位置
    function mountInline() {
      if (!mobile.matches) return;
      article.querySelectorAll('[data-ptx-inline]').forEach(host => {
        if (host.dataset.mounted) return;
        const key = scenarioKey(host.dataset.ptxInline);
        if (key) mountFigure(host, key);
        host.dataset.mounted = 'true';
      });
    }
    window.addEventListener('scroll', schedule, { passive: true });
    window.addEventListener('resize', schedule, { passive: true });
    window.addEventListener('hashchange', schedule);
    mobile.addEventListener('change', () => { mountInline(); schedule(); });
    if (header) new ResizeObserver(schedule).observe(header);
    mountInline();
    sync();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
