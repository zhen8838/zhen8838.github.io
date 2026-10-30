(function () {
  'use strict';
  function init() {
    const lab = document.getElementById('ptx-lab');
    const article = document.getElementById('quarto-document-content');
    const grid = document.getElementById('quarto-content');
    const scenarios = window.PTX_SCENARIOS;
    if (!lab || !article || !grid || !window.Timeline || !scenarios) return;
    grid.append(lab);
    const figure = document.getElementById('ptx-lab-figure');
    const header = document.getElementById('quarto-header');
    const headings = [...article.querySelectorAll('section[id] > h2, section[id] > h3')];
    const panels = new Map();
    const mobile = matchMedia('(max-width: 1239px)');
    let active = null, scheduled = false;
    function scenarioKey(section) {
      // The ideal handover figure belongs to section 1 as a whole.
      if (section === '1' || section === '1.1' || section === '1.2') return 'intro';
      // 2.x 都在正文里，不再上侧栏。
      if (/^2(\.[1-4])?$/.test(section)) return null;
      // The supplied 2.1 figure also demonstrates 3.6, as its original title says.
      const key = section === '3.6' ? '2.1' : section;
      return scenarios[key] ? key : null;
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
        if (key === 'intro') window.PTXIntro.mount(panel);
        else window.PTXVariables.enhance(Timeline.mount(panel, scenarios[key]));
        figure.append(panel);
        panels.set(key, panel);
      }
      panels.get(key).hidden = false;
    }
    function sync() {
      scheduled = false;
      const visibleHeader = header ? Math.max(0, header.getBoundingClientRect().bottom) : 0;
      lab.style.setProperty('--ptx-header-bottom', Math.ceil(visibleHeader) + 'px');
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
    // [data-timeline] 无论宽窄都内嵌（侧栏已被象限图占用）
    function mountAlways() {
      article.querySelectorAll('[data-timeline]').forEach(host => {
        if (host.dataset.mounted) return;
        const key = host.dataset.timeline;
        if (!scenarios[key]) return;
        host.dataset.mounted = 'true';
        window.PTXVariables.enhance(Timeline.mount(host, scenarios[key]));
      });
    }

    function mountInline() {
      // 正文里始终显示的时间线
      mountAlways();
      if (!mobile.matches) return;
      article.querySelectorAll('[data-ptx-inline]').forEach(host => {
        if (host.dataset.mounted) return;
        const key = scenarioKey(host.dataset.ptxInline);
        if (key === 'intro') window.PTXIntro.mount(host);
        else if (key) window.PTXVariables.enhance(Timeline.mount(host, scenarios[key]));
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
