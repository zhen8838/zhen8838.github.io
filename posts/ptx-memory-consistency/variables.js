/* A shared variable palette for article code and the supplied SVG timelines. */
(function () {
  'use strict';
  const SVG = 'http://www.w3.org/2000/svg';
  const variables = /\b(data|full|done1|done2|empty)\b/g;

  function colorize(root) {
    if (!root) return;
    for (const target of root.querySelectorAll('code, .tl-code, .tl-val, .cr-src')) {
      const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (walker.nextNode()) {
        const text = walker.currentNode;
        if (!text.parentElement.closest('.ptx-var')) nodes.push(text);
      }
      for (const text of nodes) {
        variables.lastIndex = 0;
        if (!variables.test(text.data)) continue;
        variables.lastIndex = 0;
        const fragment = document.createDocumentFragment();
        let start = 0;
        for (const match of text.data.matchAll(variables)) {
          fragment.append(document.createTextNode(text.data.slice(start, match.index)));
          const span = target.namespaceURI === SVG ? document.createElementNS(SVG, 'tspan') : document.createElement('span');
          span.setAttribute('class', 'ptx-var ptx-v-' + match[0]);
          span.textContent = match[0];
          fragment.append(span);
          start = match.index + match[0].length;
        }
        fragment.append(document.createTextNode(text.data.slice(start)));
        text.replaceWith(fragment);
      }
    }
    // In the introductory timeline, event diamonds use the same variable colors.
    for (const label of root.querySelectorAll('.ptx-intro-card .tl-val')) {
      const match = label.textContent.match(/^(data|full|done1|done2|empty)/);
      const marker = label.previousElementSibling;
      if (match && marker && marker.classList.contains('tl-dia')) {
        marker.classList.add('ptx-var', 'ptx-v-' + match[0]);
      }
    }
  }
  function enhance(card) {
    colorize(card);
    if (card.dataset.variablesObserved) return;
    card.dataset.variablesObserved = 'true';
    const options = { childList: true, subtree: true, characterData: true };
    const observer = new MutationObserver(() => {
      observer.disconnect();
      colorize(card);
      observer.observe(card, options);
    });
    observer.observe(card, options);
  }
  window.PTXVariables = { colorize, enhance };
  function init() { colorize(document.getElementById('quarto-document-content')); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
