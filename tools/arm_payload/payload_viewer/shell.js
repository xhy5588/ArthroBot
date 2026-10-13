/* Navigation and progressive disclosure for the offline workspace. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const dialog = $('geometry-dialog');
  $('open-geometry').addEventListener('click', () => dialog.showModal());
  for (const id of ['close-geometry', 'done-geometry']) {
    $(id).addEventListener('click', () => dialog.close());
  }
  dialog.addEventListener('close', () => workspace.stopLengthAnimation());
  function updateGeometryLabel() {
    const lengths = workspace.state.design ? workspace.design.lengths : [150, 150];
    $('geometry-summary').textContent = lengths.join(' + ') + ' mm rods';
  }
  window.addEventListener('arm-model-change', updateGeometryLabel);
  updateGeometryLabel();

  function reveal(target) {
    let node = target;
    while (node) {
      if (node.tagName === 'DETAILS') node.open = true;
      node = node.parentElement;
    }
    requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  }
  document.addEventListener('click', (event) => {
    const anchor = event.target.closest('a[href^="#"]');
    if (!anchor) return;
    const target = document.getElementById(anchor.hash.slice(1));
    if (!target) return;
    reveal(target);
  });
  window.addEventListener('hashchange', () => {
    const target = document.getElementById(location.hash.slice(1));
    if (target) reveal(target);
  });
  // Slider fill (style.css reads --frac, 0..1). Values also change from code, for
  // example the payload animation and pose solving, so every slider is repainted
  // once per frame when its value has moved.
  const painted = new WeakMap();
  function paintSliders() {
    for (const range of document.querySelectorAll('input[type="range"]')) {
      const min = Number(range.min || 0), max = Number(range.max || 100);
      const frac = max > min ? Math.min(1, Math.max(0, (Number(range.value) - min) / (max - min))) : 0;
      if (painted.get(range) !== frac) {
        painted.set(range, frac);
        range.style.setProperty('--frac', String(frac));
      }
    }
    requestAnimationFrame(paintSliders);
  }
  paintSliders();

  if (['#design', '#lengths'].includes(location.hash)) dialog.showModal();
  if (location.hash) {
    const target = document.getElementById(location.hash.slice(1));
    if (target) reveal(target);
  }
}());
