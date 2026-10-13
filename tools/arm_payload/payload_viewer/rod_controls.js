'use strict';

const design = {
  lengths: [150, 150],
  requested: [150, 150],
  serial: 0,
  worker: null,
  pending: null,
  debounce: null,
  touring: false,
  timer: null,
  tourIndex: 0,
  cache: new Map(),
  lastResult: null,
};
state.design = false;
state.designDatasets = datasets;
const originalSampleInfo = $('sample-info').textContent;
const originalComparison = $('comparison-note').textContent;
const workerURL = URL.createObjectURL(
  new Blob([$('rod-worker').textContent], { type: 'application/javascript' })
);

function modelAtLengths(lengths) {
  const model = JSON.parse(JSON.stringify(META.model));
  META.design.rods.forEach((rod, k) => {
    const delta = lengths[k] / 1000 - 0.15,
      i = rod.joint - 1,
      oldMass = META.model.masses[i];
    const mass = oldMass + (rod.rod_mass * delta) / 0.15;
    const moment = META.model.centers[i].map(
      (c, j) =>
        oldMass * c +
        ((rod.rod_mass * delta) / 0.15) * rod.rod_center[j] +
        (0.5 * rod.rod_mass * (1 + delta / 0.15) + rod.distal_mass) * delta * rod.direction[j]
    );
    model.masses[i] = mass;
    model.centers[i] = moment.map((x) => x / mass);
    for (let j = 0; j < 3; j++) model.origins[rod.joint][j][3] += delta * rod.direction[j];
  });
  return model;
}
function lengthInputs(lengths) {
  ['upper', 'forearm'].forEach((name, i) => {
    $(name + '-length').value = lengths[i];
    $(name + '-number').value = lengths[i];
  });
  document.querySelectorAll('[data-length]').forEach((button) =>
    button.classList.toggle(
      'active',
      lengths.every((x) => x === +button.dataset.length)
    )
  );
}
function designCamera(reset = false) {
  if (reset) state.zoom = 1;
  const bound = 0.722 + (design.lengths[0] + design.lengths[1] - 300) / 1000;
  sceneSpan = $('same-scale').checked
    ? 4.15
    : Math.max(1.25, 2.2 * bound, $('cube').checked ? 2.65 : 0);
  sceneCenter = [$('cube').checked ? 0.2 : 0, 0, 0.0491173688];
  requestDraw();
}
function setDesignLabels(enabled) {
  $('design-controls').hidden = !enabled;
  $('design-assumptions').hidden = !enabled;
  $('original-map').classList.toggle('active', !enabled);
  $('design-map').classList.toggle('active', enabled);
  $('original-map').setAttribute('aria-pressed', String(!enabled));
  $('design-map').setAttribute('aria-pressed', String(enabled));
  $('scene-design-note').hidden = !enabled;
  $('ghost-label').textContent = enabled ? 'Geometric reach' : 'Unloaded outline';
  $('ghost-caption').textContent = enabled
    ? 'Geometric reach · torque unchecked'
    : 'Unloaded reach';
  $('coverage-label').textContent = enabled
    ? 'Sampled cells meeting this load'
    : 'Sampled cells meeting this load';
  $('cell-description').textContent = enabled
    ? 'Each point uses the best empty-arm-feasible posture found in its 5 cm cell.'
    : 'Each point uses the best posture found in its 2.5 cm cell.';
  $('comparison-note').textContent = enabled
    ? 'Gray shows sampled geometric reach; color meets each payload. These side envelopes combine azimuths. Collisions are unchecked, and reach in every orientation is not guaranteed.'
    : originalComparison;
  $('slice').min = enabled ? -1.95 : -0.525;
  $('slice').max = enabled ? 1.95 : 0.725;
  $('slice').step = enabled ? 0.05 : 0.025;
  $('slice-controls').querySelector('small').textContent = enabled ? '5 cm thick' : '2.5 cm thick';
  $('sample-info').textContent = enabled
    ? 'Design sampling uses the same 262,144 Sobol joint poses for each length pair, with 5 cm cells. The full joint chain and gravity torques are recalculated. Compare rod lengths within this preview; the original map uses much denser sampling and a collision screen. Gray includes configurations that cannot hold the empty arm; colored points require both empty-arm and selected-load feasibility. An unsampled cell does not prove a location is unreachable.'
    : originalSampleInfo;
}
function busy(value) {
  scene.dataset.busy = String(value);
  $('mode').disabled = value;
  document.querySelector('.metrics').style.opacity = value ? '.4' : '1';
  document.querySelector('.comparison').style.opacity = value ? '.4' : '1';
}
function cancelJob() {
  if (design.worker) {
    design.worker.terminate();
    design.worker = null;
  }
  if (design.pending) {
    design.pending(false);
    design.pending = null;
  }
}
function applyDesign(result, lengths, model) {
  const first = !state.design;
  state.design = true;
  state.designDatasets = result.datasets;
  current = result.datasets[state.mode];
  activeModel = model;
  currentVoxel = META.design.voxel;
  design.lengths = lengths.slice();
  design.lastResult = result;
  setDesignLabels(true);
  lengthInputs(lengths);
  busy(false);
  designCamera(first);
  state.selected = -1;
  state.pose = null;
  upload();
  refreshScope();
  window.dispatchEvent(new Event('arm-model-change'));
  let radius = 0;
  for (let i = 0; i < current.n; i++)
    radius = Math.max(radius, Math.hypot(current.xyz[i * 3], current.xyz[i * 3 + 1]));
  $('scene-design-note').textContent = `${lengths[0]} + ${
    lengths[1]
  } mm rods · Geometric reach ${radius.toFixed(2)} m horizontally · Collisions unchecked`;
  $('design-status').textContent = `${result.samples.toLocaleString()} poses · ${(
    result.elapsedMs / 1000
  ).toFixed(1)} s calculation`;
}
function setDesignLengths(lengths, options = {}) {
  if (!options.tour) stopLengthAnimation();
  stopAnimation();
  clearTimeout(design.debounce);
  cancelJob();
  lengths = lengths.map((x) => Math.max(50, Math.min(750, Math.round(Number(x) / 25) * 25)));
  if (lengths.some((x) => !Number.isFinite(x)))
    return Promise.reject(Error('Rod length must be a number'));
  design.requested = lengths.slice();
  lengthInputs(lengths);
  setDesignLabels(true);
  const count = options.count || META.design.count,
    key = lengths.join(':') + ':' + count;
  if (!options.count && !state.design) updateLoad(2);
  if (design.cache.has(key)) {
    applyDesign(design.cache.get(key), lengths, modelAtLengths(lengths));
    return Promise.resolve(design.cache.get(key));
  }
  const id = ++design.serial,
    model = modelAtLengths(lengths);
  busy(true);
  $('design-status').textContent = `Calculating ${lengths[0]} + ${lengths[1]} mm…`;
  return new Promise((resolve, reject) => {
    design.pending = resolve;
    const worker = new Worker(workerURL);
    design.worker = worker;
    const fail = (message) => {
      worker.terminate();
      design.worker = null;
      design.pending = null;
      busy(false);
      diagnostics.errors.push(message);
      $('design-status').textContent =
        'Calculation failed. Return to the original map or try again.';
      stopLengthAnimation();
      reject(Error(message));
    };
    worker.onerror = (event) => fail(event.message || 'Rod preview worker failed');
    worker.onmessage = (event) => {
      const result = event.data;
      if (result.id !== id) return;
      if (result.error) {
        fail(result.error);
        return;
      }
      worker.terminate();
      design.worker = null;
      design.pending = null;
      design.cache.set(key, result);
      while (design.cache.size > 4) design.cache.delete(design.cache.keys().next().value);
      applyDesign(result, lengths, model);
      resolve(result);
    };
    worker.postMessage({
      id,
      model,
      count,
      voxel: META.design.voxel,
      samples: META.design.samples,
      probes: [0, 1, 42, Math.min(1048, count - 1), count - 1],
    });
  });
}
function showOriginal() {
  stopLengthAnimation();
  stopAnimation();
  clearTimeout(design.debounce);
  cancelJob();
  busy(false);
  state.design = false;
  current = datasets[state.mode];
  activeModel = META.model;
  currentVoxel = META.voxel;
  sceneSpan = 1.43;
  sceneCenter = [0, 0, 0.1];
  state.zoom = 1;
  state.selected = -1;
  state.pose = null;
  setDesignLabels(false);
  upload();
  refreshScope();
  window.dispatchEvent(new Event('arm-model-change'));
}
function queueLength(index, value) {
  stopLengthAnimation();
  stopAnimation();
  const number = Number(value);
  if (!Number.isFinite(number)) return;
  const lengths = design.requested.slice();
  lengths[index] = Math.max(50, Math.min(750, Math.round(number / 25) * 25));
  if ($('equal-rods').checked) lengths[1 - index] = lengths[index];
  design.requested = lengths;
  lengthInputs(lengths);
  busy(true);
  $('design-status').textContent = `Preparing ${lengths[0]} + ${lengths[1]} mm…`;
  cancelJob();
  clearTimeout(design.debounce);
  design.debounce = setTimeout(() => setDesignLengths(lengths).catch(() => {}), 180);
}
function stopLengthAnimation() {
  design.touring = false;
  clearTimeout(design.timer);
  $('animate-length').textContent = '▶ Animate lengths';
  $('animate-length').setAttribute('aria-pressed', 'false');
}
// Each tour has its own token. A paused tour whose job is later cancelled (the
// job resolves with false) must not schedule further steps next to a newer tour.
async function tourStep(token) {
  if (!design.touring || token !== design.tourToken) return;
  const lengths = [100, 150, 250, 350, 500, 650],
    length = lengths[design.tourIndex++ % lengths.length];
  let result;
  try {
    result = await setDesignLengths([length, length], { tour: true });
  } catch (error) {
    return;
  }
  if (result === false || token !== design.tourToken) return;
  if (design.touring) design.timer = setTimeout(() => tourStep(token), 1600);
}
function toggleLengthAnimation() {
  if (design.touring) {
    stopLengthAnimation();
    return;
  }
  stopAnimation();
  design.touring = true;
  design.tourToken = (design.tourToken || 0) + 1;
  design.tourIndex = 0;
  $('equal-rods').checked = true;
  $('same-scale').checked = true;
  $('animate-length').textContent = 'Ⅱ Pause length animation';
  $('animate-length').setAttribute('aria-pressed', 'true');
  tourStep(design.tourToken);
}
window.drawDesignCube = () => {
  if (!$('cube').checked) return;
  const corners = [];
  for (let i = 0; i < 8; i++)
    corners.push([i & 1 ? 1.1 : 0.1, i & 2 ? 0.5 : -0.5, 0.0491173688 + (i & 4 ? 0.5 : -0.5)]);
  ctx.setLineDash([5, 4]);
  for (let i = 0; i < 8; i++)
    for (const bit of [1, 2, 4])
      if (!(i & bit)) line(corners[i], corners[i | bit], '#b89a3ec0', 1.6);
  ctx.setLineDash([]);
  const p = project([1.1, 0.5, 0.5491173688]);
  const lx = Math.min(state.width - 140, Math.max(12, p[0] + 8)),
    ly = Math.max(88, p[1] - 8);
  ctx.fillStyle = '#ffffffed';
  ctx.fillRect(lx - 5, ly - 15, 130, 22);
  label('1 m × 1 m × 1 m', lx, ly, '#8a7430', 12);
  const b = project([0.1, -0.5, -0.4508826312]);
  label(
    '10 cm gap · base at cube mid-height',
    Math.min(state.width - 220, Math.max(12, b[0])),
    Math.min(state.height - 65, b[1] + 16),
    '#8a714c',
    9
  );
};
$('design-map').onclick = () => setDesignLengths(design.requested).catch(() => {});
$('original-map').onclick = showOriginal;
['upper', 'forearm'].forEach((name, index) => {
  $(name + '-length').oninput = () => queueLength(index, $(name + '-length').value);
  $(name + '-number').onchange = () => queueLength(index, $(name + '-number').value);
});
$('equal-rods').onchange = () => {
  if ($('equal-rods').checked) queueLength(0, $('upper-length').value);
};
document
  .querySelectorAll('[data-length]')
  .forEach(
    (button) =>
      (button.onclick = () =>
        setDesignLengths([+button.dataset.length, +button.dataset.length]).catch(() => {}))
  );
$('animate-length').onclick = toggleLengthAnimation;
$('cube').onchange = () => designCamera();
$('same-scale').onchange = () => designCamera(true);
const oldModeChange = $('mode').onchange;
$('mode').onchange = () => {
  oldModeChange();
  if (state.design) {
    let radius = 0;
    for (let i = 0; i < current.n; i++)
      radius = Math.max(radius, Math.hypot(current.xyz[i * 3], current.xyz[i * 3 + 1]));
    $('scene-design-note').textContent = `${design.lengths[0]} + ${
      design.lengths[1]
    } mm rods · Geometric reach ${radius.toFixed(2)} m horizontally · Collisions unchecked`;
  }
};
const oldPlay = $('play').onclick;
$('play').onclick = () => {
  stopLengthAnimation();
  oldPlay();
};
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopLengthAnimation();
});
Object.assign(window.workspace, {
  design,
  modelAtLengths,
  setDesignLengths,
  showOriginal,
  toggleLengthAnimation,
  stopLengthAnimation,
  getModel: () => activeModel,
  getVoxel: () => currentVoxel,
});
if (location.hash === '#design' || location.hash === '#lengths')
  setDesignLengths([150, 150]).catch(() => {});
