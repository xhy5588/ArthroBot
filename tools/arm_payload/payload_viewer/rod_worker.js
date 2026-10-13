'use strict';

// Runs off the UI thread. Every retained point has an actual sampled joint pose.
let samples;
function decodeSamples(encoded) {
  const raw = atob(encoded),
    bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Uint16Array(bytes.buffer);
}
function product(a, b, out) {
  for (let i = 0; i < 3; i++) {
    const k = i * 4;
    for (let j = 0; j < 3; j++)
      out[k + j] = a[k] * b[j] + a[k + 1] * b[4 + j] + a[k + 2] * b[8 + j];
    out[k + 3] = a[k] * b[3] + a[k + 1] * b[7] + a[k + 2] * b[11] + a[k + 3];
  }
}
function analyze(model, sampleCount, width, probeIds) {
  const started = performance.now(),
    size = 100,
    half = (size * width) / 2,
    total = size ** 3;
  const best = new Int32Array(total).fill(-1),
    caps = new Float64Array(total).fill(-1);
  const positions = new Float64Array(sampleCount * 3),
    visited = [];
  const origins = model.origins.map((x) => x.flat().slice(0, 12));
  const mount = model.mount.flat().slice(0, 12),
    axes = model.axes,
    centers = model.centers;
  const t = new Float64Array(12),
    next = new Float64Array(12),
    motion = new Float64Array(12);
  const pivots = new Float64Array(18),
    directions = new Float64Array(18),
    com = new Float64Array(18);
  const own = new Float64Array(6),
    unit = new Float64Array(6),
    probes = [];
  const probeSet = new Set(probeIds || []),
    cosDown = Math.cos(Math.PI / 12);
  const cells = {
    any_orientation: { best, caps, visited },
    downward_15deg: {
      best: new Int32Array(total).fill(-1),
      caps: new Float64Array(total).fill(-1),
      visited: [],
    },
  };
  const graspNorm = Math.hypot(...model.grasp);
  for (let n = 0; n < sampleCount; n++) {
    t.set(mount);
    for (let j = 0; j < 6; j++) {
      product(t, origins[j], next);
      t.set(next);
      const k = j * 3,
        [ax, ay, az] = axes[j];
      for (let i = 0; i < 3; i++) {
        pivots[k + i] = t[i * 4 + 3];
        directions[k + i] = t[i * 4] * ax + t[i * 4 + 1] * ay + t[i * 4 + 2] * az;
      }
      const q = ((samples[n * 6 + j] + 0.5) / 65536) * 2 * Math.PI - Math.PI;
      const c = Math.cos(q),
        s = Math.sin(q),
        u = 1 - c;
      motion.set([
        c + ax * ax * u,
        ax * ay * u - az * s,
        ax * az * u + ay * s,
        0,
        ay * ax * u + az * s,
        c + ay * ay * u,
        ay * az * u - ax * s,
        0,
        az * ax * u - ay * s,
        az * ay * u + ax * s,
        c + az * az * u,
        0,
      ]);
      product(t, motion, next);
      t.set(next);
      const center = centers[j];
      for (let i = 0; i < 3; i++)
        com[k + i] =
          t[i * 4] * center[0] + t[i * 4 + 1] * center[1] + t[i * 4 + 2] * center[2] + t[i * 4 + 3];
    }
    const tcp = [0, 1, 2].map(
      (i) =>
        t[i * 4] * model.grasp[0] +
        t[i * 4 + 1] * model.grasp[1] +
        t[i * 4 + 2] * model.grasp[2] +
        t[i * 4 + 3]
    );
    positions.set(tcp, n * 3);
    let capacity = Infinity,
      emptyOK = true;
    for (let j = 0; j < 6; j++) {
      const k = j * 3;
      let torque = 0;
      for (let b = j; b < 6; b++)
        torque +=
          9.81 *
          model.masses[b] *
          ((com[b * 3] - pivots[k]) * directions[k + 1] -
            (com[b * 3 + 1] - pivots[k + 1]) * directions[k]);
      const slope =
        9.81 *
        ((tcp[0] - pivots[k]) * directions[k + 1] - (tcp[1] - pivots[k + 1]) * directions[k]);
      own[j] = torque;
      unit[j] = slope;
      if (Math.abs(torque) > model.caps[j]) emptyOK = false;
      if (Math.abs(slope) > 1e-10)
        capacity = Math.min(
          capacity,
          Math.max((-model.caps[j] - torque) / slope, (model.caps[j] - torque) / slope)
        );
    }
    // Require the same pose to hold the empty arm, so all loads from 0 to its ceiling are feasible.
    if (!emptyOK) capacity = -1;
    const ix = Math.floor((tcp[0] + half) / width),
      iy = Math.floor((tcp[1] + half) / width),
      iz = Math.floor((tcp[2] + half) / width);
    if (ix < 0 || ix >= size || iy < 0 || iy >= size || iz < 0 || iz >= size)
      throw Error('Design sample outside spatial grid');
    const key = (ix * size + iy) * size + iz;
    const down =
      (t[8] * model.grasp[0] + t[9] * model.grasp[1] + t[10] * model.grasp[2]) / graspNorm <=
      -cosDown;
    for (const name of down ? ['any_orientation', 'downward_15deg'] : ['any_orientation']) {
      const data = cells[name];
      if (data.best[key] === -1) {
        data.best[key] = n;
        data.caps[key] = capacity;
        data.visited.push(key);
      } else if (capacity > data.caps[key]) {
        data.best[key] = n;
        data.caps[key] = capacity;
      }
    }
    if (probeSet.has(n))
      probes.push({ index: n, tcp, own: Array.from(own), unit: Array.from(unit), capacity, down });
  }
  const result = {};
  for (const [name, data] of Object.entries(cells)) {
    const count = data.visited.length,
      xyz = new Float32Array(count * 3),
      capacity = new Float64Array(count),
      angles = new Float64Array(count * 6),
      neighbor = new Float32Array(count),
      edge = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const key = data.visited[i],
        n = data.best[key];
      xyz.set(positions.subarray(n * 3, n * 3 + 3), i * 3);
      capacity[i] = data.caps[key];
      for (let j = 0; j < 6; j++)
        angles[i * 6 + j] = ((samples[n * 6 + j] + 0.5) / 65536) * 2 * Math.PI - Math.PI;
      let min = Infinity,
        exposed = false;
      for (const shift of [-size * size, size * size, -size, size, -1, 1]) {
        const adjacent = key + shift;
        if (adjacent < 0 || adjacent >= total || data.best[adjacent] === -1) {
          exposed = true;
          min = -1;
        } else min = Math.min(min, data.caps[adjacent]);
      }
      neighbor[i] = min;
      edge[i] = exposed ? 1 : 0;
    }
    result[name] = { n: count, xyz, capacity, angles, neighbor, edge };
  }
  return { datasets: result, probes, samples: sampleCount, elapsedMs: performance.now() - started };
}
self.onmessage = (event) => {
  const message = event.data;
  try {
    if (message.samples) samples = decodeSamples(message.samples);
    if (!samples) throw Error('No joint samples loaded');
    const result = analyze(message.model, message.count, message.voxel, message.probes);
    const transfers = [];
    for (const data of Object.values(result.datasets))
      for (const field of ['xyz', 'capacity', 'angles', 'neighbor', 'edge'])
        transfers.push(data[field].buffer);
    self.postMessage({ id: message.id, ...result }, transfers);
  } catch (error) {
    self.postMessage({ id: message.id, error: String(error) });
  }
};
