'use strict';

const $ = (id) => document.getElementById(id);
const levels = [0.5, 1, 1.5, 2];
const palette = ['#8c8c8c', '#4c72b0', '#55a868', '#dd8452', '#8172b3'];  // seaborn deep: grey, blue, green, orange, purple
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const datasets = {};
function decode(encoded, Type) {
  const text = atob(encoded),
    bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return new Type(bytes.buffer);
}
for (const [name, source] of Object.entries(PACKED)) {
  datasets[name] = {
    n: source.count,
    xyz: decode(source.xyz, Float32Array),
    capacity: decode(source.capacity, Float64Array),
    angles: decode(source.angles, Float64Array),
    neighbor: decode(source.neighbor, Float32Array),
  };
  PACKED[name] = null;
}

const state = {
  mode: 'any_orientation',
  load: 1,
  yaw: Math.PI - 0.7,
  pitch: 0.35,
  zoom: 1,
  bounds: [-2, 2],
  selected: -1,
  pose: null,
  scope: [],
  visible: [],
  playing: false,
  rotating: false,
  phase: Math.PI / 2,
  lastTime: 0,
  lastLoadUpdate: 0,
  width: 1,
  height: 1,
  scale: 1,
  dirty: true,
  scheduled: false,
  drag: null,
  mostLoaded: -1,
};
let current = datasets[state.mode];
let activeModel = META.model,
  currentVoxel = META.voxel,
  sceneSpan = 1.43,
  sceneCenter = [0, 0, 0.1];
for (const data of Object.values(datasets))
  data.edge = Float32Array.from(data.neighbor, (x) => (x < 0 ? 1 : 0));
const scene = $('scene'),
  cloud = $('cloud'),
  overlay = $('overlay'),
  ctx = overlay.getContext('2d');
const gl = cloud.getContext('webgl', {
  alpha: false,
  antialias: true,
  powerPreference: 'low-power',
});
const fallback = gl ? null : cloud.getContext('2d');
const diagnostics = { renderer: gl ? 'WebGL' : 'Canvas 2D fallback', errors: [], frames: 0 };
let program,
  attributes = {},
  uniforms = {},
  buffers = {};

const vertexSource = `
precision highp float;
attribute vec3 a_position;
attribute float a_capacity;
attribute float a_neighbor;
attribute float a_edge;
uniform vec3 u_center;
uniform float u_span;
uniform vec4 u_camera;
uniform vec2 u_scale;
uniform vec2 u_bounds;
uniform float u_load;
uniform float u_voxel;
uniform float u_size;
uniform float u_surface;
uniform float u_pass;
varying float v_shade;
void main() {
  vec3 centered=a_position-u_center;
  float x = centered.y*u_camera.y-centered.x*u_camera.x;  // right-handed: right x up = toward the viewer
  float d = centered.x*u_camera.y+centered.y*u_camera.x;
  float z = centered.z;
  float y = z*u_camera.z-d*u_camera.w;
  float depth = d*u_camera.z+z*u_camera.w;
  bool inHeight = a_position.z >= u_bounds.x && a_position.z < u_bounds.y;
  bool supports = a_capacity >= u_load;
  bool edge = a_neighbor < u_load || a_position.z-u_bounds.x < u_voxel*1.1 || u_bounds.y-a_position.z < u_voxel*1.1;
  bool show = inHeight && supports && (u_surface < .5 || edge);
  if (u_pass < .5) show = inHeight && !supports && a_edge > .5;
  gl_Position = show ? vec4(x*u_scale.x, y*u_scale.y, -depth/(u_span*2.), 1.) : vec4(4.,4.,4.,1.);
  gl_PointSize = u_size;
  v_shade = .84+.16*clamp(depth+.5,0.,1.);
}`;
const fragmentSource = `
precision mediump float;
uniform vec4 u_color;
varying float v_shade;
void main() {
  float r = length(gl_PointCoord-vec2(.5))*2.;
  if (r > 1.) discard;
  float fade = 1.-smoothstep(.65,1.,r);
  gl_FragColor = vec4(u_color.rgb*v_shade,u_color.a*fade);
}`;

function initGL() {
  function shader(type, text) {
    const s = gl.createShader(type);
    gl.shaderSource(s, text);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  ['position', 'capacity', 'neighbor', 'edge'].forEach((name) => {
    attributes[name] = gl.getAttribLocation(program, 'a_' + name);
    buffers[name] = gl.createBuffer();
  });
  [
    'camera',
    'scale',
    'bounds',
    'load',
    'voxel',
    'size',
    'surface',
    'pass',
    'color',
    'center',
    'span',
  ].forEach((name) => {
    uniforms[name] = gl.getUniformLocation(program, 'u_' + name);
  });
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthFunc(gl.LEQUAL);
}
function upload() {
  if (!gl) return;
  for (const [name, data, width] of [
    ['position', current.xyz, 3],
    ['capacity', new Float32Array(current.capacity), 1],
    ['neighbor', current.neighbor, 1],
    ['edge', current.edge, 1],
  ]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[name]);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(attributes[name]);
    gl.vertexAttribPointer(attributes[name], width, gl.FLOAT, false, 0, 0);
  }
}
if (gl) {
  initGL();
  upload();
}
cloud.addEventListener('webglcontextlost', (event) => {
  event.preventDefault();
  stopAnimation();
  state.rotating = false;
  $('rotate').checked = false;
  $('render-note').textContent = 'Graphics paused. Reload the page to restore the view.';
  diagnostics.errors.push('WebGL context lost');
});

// Rigid transforms and torques use the same URDF chain and masses as the offline calculation.
function multiply(a, b) {
  const c = Array(16).fill(0);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++)
      for (let k = 0; k < 4; k++) c[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return c;
}
function rotate(t, v) {
  return [0, 1, 2].map((i) => t[4 * i] * v[0] + t[4 * i + 1] * v[1] + t[4 * i + 2] * v[2]);
}
function apply(t, v) {
  return rotate(t, v).map((x, i) => x + t[4 * i + 3]);
}
function rotation(axis, q) {
  const [x, y, z] = axis,
    c = Math.cos(q),
    s = Math.sin(q),
    u = 1 - c;
  return [
    c + x * x * u,
    x * y * u - z * s,
    x * z * u + y * s,
    0,
    y * x * u + z * s,
    c + y * y * u,
    y * z * u - x * s,
    0,
    z * x * u - y * s,
    z * y * u + x * s,
    c + z * z * u,
    0,
    0,
    0,
    0,
    1,
  ];
}
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const subtract = (a, b) => a.map((x, i) => x - b[i]);
function kinematics(q, m = activeModel) {
  let t = m.mount.flat();
  const pivots = [],
    axes = [],
    centers = [];
  for (let i = 0; i < 6; i++) {
    t = multiply(t, m.origins[i].flat());
    pivots.push(apply(t, [0, 0, 0]));
    axes.push(rotate(t, m.axes[i]));
    t = multiply(t, rotation(m.axes[i], q[i]));
    centers.push(apply(t, m.centers[i]));
  }
  const tcp = apply(t, m.grasp),
    own = [],
    unit = [];
  for (let i = 0; i < 6; i++) {
    let torque = 0;
    for (let j = i; j < 6; j++)
      torque += dot(cross(subtract(centers[j], pivots[i]), [0, 0, -9.81 * m.masses[j]]), axes[i]);
    own.push(torque);
    unit.push(dot(cross(subtract(tcp, pivots[i]), [0, 0, -9.81]), axes[i]));
  }
  return { pivots, axes, centers, tcp, own, unit, transform: t };
}
function xyz(index) {
  return Array.from(current.xyz.subarray(index * 3, index * 3 + 3));
}
function pointIsSurface(index) {
  const z = current.xyz[index * 3 + 2];
  return (
    current.neighbor[index] < state.load ||
    z - state.bounds[0] < currentVoxel * 1.1 ||
    state.bounds[1] - z < currentVoxel * 1.1
  );
}
function colorAt(load) {
  const i = Math.min(3, Math.floor(load * 2)),
    a = hexRGB(palette[i]),
    b = hexRGB(palette[i + 1]);
  return a.map((v, k) => v + (b[k] - v) * Math.min(1, load * 2 - i));
}
function hexRGB(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
}
const cssRGB = (rgb) => `rgb(${rgb.map((v) => Math.round(v * 255)).join(',')})`;

function refreshScope() {
  const region = $('height').value,
    center = +$('slice').value;
  state.bounds =
    region === 'above'
      ? [0, 3]
      : region === 'slice'
      ? [
          Math.round(center / currentVoxel) * currentVoxel,
          (Math.round(center / currentVoxel) + 1) * currentVoxel,
        ]
      : [-3, 3];
  $('slice-controls').hidden = region !== 'slice';
  const sliceLow = Math.round(center / currentVoxel) * currentVoxel;
  $('slice-value').textContent = `${(sliceLow * 100).toFixed(1)}–${((sliceLow + currentVoxel) * 100).toFixed(1)} cm`;
  state.scope = [];
  for (let i = 0; i < current.n; i++) {
    const z = current.xyz[3 * i + 2];
    if (z >= state.bounds[0] && z < state.bounds[1]) state.scope.push(i);
  }
  if (!state.scope.includes(state.selected)) chooseDefaultPose();
  renderComparisons();
  updateLoad(state.load, false);
}
function chooseDefaultPose() {
  let best = -1,
    score = Infinity;
  for (const i of state.scope) {
    const p = xyz(i),
      s =
        (p[0] - 0.42) ** 2 +
        (p[1] + 0.1) ** 2 +
        (p[2] - 0.12) ** 2 +
        (current.capacity[i] < state.load ? 10 : 0);
    if (s < score) {
      score = s;
      best = i;
    }
  }
  selectPose(best);
}
function selectPose(index) {
  state.selected = index;
  if (index < 0) {
    state.pose = null;
    $('pose-status').textContent = 'No sampled pose in this height region.';
    $('coordinates').innerHTML = '';
    $('angles').innerHTML = '';
    $('pose-capacity').textContent = '—';
    for (let j = 0; j < 6; j++) {
      $('torque-fill-' + j).style.width = '0';
      $('torque-value-' + j).textContent = '—';
    }
    return;
  }
  const q = Array.from(current.angles.subarray(index * 6, index * 6 + 6));
  state.pose = kinematics(q);
  const p = xyz(index),
    cap = current.capacity[index];
  $('coordinates').innerHTML = p
    .map((x, i) => `<span>${['X', 'Y', 'Z'][i]}<b>${(100 * x).toFixed(1)} cm</b></span>`)
    .join('');
  $('angles').innerHTML = q
    .map((q, i) => `<span>J${i + 1} ${((q * 180) / Math.PI).toFixed(1)}°</span>`)
    .join('');
  $('pose-capacity').textContent =
    cap < 0
      ? 'Empty arm overloaded'
      : cap >= 2
      ? 'Supports ≥2 kg'
      : `Up to ${(Math.floor(cap * 100 + 1e-9) / 100).toFixed(2)} kg`;
  $('pose-capacity').title =
    cap < 0
      ? 'This geometric sample exceeds a motor limit with no payload.'
      : `Calculated gravity-only ceiling: ${cap.toFixed(
          2
        )} kg. This is not a structural or dynamic rating.`;
  updateInspector();
  requestDraw();
}
function updateInspector() {
  if (!state.pose) return;
  const { own, unit } = state.pose,
    torques = own.map((v, j) => Math.abs(v + state.load * unit[j]));
  state.mostLoaded = torques.indexOf(Math.max(...torques));
  for (let j = 0; j < 6; j++) {
    const ratio = torques[j] / activeModel.caps[j],
      fill = $('torque-fill-' + j);
    fill.style.width = `${Math.min(100, ratio * 100)}%`;
    fill.style.background = ratio > 1 ? '#c44e52' : ratio > 0.85 ? '#b89a3e' : '#55a868';
    $('torque-value-' + j).textContent = torques[j].toFixed(1);
  }
  const valid = torques.every((value, j) => value <= activeModel.caps[j] + 1e-8);
  $('pose-status').textContent = valid
    ? current.capacity[state.selected] < 0
      ? 'Load balances this pose; empty arm exceeds limits'
      : '✓ Within the motor torque limits'
    : 'Torque limit exceeded at this posture';
  $('pose-status').classList.toggle('exceeded', !valid);
}
function updateLoad(value, manual = true) {
  if (manual) stopAnimation();
  state.load = Math.max(0, Math.min(2, Math.round(value * 100) / 100));
  $('payload').value = state.load;
  $('load-value').textContent = state.load.toFixed(2);
  $('scene-load').textContent = `${state.load.toFixed(2)} kg`;
  const color = cssRGB(colorAt(state.load));
  document.documentElement.style.setProperty('--load-color', color);
  $('active-color').style.background = color;
  $('load-dot').style.background = color;
  let maxRadius = 0,
    minZ = Infinity,
    maxZ = -Infinity;
  state.visible = [];
  for (const i of state.scope)
    if (current.capacity[i] >= state.load) {
      state.visible.push(i);
      const offset = i * 3,
        z = current.xyz[offset + 2];
      maxRadius = Math.max(maxRadius, Math.hypot(current.xyz[offset], current.xyz[offset + 1]));
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
  $('coverage').textContent = state.scope.length
    ? `${((100 * state.visible.length) / state.scope.length).toFixed(1)}%`
    : '—';
  $(
    'count'
  ).textContent = `${state.visible.length.toLocaleString()} of ${state.scope.length.toLocaleString()} sampled cells`;
  $('reach').textContent = state.visible.length ? `${(100 * maxRadius).toFixed(1)} cm` : '—';
  $('span').textContent = state.visible.length
    ? `${Math.round(minZ * 100)} to ${Math.round(maxZ * 100)} cm`
    : '—';
  $('empty').hidden = state.visible.length > 0;
  document
    .querySelectorAll('[data-load]')
    .forEach((button) =>
      button.classList.toggle('active', Math.abs(+button.dataset.load - state.load) < 0.005)
    );
  updateInspector();
  requestDraw();
}

function renderComparisons() {
  const width = currentVoxel,
    rmax = state.design ? 2 : 0.7,
    zmin = state.design ? -2 : -0.6,
    zmax = state.design ? 2 : 0.8,
    nr = Math.ceil(rmax / width),
    nz = Math.ceil((zmax - zmin) / width);
  const envelope = new Float64Array(nr * nz).fill(-Infinity),
    counts = [0, 0, 0, 0];
  for (const i of state.scope) {
    const p = xyz(i),
      x = Math.floor(Math.hypot(p[0], p[1]) / width),
      y = Math.floor((p[2] - zmin) / width);
    if (x >= 0 && x < nr && y >= 0 && y < nz)
      envelope[y * nr + x] = Math.max(envelope[y * nr + x], current.capacity[i]);
    levels.forEach((m, k) => {
      if (current.capacity[i] >= m) counts[k]++;
    });
  }
  levels.forEach((mass, k) => {
    const canvas = $('layer-' + k),
      percent = $('layer-percent-' + k),
      c = canvas.getContext('2d');
    const w = Math.max(90, canvas.clientWidth),
      h = canvas.clientHeight,
      dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    c.scale(dpr, dpr);
    percent.textContent = state.scope.length
      ? `${Math.round((100 * counts[k]) / state.scope.length)}%`
      : '—';
    $('layer-count-' + k).textContent = `${counts[k].toLocaleString()} qualifying cells`;
    const scale = Math.min((w - 22) / rmax, (h - 4) / (zmax - zmin)),
      ox = (w - rmax * scale) / 2,
      oy = 2;
    c.fillStyle = '#e5eaf0';
    for (let y = 0; y < nz; y++)
      for (let x = 0; x < nr; x++)
        if (envelope[y * nr + x] > -Infinity) {
          c.fillRect(
            ox + x * width * scale,
            oy + (nz - y - 1) * width * scale,
            width * scale + 0.2,
            width * scale + 0.2
          );
        }
    c.fillStyle = palette[k + 1];
    for (let y = 0; y < nz; y++)
      for (let x = 0; x < nr; x++)
        if (envelope[y * nr + x] >= mass) {
          c.fillRect(
            ox + x * width * scale,
            oy + (nz - y - 1) * width * scale,
            width * scale + 0.2,
            width * scale + 0.2
          );
        }
    c.strokeStyle = '#8da0b0';
    c.lineWidth = 0.7;
    c.setLineDash([2, 2]);
    c.beginPath();
    c.moveTo(ox - 3, oy + zmax * scale);
    c.lineTo(ox + rmax * scale, oy + zmax * scale);
    c.stroke();
    c.setLineDash([]);
    c.fillStyle = '#344b60';
    c.beginPath();
    c.arc(ox, oy + zmax * scale, 2, 0, 2 * Math.PI);
    c.fill();
  });
}

function project(p) {
  const cy = Math.cos(state.yaw),
    sy = Math.sin(state.yaw),
    cp = Math.cos(state.pitch),
    sp = Math.sin(state.pitch);
  const px = p[0] - sceneCenter[0],
    py = p[1] - sceneCenter[1];
  const x = py * sy - px * cy, // right-handed, as in the shader
    d = px * sy + py * cy,
    z = p[2] - sceneCenter[2];
  return [
    state.width / 2 + x * state.scale,
    state.height / 2 - (z * cp - d * sp) * state.scale,
    d * cp + z * sp,
  ];
}
function line(a, b, color, width = 1) {
  const x = project(a),
    y = project(b);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x[0], x[1]);
  ctx.lineTo(y[0], y[1]);
  ctx.stroke();
}
function ring(radius, z, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 0.7;
  ctx.beginPath();
  for (let k = 0; k <= 80; k++) {
    const a = (k / 80) * Math.PI * 2,
      p = project([radius * Math.cos(a), radius * Math.sin(a), z]);
    if (k === 0) ctx.moveTo(p[0], p[1]);
    else ctx.lineTo(p[0], p[1]);
  }
  ctx.stroke();
}
function label(text, x, y, color = '#7a8190', size = 10) {
  ctx.font = `${size}px system-ui`;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}
function drawOverlay() {
  ctx.clearRect(0, 0, state.width, state.height);
  for (const r of state.design ? [0.5, 1, 1.5] : [0.2, 0.4, 0.6]) ring(r, 0, '#8ea3b633');
  const grid = state.design ? 1.8 : 0.7;
  line([-grid, 0, 0], [grid, 0, 0], '#99aabc45');
  line([0, -grid, 0], [0, grid, 0], '#99aabc45');
  if (state.design && window.drawDesignCube) window.drawDesignCube();
  const base = project([0, 0, 0]);
  ring(0.043, 0, '#4b5260');
  ring(0.043, 0.014, '#7a8190');
  label('Mounting plane · z = 0', base[0] + 10, base[1] + 18, '#7a8190', 9);
  const sx = 40,
    sy = state.height - 51,
    origin = project([0, 0, 0]);
  [
    ['X', [0.12, 0, 0], '#c44e52'],
    ['Y', [0, 0.12, 0], '#55a868'],
    ['Z', [0, 0, 0.12], '#4c72b0'],
  ].forEach(([name, v, color]) => {
    const p = project(v),
      dx = ((p[0] - origin[0]) / state.scale) * 250,
      dy = ((p[1] - origin[1]) / state.scale) * 250;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(sx + dx, sy + dy);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.7;
    ctx.stroke();
    label(name, sx + dx + 3, sy + dy, color, 9);
  });
  const barLength = state.design ? 0.5 : 0.2,
    bar = barLength * state.scale;
  ctx.strokeStyle = '#8fa1b1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(state.width - bar - 25, state.height - 43);
  ctx.lineTo(state.width - 25, state.height - 43);
  ctx.stroke();
  label(
    state.design ? '50 cm' : '20 cm',
    state.width - bar / 2 - 42,
    state.height - 49,
    '#8b9dad',
    9
  );
  if (!state.pose || state.selected < 0) return;
  const pose = state.pose,
    p = project(pose.tcp),
    valid = current.capacity[state.selected] >= state.load;
  const color = valid ? '#2b2f36' : '#c44e52';
  if ($('show-arm').checked) {
    const chain = [[0, 0, 0], ...pose.pivots, pose.tcp];
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let i = 1; i < chain.length; i++) line(chain[i - 1], chain[i], '#fffffff0', 10);
    for (let i = 1; i < chain.length; i++)
      line(
        chain[i - 1],
        chain[i],
        state.design && i === 3 ? '#dd8452' : state.design && i === 5 ? '#4c72b0' : color,
        5.3
      );
    pose.pivots.forEach((v, j) => {
      const q = project(v);
      ctx.beginPath();
      ctx.arc(q[0], q[1], 6.6, 0, 2 * Math.PI);
      ctx.fillStyle = 'white';
      ctx.fill();
      ctx.beginPath();
      ctx.arc(q[0], q[1], 4.3, 0, 2 * Math.PI);
      ctx.fillStyle = j === state.mostLoaded ? '#b89a3e' : color;
      ctx.fill();
    });
    // A schematic jaw crossbar; it does not stand in for collision geometry.
    const side = rotate(pose.transform, [0, 1, 0]);
    const jaws = [-1, 1].map((sign) => pose.tcp.map((v, j) => v + side[j] * 0.025 * sign));
    line(jaws[0], jaws[1], color, 3.5);
  }
  ctx.beginPath();
  ctx.arc(p[0], p[1], 8, 0, 2 * Math.PI);
  ctx.fillStyle = 'white';
  ctx.fill();
  ctx.strokeStyle = valid ? cssRGB(colorAt(state.load)) : '#c44e52';
  ctx.lineWidth = 2.6;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI);
  ctx.fillStyle = ctx.strokeStyle;
  ctx.fill();
  const text = `${state.load.toFixed(2)} kg`,
    lx = Math.max(8, Math.min(state.width - 87, p[0] + 14)),
    ly = Math.max(69, Math.min(state.height - 40, p[1] - 14));
  ctx.fillStyle = '#fffffff2';
  ctx.fillRect(lx - 5, ly - 16, 77, 25);
  label(text, lx, ly, valid ? '#1f2328' : '#c44e52', 12);
}
function resizeCanvas() {
  const w = scene.clientWidth,
    h = scene.clientHeight,
    dpr = Math.min(devicePixelRatio || 1, 2);
  state.width = w;
  state.height = h;
  state.scale = Math.min(w / (sceneSpan * 1.05), h / sceneSpan) * state.zoom;
  for (const canvas of [cloud, overlay])
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return dpr;
}
function drawGL(dpr) {
  gl.viewport(0, 0, cloud.width, cloud.height);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);
  gl.uniform4f(
    uniforms.camera,
    Math.cos(state.yaw),
    Math.sin(state.yaw),
    Math.cos(state.pitch),
    Math.sin(state.pitch)
  );
  gl.uniform3f(uniforms.center, ...sceneCenter);
  gl.uniform1f(uniforms.span, sceneSpan);
  gl.uniform2f(uniforms.scale, (2 * state.scale) / state.width, (2 * state.scale) / state.height);
  gl.uniform2f(uniforms.bounds, ...state.bounds);
  gl.uniform1f(uniforms.load, state.load);
  gl.uniform1f(uniforms.voxel, currentVoxel);
  gl.uniform1f(uniforms.surface, $('surface').checked ? 1 : 0);
  const size = Math.max(2.3, Math.min(13, currentVoxel * state.scale * 0.48)) * dpr;
  if ($('ghost').checked) {
    gl.disable(gl.DEPTH_TEST);
    gl.uniform1f(uniforms.pass, 0);
    gl.uniform1f(uniforms.size, size * 0.62);
    gl.uniform4f(uniforms.color, 0.57, 0.65, 0.73, 0.19);
    gl.drawArrays(gl.POINTS, 0, current.n);
  }
  gl.enable(gl.DEPTH_TEST);
  gl.uniform1f(uniforms.pass, 1);
  gl.uniform1f(uniforms.size, size);
  gl.uniform4f(uniforms.color, ...colorAt(state.load), 0.84);
  gl.drawArrays(gl.POINTS, 0, current.n);
}
function drawFallback(dpr) {
  fallback.setTransform(dpr, 0, 0, dpr, 0, 0);
  fallback.fillStyle = '#ffffff';
  fallback.fillRect(0, 0, state.width, state.height);
  const step = Math.max(1, Math.floor(state.scope.length / 20000));
  for (let k = 0; k < state.scope.length; k += step) {
    const i = state.scope[k],
      supports = current.capacity[i] >= state.load;
    if (!supports && !$('ghost').checked) continue;
    if (supports && $('surface').checked && !pointIsSurface(i)) continue;
    if (!supports && current.edge[i] < 0.5) continue;
    const p = project(xyz(i));
    fallback.fillStyle = supports ? cssRGB(colorAt(state.load)) : '#c9cdd4';
    fallback.globalAlpha = supports ? 0.75 : 0.25;
    fallback.fillRect(p[0] - 1.3, p[1] - 1.3, 2.6, 2.6);
  }
  fallback.globalAlpha = 1;
  $('render-note').textContent = 'Reduced-density preview · all cells used for statistics';
}
function draw() {
  if (!scene.clientWidth || !scene.clientHeight) return;
  const dpr = resizeCanvas();
  if (gl) drawGL(dpr);
  else drawFallback(dpr);
  drawOverlay();
  diagnostics.frames++;
}
function requestDraw() {
  state.dirty = true;
  if (!state.scheduled) {
    state.scheduled = true;
    requestAnimationFrame(frame);
  }
}
function frame(now) {
  state.scheduled = false;
  const delta = state.lastTime ? Math.min(80, now - state.lastTime) : 0;
  state.lastTime = now;
  if (!document.hidden) {
    if (state.playing) {
      state.phase += (delta * 2 * Math.PI) / 22000;
      if (now - state.lastLoadUpdate > 75) {
        state.lastLoadUpdate = now;
        updateLoad(1 - Math.cos(state.phase), false);
      }
    }
    if (state.rotating && !state.drag) {
      state.yaw += delta * 0.0001;
      state.dirty = true;
    }
    if (state.dirty) {
      state.dirty = false;
      draw();
    }
    if (state.playing || state.rotating) requestDraw();
  }
}
function stopAnimation() {
  state.playing = false;
  $('play').setAttribute('aria-pressed', 'false');
  $('play-icon').textContent = '▶';
  $('play-label').textContent = 'Animate payload';
}
function toggleAnimation() {
  if (state.playing) {
    stopAnimation();
    return;
  }
  state.playing = true;
  state.phase = Math.acos(1 - state.load);
  state.lastTime = 0;
  $('play').setAttribute('aria-pressed', 'true');
  $('play-icon').textContent = 'Ⅱ';
  $('play-label').textContent = 'Pause animation';
  requestDraw();
}
function setView(view) {
  state.rotating = false;
  $('rotate').checked = false;
  [state.yaw, state.pitch] =
    view === 'top' ? [Math.PI, Math.PI / 2] : view === 'side' ? [Math.PI, 0] : [Math.PI - 0.7, 0.35];
  document
    .querySelectorAll('[data-view]')
    .forEach((b) => b.classList.toggle('active', b.dataset.view === view));
  requestDraw();
}
function pickPoint(x, y) {
  let best = -1,
    bestDistance = 144,
    bestDepth = -Infinity;
  const pickable = state.design && $('ghost').checked ? state.scope : state.visible;
  for (const i of pickable) {
    if (current.capacity[i] < state.load && current.edge[i] < 0.5) continue;
    if (current.capacity[i] >= state.load && $('surface').checked && !pointIsSurface(i)) continue;
    const p = project(xyz(i)),
      d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
    if (d < bestDistance - 1 || (Math.abs(d - bestDistance) <= 1 && p[2] > bestDepth)) {
      best = i;
      bestDistance = d;
      bestDepth = p[2];
    }
  }
  if (best >= 0) selectPose(best);
  return best;
}

// Controls use the original numerical capacities; rendering never changes the calculation.
$('layers').innerHTML = levels
  .map(
    (m, k) =>
      `<button class="layer-card" data-load="${m}" style="--layer:${
        palette[k + 1]
      }"><div><strong>${m.toFixed(
        1
      )} kg</strong><span class="percent" id="layer-percent-${k}"></span></div><canvas id="layer-${k}" aria-label="Side envelope for ${m} kg"></canvas><small id="layer-count-${k}"></small></button>`
  )
  .join('');
$('torques').innerHTML = Array.from(
  { length: 6 },
  (_, j) =>
    `<div class="torque-row"><span>J${
      j + 1
    }</span><div class="torque-track"><div class="torque-fill" id="torque-fill-${j}"></div></div><span class="torque-number" id="torque-value-${j}">—</span></div>`
).join('');
document
  .querySelectorAll('[data-load]')
  .forEach((b) => (b.onclick = () => updateLoad(+b.dataset.load)));
document
  .querySelectorAll('[data-view]')
  .forEach((b) => (b.onclick = () => setView(b.dataset.view)));
$('payload').oninput = () => updateLoad(+$('payload').value);
$('play').onclick = toggleAnimation;
$('mode').onchange = () => {
  state.mode = $('mode').value;
  current = (state.design ? state.designDatasets : datasets)[state.mode];
  state.selected = -1;
  upload();
  $('scene-direction').textContent =
    state.mode === 'any_orientation'
      ? 'Best found gripper direction'
      : 'Pointing down · within 15°';
  refreshScope();
};
$('height').onchange = refreshScope;
$('slice').oninput = refreshScope;
$('surface').onchange = () => {
  $('render-note').textContent = $('surface').checked
    ? 'Sampled workspace surface'
    : 'All qualifying sampled points';
  requestDraw();
};
$('ghost').onchange = requestDraw;
$('show-arm').onchange = requestDraw;
$('rotate').onchange = () => {
  state.rotating = $('rotate').checked;
  state.lastTime = 0;
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.remove('active'));
  requestDraw();
};
$('reset-camera').onclick = () => {
  state.zoom = 1;
  setView('three');
};
scene.onpointerdown = (e) => {
  if (e.button !== 0 || state.drag) return;
  scene.setPointerCapture(e.pointerId);
  state.drag = { id: e.pointerId, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY };
};
scene.onpointermove = (e) => {
  if (!state.drag || e.pointerId !== state.drag.id) return;
  state.yaw += (e.clientX - state.drag.x) * 0.006;
  state.pitch = Math.max(
    -Math.PI / 2,
    Math.min(Math.PI / 2, state.pitch + (e.clientY - state.drag.y) * 0.006)
  );
  state.drag.x = e.clientX;
  state.drag.y = e.clientY;
  document.querySelectorAll('[data-view]').forEach((b) => b.classList.remove('active'));
  requestDraw();
};
scene.onpointerup = (e) => {
  if (state.drag && e.pointerId !== state.drag.id) return;
  if (state.drag && Math.hypot(e.clientX - state.drag.startX, e.clientY - state.drag.startY) < 5) {
    const r = scene.getBoundingClientRect();
    pickPoint(e.clientX - r.left, e.clientY - r.top);
  }
  state.drag = null;
};
scene.onpointercancel = () => {
  state.drag = null;
};
scene.onlostpointercapture = () => {
  state.drag = null;
};
scene.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    state.zoom = Math.max(0.55, Math.min(3.5, state.zoom * Math.exp(-e.deltaY * 0.001)));
    requestDraw();
  },
  { passive: false }
);
scene.onkeydown = (e) => {
  if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
    e.preventDefault();
    if (e.key === 'ArrowLeft') state.yaw -= 0.1;
    if (e.key === 'ArrowRight') state.yaw += 0.1;
    if (e.key === 'ArrowUp') state.pitch = Math.min(Math.PI / 2, state.pitch + 0.1);
    if (e.key === 'ArrowDown') state.pitch = Math.max(-Math.PI / 2, state.pitch - 0.1);
    requestDraw();
  }
};
document.addEventListener('visibilitychange', () => {
  state.lastTime = 0;
  if (!document.hidden) requestDraw();
});
let resizeTimer;
window.addEventListener('resize', () => {
  requestDraw();
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderComparisons, 100);
});
$(
  'sample-info'
).textContent = `${META.samples.toLocaleString()} joint poses sampled across ±180°; ${
  currentVoxel * 100
} cm spatial cells. Percentages count sampled cells in the current orientation and height region. They are not certified workspace volumes. The retained capacities and joint angles are stored at full numerical precision.`;
refreshScope();
// A small inspection interface also supports checking the browser against the Python model.
window.workspace = {
  state,
  datasets,
  diagnostics,
  kinematics,
  project,
  pickPoint,
  updateLoad,
  selectPose,
  refreshScope,
  draw,
  stopAnimation,
  toggleAnimation,
  setView,
  renderComparisons,
  get current() {
    return current;
  },
  reducedMotion,
};

