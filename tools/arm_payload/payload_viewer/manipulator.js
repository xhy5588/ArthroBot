/* Direct Cartesian/joint manipulation of the original URDF model.
 * Canvas geometry is a schematic, not collision geometry. Failed Cartesian
 * requests keep the last reached pose; they never inherit a target payload rating.
 */
(function () {
  'use strict';
  const solver = window.ArmPoseSolver;
  const $ = (id) => document.getElementById(id);
  const stage = $('arm-stage');
  const canvas = $('arm-canvas');
  const ctx = canvas.getContext('2d');
  const colors = ['#c44e52', '#55a868', '#4c72b0'];  // tool X, Y, Z
  const names = ['X', 'Y', 'Z'];
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const mul = (a, k) => a.map((x) => x * k);
  const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
  const norm = (v) => Math.hypot(...v);
  const column = (R, i) => R.map((row) => row[i]);
  const wrap = (q) => Math.atan2(Math.sin(q), Math.cos(q));
  const deg = (x) => (x * 180) / Math.PI;
  const rad = (x) => (x * Math.PI) / 180;
  const identity = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const state = {
    q: Array(6).fill(0),
    load: 1,
    tool: 'move',
    mode: 'arm',
    yaw: Math.PI - 0.45,
    elevation: 0.55,
    zoom: 1.25,
    center: [0.24, -0.04, 0.07],
    width: 1,
    height: 1,
    scale: 1,
    pose: null,
    targetP: null,
    targetR: null,
    matched: true,
    positionError: 0,
    orientationError: 0,
    interval: null,
    drag: null,
    handles: [],
    scheduled: false,
    solveScheduled: false,
    pending: false,
    lastSolveMs: 0,
    statusReason: '',
    solverResult: null,
  };
  let model = workspace.getModel();
  let requestSerial = 0;

  function rotationFromEuler(roll, pitch, yaw) {
    return solver.multiply3(
      solver.axisRotation([0, 0, 1], yaw),
      solver.multiply3(solver.axisRotation([0, 1, 0], pitch), solver.axisRotation([1, 0, 0], roll))
    );
  }
  function eulerFromRotation(R) {
    const pitch = Math.asin(Math.max(-1, Math.min(1, -R[2][0])));
    if (Math.abs(Math.cos(pitch)) < 1e-7) return [0, pitch, Math.atan2(-R[0][1], R[1][1])];
    return [Math.atan2(R[2][1], R[2][2]), pitch, Math.atan2(R[1][0], R[0][0])];
  }
  function computeErrors() {
    state.positionError = norm(sub(state.targetP, state.pose.tcp));
    state.orientationError = norm(
      solver.rotationLog(solver.multiply3(state.targetR, solver.transpose3(state.pose.R)))
    );
    state.matched = state.positionError <= 0.001 && state.orientationError <= rad(1);
  }
  function syncInputs() {
    names.forEach((name, i) => {
      $('arm-target-' + name.toLowerCase()).value = state.targetP[i].toFixed(4);
    });
    eulerFromRotation(state.targetR).forEach((angle, i) => {
      $('arm-' + ['roll', 'pitch', 'yaw'][i]).value = deg(angle).toFixed(1);
    });
    state.q.forEach((angle, i) => {
      $('arm-joint-' + i).value = deg(wrap(angle));
      $('arm-joint-value-' + i).textContent = deg(wrap(angle)).toFixed(1) + '°';
    });
  }
  function intervalLabel(interval) {
    if (!interval.feasible) return 'No feasible load';
    if (!Number.isFinite(interval.upper)) return 'No gravity bound';
    const digits = interval.upper < 10 ? 3 : 1,
      scale = 10 ** digits;
    const upper = (Math.floor(interval.upper * scale + 1e-9) / scale).toFixed(digits);
    const lower = Math.ceil(interval.lower * 1000 - 1e-9) / 1000;
    return (interval.lower > 1e-8 ? lower.toFixed(3) + '–' : '') + upper + ' kg';
  }
  function updateReadout() {
    const pose = state.pose;
    const interval = solver.capacity(pose.own, pose.unit, model.caps);
    state.interval = interval;
    const total = pose.own.map((x, j) => x + state.load * pose.unit[j]);
    const usage = total.map((x, j) => Math.abs(x) / model.caps[j]);
    const worst = usage.indexOf(Math.max(...usage));
    const validLoad = usage.every((x) => x <= 1 + 1e-9);
    $('arm-capacity').textContent = state.matched ? intervalLabel(interval) : '—';
    $('arm-limiting').textContent =
      state.matched &&
      interval.feasible &&
      Number.isInteger(interval.limitingJoint) &&
      interval.limitingJoint >= 0
        ? 'Joint ' + (interval.limitingJoint + 1)
        : '—';
    $('arm-position-error').textContent = (state.positionError * 1000).toFixed(2) + ' mm';
    $('arm-orientation-error').textContent = deg(state.orientationError).toFixed(2) + '°';
    $('arm-load').value = state.load;
    $('arm-load-value').textContent = state.load.toFixed(2);
    $('arm-model-label').textContent =
      (workspace.state.design ? workspace.design.lengths.join(' + ') : '150 + 150') +
      ' mm rods';
    $('arm-reset').hidden = state.matched;
    const status = $('arm-status');
    status.dataset.state = state.pending
      ? 'warning'
      : !state.matched
      ? 'error'
      : validLoad
      ? 'ok'
      : 'warning';
    status.textContent = state.pending
      ? 'Solving position and orientation…'
      : !state.matched
      ? (state.statusReason || 'Target not reached by the local solver.') +
        ' Arm held at the last reached pose.'
      : validLoad
      ? 'Pose reached · ' + state.load.toFixed(2) + ' kg is within the static motor limits.'
      : 'Pose reached · Joint ' +
        (worst + 1) +
        ' exceeds its torque limit at ' +
        state.load.toFixed(2) +
        ' kg.';
    $('arm-target-note').textContent = !state.matched
      ? 'No payload limit is assigned to the requested target. Last reached pose: ' +
        intervalLabel(interval) +
        ' (gravity only).'
      : interval.feasible && interval.lower > 1e-8
      ? 'This pose needs a counterbalancing payload of at least ' +
        interval.lower.toFixed(3) +
        ' kg; the empty arm exceeds a limit.'
      : 'Limit for this exact reached posture and gripper orientation. Other postures can give different limits. Collisions are unchecked.';
    for (let j = 0; j < 6; j++) {
      $('arm-torque-value-' + j).textContent =
        Math.abs(total[j]).toFixed(2) + ' / ' + model.caps[j].toFixed(1);
      const fill = $('arm-torque-fill-' + j);
      fill.style.width = Math.min(100, usage[j] * 100) + '%';
      fill.style.background = usage[j] > 1 ? '#c44e52' : usage[j] > 0.85 ? '#b89a3e' : '#55a868';
    }
    $('arm-adopt').disabled = workspace.state.selected < 0;
  }
  function setQ(q) {
    requestSerial++;
    state.pending = false;
    state.q = q.map(wrap);
    state.pose = solver.forward(state.q, model);
    state.targetP = state.pose.tcp.slice();
    state.targetR = state.pose.R.map((row) => row.slice());
    state.statusReason = '';
    computeErrors();
    syncInputs();
    updateReadout();
    requestDraw();
    return state.pose;
  }
  function reachableBound() {
    const base = solver.forward(Array(6).fill(0), model).pivots[0];
    const bound = model.origins
      .slice(1)
      .reduce((sum, t) => sum + Math.hypot(t[0][3], t[1][3], t[2][3]), norm(model.grasp));
    return { base, bound };
  }
  function solveTarget(final = false) {
    const bound = reachableBound();
    state.pending = false;
    if (norm(sub(state.targetP, bound.base)) > bound.bound + 1e-6) {
      state.statusReason = 'Target is beyond the arm’s geometric reach bound.';
      state.solverResult = { converged: false, reason: 'reach-bound' };
    } else {
      const started = performance.now();
      const result = solver.solve(state.targetP, state.targetR, state.q, model, {
        maxIterations: final ? 100 : 45,
        maxTimeMs: final ? 90 : 18,
        restarts: final ? 2 : 0,
        positionTolerance: 0.0007,
        orientationTolerance: rad(0.7),
      });
      state.lastSolveMs = performance.now() - started;
      state.solverResult = result;
      if (result.converged) {
        state.q = result.q.map(wrap);
        state.pose = solver.forward(state.q, model);
      }
      state.statusReason =
        'Target not reached by the local solver; try a smaller move or a different joint posture.';
    }
    computeErrors();
    syncInputs();
    updateReadout();
    requestDraw();
    return state.matched;
  }
  function setTarget(position, R, options = {}) {
    if (!position.every(Number.isFinite) || !R.flat().every(Number.isFinite)) return false;
    state.targetP = position.slice();
    state.targetR = R.map((row) => row.slice());
    requestSerial++;
    if (options.immediate !== false) return solveTarget(options.final !== false);
    state.pending = true;
    if (!state.solveScheduled) {
      state.solveScheduled = true;
      const serial = requestSerial;
      requestAnimationFrame(() => {
        state.solveScheduled = false;
        if (state.pending && requestSerial >= serial) solveTarget(false);
      });
    }
    computeErrors();
    syncInputs();
    updateReadout();
    requestDraw();
    return null;
  }
  function setLoad(value) {
    state.load = Math.max(0, Math.min(2, Number(value) || 0));
    updateReadout();
    requestDraw();
  }
  function setTool(tool) {
    state.tool = tool === 'rotate' ? 'rotate' : 'move';
    $('arm-tool-hint').textContent = state.tool === 'rotate'
      ? 'Drag a ring to rotate the gripper in place.'
      : 'Drag an arrow to move the gripper.';
    document.querySelectorAll('[data-arm-tool]').forEach((button) => {
      const active = button.dataset.armTool === state.tool;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    requestDraw();
  }
  function setView(view) {
    [state.yaw, state.elevation] =
      view === 'top'
        ? [Math.PI, Math.PI / 2 - 0.001]
        : view === 'side'
        ? [Math.PI, 0]
        : [Math.PI - 0.45, 0.55];
    state.zoom = 1.25;
    document.querySelectorAll('[data-arm-view]').forEach((button) => {
      const active = button.dataset.armView === view;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    requestDraw();
  }
  function setMode(mode) {
    state.mode = mode === 'workspace' ? 'workspace' : 'arm';
    document.body.dataset.workspaceMode = state.mode;
    $('arm-editor').hidden = state.mode !== 'arm';
    document.querySelectorAll('.map-only').forEach((element) => {
      element.hidden = state.mode !== 'workspace';
    });
    for (const [id, selected] of [
      ['arm-mode', 'arm'],
      ['workspace-mode', 'workspace'],
    ]) {
      $(id).classList.toggle('active', state.mode === selected);
      $(id).setAttribute('aria-pressed', String(state.mode === selected));
    }
    workspace.stopAnimation();
    workspace.stopLengthAnimation();
    workspace.state.rotating = false;
    $('rotate').checked = false;
    if (state.mode === 'workspace') {
      workspace.draw();
      workspace.renderComparisons();
    } else {
      updateReadout();
      requestDraw();
    }
  }

  // Orthographic camera: screen-space dragging can be inverted without ray ambiguity.
  function cameraBasis() {
    const cy = Math.cos(state.yaw),
      sy = Math.sin(state.yaw);
    const cp = Math.cos(state.elevation),
      sp = Math.sin(state.elevation);
    // Right-handed: right × up = depth, the direction toward the viewer.
    return { right: [-cy, sy, 0], up: [-sy * sp, -cy * sp, cp], depth: [sy * cp, cy * cp, sp] };
  }
  function project(point) {
    const v = sub(point, state.center),
      basis = cameraBasis();
    return [
      state.width / 2 + dot(v, basis.right) * state.scale,
      state.height / 2 - dot(v, basis.up) * state.scale,
      dot(v, basis.depth),
    ];
  }
  function screenVector(v) {
    const basis = cameraBasis();
    return [dot(v, basis.right) * state.scale, -dot(v, basis.up) * state.scale];
  }
  function line(a, b, color, width = 1, dash = []) {
    const p = project(a),
      q = project(b);
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function dotAt(point, radius, color, border = '#ffffff') {
    const p = project(point);
    ctx.beginPath();
    ctx.arc(p[0], p[1], radius, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
  function label(point, text, color = '#7a8190', offset = [8, -8]) {
    const p = project(point);
    ctx.font = '11px system-ui';
    ctx.fillStyle = color;
    ctx.fillText(text, p[0] + offset[0], p[1] + offset[1]);
  }
  function circle(center, u, v, radius, color, width = 1, dash = []) {
    ctx.beginPath();
    const points = [];
    for (let i = 0; i <= 80; i++) {
      const a = (i / 80) * Math.PI * 2;
      const p = project(
        add(center, add(mul(u, radius * Math.cos(a)), mul(v, radius * Math.sin(a))))
      );
      if (i) ctx.lineTo(p[0], p[1]);
      else ctx.moveTo(p[0], p[1]);
      points.push(p);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
    return points;
  }
  function arrow(start, vector, color, text, register) {
    const end = add(start, vector),
      p = project(start),
      q = project(end);
    if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 10) return;
    line(start, end, '#ffffffe0', 7);
    line(start, end, color, 3);
    const angle = Math.atan2(q[1] - p[1], q[0] - p[0]);
    ctx.beginPath();
    ctx.moveTo(q[0], q[1]);
    ctx.lineTo(q[0] - 11 * Math.cos(angle - 0.42), q[1] - 11 * Math.sin(angle - 0.42));
    ctx.lineTo(q[0] - 11 * Math.cos(angle + 0.42), q[1] - 11 * Math.sin(angle + 0.42));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    label(end, text, color);
    if (register) state.handles.push({ ...register, start: p, end: q, point: q });
  }
  function gripperAxes(R) {
    const local = model.gripper_visual_axes;
    const toWorld = (v) => R.map((row) => dot(row, v));
    return { forward: toWorld(local.forward), side: toWorld(local.side), top: toWorld(local.top) };
  }
  function drawGripper(pose, color, ghost = false) {
    const { forward, side, top } = gripperAxes(pose.R);
    const back = add(pose.tcp, mul(forward, -0.055));
    const jaws = [-1, 1].map((sign) => ({
      root: add(back, mul(side, 0.033 * sign)),
      tip: add(pose.tcp, add(mul(side, 0.033 * sign), mul(forward, 0.018))),
      pad: add(pose.tcp, mul(side, 0.028 * sign)),
    }));
    const dash = ghost ? [4, 4] : [];
    line(jaws[0].root, jaws[1].root, color, ghost ? 2 : 9, dash);
    jaws.forEach((jaw) => {
      line(jaw.root, jaw.tip, color, ghost ? 2 : 7, dash);
      line(
        add(jaw.pad, mul(forward, -0.012)),
        add(jaw.pad, mul(forward, 0.016)),
        ghost ? color : '#55a868',
        ghost ? 1 : 4,
        dash
      );
      if (!ghost) line(jaw.tip, add(jaw.tip, mul(top, 0.018)), '#2b2f36', 3);
    });
    if (!ghost) line(pose.pivots[5], back, '#7a8190', 8);
  }
  function draw() {
    state.scheduled = false;
    if ($('arm-editor').hidden || !stage.clientWidth || !state.pose) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    state.width = stage.clientWidth;
    state.height = stage.clientHeight;
    const length = workspace.state.design
      ? workspace.design.lengths.reduce((a, b) => a + b) / 1000
      : 0.3;
    const span = Math.max(0.98, 0.68 + length);
    state.scale = Math.min(state.width / (span * 1.28), state.height / span) * state.zoom;
    canvas.width = Math.round(state.width * dpr);
    canvas.height = Math.round(state.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    const background = ctx.createLinearGradient(0, 0, 0, state.height);
    background.addColorStop(0, '#ffffff');
    background.addColorStop(1, '#f6f7f9');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, state.width, state.height);
    state.handles = [];
    for (let i = -5; i <= 7; i++) {
      line([i * 0.1, -0.5, 0], [i * 0.1, 0.5, 0], '#1f232814');
      line([-0.5, i * 0.1, 0], [0.7, i * 0.1, 0], '#1f232814');
    }
    circle([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.055, '#8c8c8c', 6);
    line([0, 0, 0], state.pose.pivots[0], '#2b2f36', 18);
    label([0, 0, 0], 'Fixed base', '#7a8190', [-35, 25]);
    const pose = state.pose;
    const segments = pose.pivots.slice(1).map((point, i) => ({ a: pose.pivots[i], b: point, i }));
    segments.sort(
      (a, b) => project(a.a)[2] + project(a.b)[2] - (project(b.a)[2] + project(b.b)[2])
    );
    segments.forEach(({ a, b, i }) => {
      const color = i === 1 ? '#dd8452' : i === 3 ? '#4c72b0' : '#8c8c8c';
      line(a, b, '#ffffff', 15);
      line(a, b, color, 10);
      line(a, b, '#ffffff40', 3);
    });
    pose.pivots.forEach((point, j) => {
      const axis = pose.axes[j];
      line(add(point, mul(axis, -0.022)), add(point, mul(axis, 0.022)), '#2b2f36', 17);
      dotAt(point, 7, '#ffffff', '#4b5260');
      const p = project(point);
      state.handles.push({ kind: 'joint', joint: j, point: p });
      if ($('arm-joints').closest('details').open) {
        label(point, 'J' + (j + 1), '#4b5260', [-16, -14]);
      }
    });
    drawGripper(pose, '#4b5260');
    if ($('arm-show-module').checked) {
      const axis = gripperAxes(pose.R).top;
      line(add(pose.tcp, mul(axis, -0.25)), add(pose.tcp, mul(axis, 0.25)), '#8172b388', 9);
      label(add(pose.tcp, mul(axis, 0.25)), '50 cm module · clearance unchecked', '#8172b3');
    }
    if (!state.matched && !state.pending) {
      drawGripper({ tcp: state.targetP, R: state.targetR }, '#c44e5299', true);
      line(pose.tcp, state.targetP, '#c44e52', 1.5, [4, 4]);
      dotAt(state.targetP, 5, '#fff7f3', '#c44e52');
      label(state.targetP, 'Requested target', '#c44e52');
    }
    const size = 78 / state.scale;
    if ($('arm-show-frame').checked && state.tool === 'move') {
      for (let j = 0; j < 3; j++)
        arrow(pose.tcp, mul(column(pose.R, j), size * 0.62), colors[j] + 'a0', 't' + names[j]);
    }
    if (state.tool === 'move') {
      for (let j = 0; j < 3; j++)
        arrow(pose.tcp, mul(identity[j], size), colors[j], names[j], {
          kind: 'translate',
          axis: identity[j],
          index: j,
        });
      dotAt(pose.tcp, 9, '#ffffff', '#dd8452');
      state.handles.push({ kind: 'translate-plane', point: project(pose.tcp) });
    } else {
      for (let j = 0; j < 3; j++) {
        const u = column(pose.R, (j + 1) % 3),
          v = column(pose.R, (j + 2) % 3);
        const radius = size * (1 + j * 0.11);
        const points = circle(pose.tcp, u, v, radius, colors[j], 2.7);
        state.handles.push({
          kind: 'rotate',
          axis: column(pose.R, j),
          u,
          v,
          index: j,
          radius,
          points,
        });
        const labelPoint = add(pose.tcp, mul(u, radius));
        label(labelPoint, 't' + names[j], colors[j]);
      }
      dotAt(pose.tcp, 5, '#fbeee6', '#dd8452');
    }
    ctx.fillStyle = '#7a8190';
    ctx.font = '11px system-ui';
    ctx.fillText('10 cm grid · schematic', 18, state.height - 42);
  }
  function requestDraw() {
    if (!state.scheduled) {
      state.scheduled = true;
      requestAnimationFrame(draw);
    }
  }
  function segmentDistance(point, a, b) {
    const v = sub(b.slice(0, 2), a.slice(0, 2)),
      p = sub(point, a.slice(0, 2));
    const t = Math.max(0, Math.min(1, dot(p, v) / Math.max(dot(v, v), 1e-12)));
    return norm(sub(p, mul(v, t)));
  }
  function hitTest(point) {
    let best = null,
      score = 11;
    for (const handle of state.handles) {
      let distance;
      if (handle.kind === 'rotate') {
        let closest = Infinity,
          angle = 0;
        for (let i = 1; i < handle.points.length; i++) {
          const d = segmentDistance(point, handle.points[i - 1], handle.points[i]);
          if (d < closest) {
            closest = d;
            angle = ((i - 0.5) / 80) * Math.PI * 2;
          }
        }
        distance = closest;
        handle.hitAngle = angle;
      } else if (handle.kind === 'translate')
        distance = segmentDistance(point, handle.start, handle.end);
      else
        distance = norm(sub(point, handle.point.slice(0, 2))) + (handle.kind === 'joint' ? 2 : -3);
      if (distance < score) {
        score = distance;
        best = handle;
      }
    }
    return best;
  }
  function pointerPosition(event) {
    const rect = stage.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }
  function angleAt(point, drag) {
    const offset = sub(point, drag.center);
    const a = drag.screenU,
      b = drag.screenV,
      det = a[0] * b[1] - a[1] * b[0];
    if (Math.abs(det) > 0.05 * norm(a) * norm(b)) {
      return Math.atan2(
        (a[0] * offset[1] - a[1] * offset[0]) / det,
        (offset[0] * b[1] - offset[1] * b[0]) / det
      );
    }
    return (
      drag.startAngle +
      dot(sub(point, drag.start), drag.tangent) / Math.max(1, dot(drag.tangent, drag.tangent))
    );
  }
  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || state.drag) return;
    event.preventDefault();
    stage.focus({ preventScroll: true });
    const point = pointerPosition(event),
      handle = hitTest(point);
    state.drag = {
      ...handle,
      kind: handle ? handle.kind : 'orbit',
      pointerId: event.pointerId,
      start: point,
      previous: point,
      startQ: state.q.slice(),
      startP: state.pose.tcp.slice(),
      startR: state.pose.R.map((row) => row.slice()),
    };
    const drag = state.drag;
    if (handle && handle.kind !== 'joint') {
      state.targetP = drag.startP.slice();
      state.targetR = drag.startR.map((row) => row.slice());
      computeErrors();
    }
    if (drag.kind === 'translate') drag.screenAxis = screenVector(drag.axis);
    if (drag.kind === 'rotate') {
      drag.center = project(state.pose.tcp).slice(0, 2);
      drag.screenU = screenVector(mul(drag.u, drag.radius));
      drag.screenV = screenVector(mul(drag.v, drag.radius));
      drag.startAngle = drag.hitAngle;
      drag.tangent = add(
        mul(drag.screenU, -Math.sin(drag.startAngle)),
        mul(drag.screenV, Math.cos(drag.startAngle))
      );
      drag.lastAngle = angleAt(point, drag);
      drag.angle = 0;
    }
    stage.setPointerCapture(event.pointerId);
    stage.dataset.dragging = 'true';
  });
  stage.addEventListener('pointermove', (event) => {
    const point = pointerPosition(event),
      drag = state.drag;
    if (!drag) {
      stage.style.cursor = hitTest(point) ? 'grab' : 'default';
      return;
    }
    if (event.pointerId !== drag.pointerId) return;
    const delta = sub(point, drag.start);
    if (drag.kind === 'orbit') {
      state.yaw += (point[0] - drag.previous[0]) * 0.006;
      state.elevation = Math.max(
        -1.35,
        Math.min(1.5, state.elevation + (point[1] - drag.previous[1]) * 0.006)
      );
      drag.previous = point;
      requestDraw();
    } else if (drag.kind === 'joint') {
      const q = drag.startQ.slice();
      q[drag.joint] += delta[0] * 0.008 - delta[1] * 0.004;
      setQ(q);
    } else if (drag.kind === 'translate') {
      const distance =
        dot(delta, drag.screenAxis) / Math.max(1, dot(drag.screenAxis, drag.screenAxis));
      setTarget(add(drag.startP, mul(drag.axis, distance)), drag.startR, { immediate: false });
    } else if (drag.kind === 'translate-plane') {
      const basis = cameraBasis();
      const movement = add(
        mul(basis.right, delta[0] / state.scale),
        mul(basis.up, -delta[1] / state.scale)
      );
      setTarget(add(drag.startP, movement), drag.startR, { immediate: false });
    } else if (drag.kind === 'rotate') {
      const angle = angleAt(point, drag);
      drag.angle += wrap(angle - drag.lastAngle);
      drag.lastAngle = angle;
      const R = solver.multiply3(solver.axisRotation(drag.axis, drag.angle), drag.startR);
      setTarget(drag.startP, R, { immediate: false });
    }
  });
  function finishDrag(event) {
    if (!state.drag || event.pointerId !== state.drag.pointerId) return;
    const kind = state.drag.kind;
    state.drag = null;
    stage.dataset.dragging = 'false';
    if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
    if (kind.startsWith('translate') || kind === 'rotate') solveTarget(true);
  }
  stage.addEventListener('pointerup', finishDrag);
  stage.addEventListener('pointercancel', finishDrag);
  stage.addEventListener(
    'wheel',
    (event) => {
      event.preventDefault();
      state.zoom = Math.max(0.5, Math.min(2.6, state.zoom * Math.exp(-event.deltaY * 0.001)));
      requestDraw();
    },
    { passive: false }
  );
  stage.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      setQ(state.q);
      return;
    }
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const axis = event.key === 'ArrowLeft' || event.key === 'ArrowRight' ? 0 : 2;
    const sign = event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? -1 : 1;
    if (state.tool === 'move') {
      const p = state.pose.tcp.slice();
      p[axis] += sign * 0.005;
      setTarget(p, state.pose.R);
    } else {
      const R = solver.multiply3(
        solver.axisRotation(column(state.pose.R, axis), sign * rad(2)),
        state.pose.R
      );
      setTarget(state.pose.tcp, R);
    }
  });
  function targetFromFields(event) {
    const id = event.target.id;
    const value = event.target.valueAsNumber;
    if (!Number.isFinite(value)) {
      syncInputs();
      return;
    }
    const position = state.targetP.slice();
    let rotation = state.targetR;
    if (id.startsWith('arm-target-')) {
      position[names.indexOf(id.slice(-1).toUpperCase())] = value;
    } else {
      const angles = eulerFromRotation(state.targetR);
      angles[['arm-roll', 'arm-pitch', 'arm-yaw'].indexOf(id)] = rad(value);
      rotation = rotationFromEuler(...angles);
    }
    setTarget(position, rotation);
  }
  $('arm-joints').innerHTML = state.q
    .map(
      (q, j) =>
        '<label class="arm-joint-control" for="arm-joint-' +
        j +
        '"><span>J' +
        (j + 1) +
        '</span><input id="arm-joint-' +
        j +
        '" type="range" min="-180" max="180" step="0.1" value="0"><output id="arm-joint-value-' +
        j +
        '">0°</output></label>'
    )
    .join('');
  $('arm-torques').innerHTML = state.q
    .map(
      (q, j) =>
        '<div class="arm-torque-row"><span>J' +
        (j + 1) +
        '</span><div class="arm-torque-track"><i id="arm-torque-fill-' +
        j +
        '"></i></div><span id="arm-torque-value-' +
        j +
        '"></span></div>'
    )
    .join('');
  for (let j = 0; j < 6; j++)
    $('arm-joint-' + j).addEventListener('input', () => {
      const q = state.q.slice();
      q[j] = rad(Number($('arm-joint-' + j).value));
      setQ(q);
    });
  for (const field of ['target-x', 'target-y', 'target-z', 'roll', 'pitch', 'yaw'])
    $('arm-' + field).addEventListener('change', targetFromFields);
  $('arm-load').addEventListener('input', () => setLoad($('arm-load').value));
  $('arm-zero').addEventListener('click', () => setQ(Array(6).fill(0)));
  $('arm-reset').addEventListener('click', () => setQ(state.q));
  function adoptMapPose() {
    if (workspace.state.selected < 0) return false;
    const index = workspace.state.selected;
    setQ(Array.from(workspace.current.angles.subarray(index * 6, index * 6 + 6)));
    setMode('arm');
    return true;
  }
  $('arm-adopt').addEventListener('click', adoptMapPose);
  for (const id of ['arm-show-frame', 'arm-show-module'])
    $(id).addEventListener('change', requestDraw);
  document
    .querySelectorAll('[data-arm-tool]')
    .forEach((button) => button.addEventListener('click', () => setTool(button.dataset.armTool)));
  document
    .querySelectorAll('[data-arm-view]')
    .forEach((button) => button.addEventListener('click', () => setView(button.dataset.armView)));
  $('arm-mode').addEventListener('click', () => setMode('arm'));
  $('workspace-mode').addEventListener('click', () => setMode('workspace'));
  window.addEventListener('resize', requestDraw);
  window.addEventListener('arm-model-change', () => {
    if (state.drag && stage.hasPointerCapture(state.drag.pointerId)) {
      stage.releasePointerCapture(state.drag.pointerId);
    }
    state.drag = null;
    stage.dataset.dragging = 'false';
    model = workspace.getModel();
    setQ(state.q);
    requestDraw();
  });
  $('arm-joints').closest('details').addEventListener('toggle', requestDraw);
  new ResizeObserver(requestDraw).observe(stage);
  window.armEditor = {
    state,
    setQ,
    setTarget,
    setLoad,
    setTool,
    setView,
    setMode,
    adoptMapPose,
    project,
    draw,
    hitTest,
    rotationFromEuler,
    eulerFromRotation,
    get model() {
      return model;
    },
  };
  setQ(Array(6).fill(0));
  setLoad(1);
  setTool('move');
  setMode(['#workspace', '#animate'].includes(location.hash) ? 'workspace' : 'arm');
  if (location.hash === '#animate' && !workspace.reducedMotion) workspace.toggleAnimation();
})();
