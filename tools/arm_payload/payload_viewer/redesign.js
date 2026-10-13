(function () {
  'use strict';
  var data = JSON.parse(document.getElementById('redesign-data').textContent);
  var $ = function (id) {
    return document.getElementById(id);
  };
  var concepts = data.concepts || [],
    assumptions = data.assumptions || {};
  if (!concepts.length) {
    document.querySelector('main').innerHTML =
      '<div class="error-box">No design calculations are embedded in this page.</div>';
    return;
  }
  var state = {
    conceptId: (
      concepts.find(function (c) {
        return c.id === 'serial_geared';
      }) || concepts[0]
    ).id,
    caseIndex: 0,
    playing: false,
    view: 'perspective',
    azimuth: -0.87,
    elevation: 0.43,
    zoom: 1,
  };
  var canvas = $('scene-canvas'),
    ctx = canvas.getContext('2d'),
    width = 800,
    height = 540;
  var timer = null,
    projectedSamples = [],
    caseMaps = {},
    pickerButtons = {};
  var cubeMin = assumptions.cube_min || [0.1, -0.5, -0.5],
    cubeMax = assumptions.cube_max || [1.1, 0.5, 0.5];
  var center = [(cubeMin[0] + cubeMax[0]) / 2, 0, (cubeMin[2] + cubeMax[2]) / 2];
  var colors = { serial_short: '#b77642', serial_long: '#477eaf', serial_geared: '#279b89' };
  var fmt = function (n, p) {
    return Number.isFinite(Number(n)) ? Number(n).toFixed(p == null ? 2 : p) : '—';
  };
  var escape = function (s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  };
  var concept = function () {
    return (
      concepts.find(function (c) {
        return c.id === state.conceptId;
      }) || concepts[0]
    );
  };
  var cases = function () {
    return concept().cases || [];
  };
  var current = function () {
    return cases()[state.caseIndex] || {};
  };
  var vector = function (v) {
    return Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
  };
  var poseGood = function (c) {
    return c.reachable === true && c.torque_pass === true;
  };
  var poseColor = function (c) {
    return !c.reachable ? '#c8a5a2' : c.torque_pass ? '#1b9b8c' : '#c48d40';
  };
  var list = function (id, rows) {
    $(id).innerHTML = (rows || [])
      .map(function (t) {
        return '<li>' + escape(t) + '</li>';
      })
      .join('');
  };
  function rate(obj) {
    return Number.isFinite(obj.passed) && Number.isFinite(obj.total)
      ? obj.passed.toLocaleString() + ' / ' + obj.total.toLocaleString()
      : '—';
  }
  function statusText(c) {
    return !c.reachable
      ? 'No demonstrated reach for this pose'
      : c.torque_pass
      ? concept().id === 'serial_geared'
        ? 'Ideal pose + proposed torque budget pass'
        : 'Reach + static torque check pass'
      : 'Reached, but static torque exceeds limit';
  }
  function buildPicker() {
    concepts.forEach(function (c, i) {
      var b = document.createElement('button'),
        g = c.geometry || {},
        t = c.statics || {};
      var good = g.analytic_pass === true && t.passed === t.total && t.total > 0;
      b.type = 'button';
      b.className = 'concept-card';
      b.style.setProperty('--concept-color', colors[c.id] || '#6895a2');
      b.dataset.status = good ? 'good' : 'bad';
      b.innerHTML =
        '<span class="card-number">CONCEPT 0' +
        (i + 1) +
        '</span><strong>' +
        escape(c.name) +
        '</strong><span class="card-subtitle">' +
        escape(c.subtitle || '') +
        '</span><span class="card-verdict">' +
        escape(
          g.analytic_pass === false
            ? 'Does not cover the cube'
            : good
            ? 'Ideal reach + proposed torque budget'
            : 'Long reach · torque limited'
        ) +
        '</span>';
      b.addEventListener('click', function () {
        chooseConcept(c.id);
      });
      $('concept-picker').appendChild(b);
      pickerButtons[c.id] = b;
    });
  }
  function uniquePositions(c) {
    var byPosition = {},
      orientations = [];
    (c.cases || []).forEach(function (v, i) {
      if (!vector(v.position)) return;
      var key = v.position
        .map(function (x) {
          return Number(x).toFixed(6);
        })
        .join(',');
      if (!byPosition[key]) byPosition[key] = { position: v.position, indices: [] };
      byPosition[key].indices.push(i);
      if (orientations.indexOf(v.orientation_label || 'Orientation') < 0)
        orientations.push(v.orientation_label || 'Orientation');
    });
    var positions = Object.keys(byPosition).map(function (k) {
      return byPosition[k];
    });
    var axes = [0, 1, 2].map(function (j) {
      return Array.from(
        new Set(
          positions.map(function (p) {
            return p.position[j];
          })
        )
      ).sort(function (a, b) {
        return a - b;
      });
    });
    return { positions: positions, orientations: orientations, axes: axes };
  }
  concepts.forEach(function (c) {
    caseMaps[c.id] = uniquePositions(c);
  });
  function buildControls() {
    var map = caseMaps[state.conceptId];
    $('target-controls').innerHTML = map.axes
      .map(function (a, j) {
        return (
          '<div class="axis-control"><label class="axis-label" for="target-' +
          j +
          '"><span>' +
          ['X · depth', 'Y · width', 'Z · height'][j] +
          '</span><output id="target-value-' +
          j +
          '"></output></label><input id="target-' +
          j +
          '" type="range" min="0" max="' +
          Math.max(0, a.length - 1) +
          '" step="1" value="0" aria-label="' +
          ['Target X depth', 'Target Y width', 'Target Z height'][j] +
          '"></div>'
        );
      })
      .join('');
    [0, 1, 2].forEach(function (j) {
      $('target-' + j).addEventListener('input', function () {
        stop();
        var target = map.axes.map(function (axis, k) {
          return axis[Number($('target-' + k).value)];
        });
        selectTarget(target, $('orientation').value);
      });
    });
    $('orientation').innerHTML = map.orientations
      .map(function (o) {
        return '<option value="' + escape(o) + '">' + escape(o) + '</option>';
      })
      .join('');
    $('case-slider').max = Math.max(0, cases().length - 1);
  }
  function selectTarget(target, orientation) {
    var best = 0,
      distance = Infinity;
    cases().forEach(function (c, i) {
      if (!vector(c.position)) return;
      var d = c.position.reduce(function (s, v, k) {
        return s + (v - target[k]) * (v - target[k]);
      }, 0);
      if ((c.orientation_label || 'Orientation') !== orientation) d += 100;
      if (d < distance) {
        best = i;
        distance = d;
      }
    });
    chooseCase(best);
  }
  function chooseConcept(id) {
    if (
      !concepts.some(function (c) {
        return c.id === id;
      })
    )
      return false;
    var old = current(),
      oldPosition = old.position,
      oldOrientation = old.orientation_label;
    stop();
    state.conceptId = id;
    state.caseIndex = 0;
    var c = concept(),
      g = c.geometry || {},
      t = c.statics || {};
    Object.keys(pickerButtons).forEach(function (k) {
      pickerButtons[k].classList.toggle('active', k === id);
      pickerButtons[k].setAttribute('aria-pressed', String(k === id));
    });
    $('concept-kicker').textContent =
      'DESIGN CONCEPT ' + String(concepts.indexOf(c) + 1).padStart(2, '0');
    $('concept-title').textContent = c.name;
    $('concept-description').textContent = c.description || '';
    $('concept-verdict').textContent = c.verdict || '';
    $('concept-verdict').classList.toggle('bad', g.analytic_pass !== true || t.passed !== t.total);
    $('geometry-result').textContent = rate(g);
    $('geometry-note').textContent =
      (assumptions.position_count || caseMaps[id].positions.length) +
      ' positions × ' +
      (assumptions.orientation_count || caseMaps[id].orientations.length) +
      ' orientations';
    $('torque-result').textContent = rate(t);
    $('torque-note').textContent =
      id === 'serial_geared'
        ? 'Conditional on added transmission budgets'
        : 'Current motors · 13 N·m per rotary joint';
    $('analytic-result').textContent =
      g.analytic_pass === true
        ? 'Covered'
        : g.analytic_pass === false
        ? 'Not covered'
        : 'Sampled only';
    $('analytic-note').textContent =
      g.analytic_pass === true
        ? 'Shoulder at cube middle height only'
        : g.analytic_pass === false
        ? 'A necessary reach condition fails'
        : 'Pose samples do not prove full coverage';
    $('scene-title').textContent =
      id === 'serial_short'
        ? '350 + 350 mm pivot spans'
        : id === 'serial_geared'
        ? '750 + 750 mm · proposed reduction'
        : '750 + 750 mm pivot spans';
    $('analytic-claim').textContent =
      g.analytic_claim || 'Only the discrete cases shown here have been checked.';
    list('changes', c.changes);
    list('limitations', c.limitations);
    list('mass-model', t.mass_model);
    $('validation-method').textContent = c.validation || '';
    var dimensions = c.dimensions || [];
    if (!Array.isArray(dimensions))
      dimensions = Object.keys(dimensions).map(function (k) {
        return { label: k, value: dimensions[k] };
      });
    $('dimensions').innerHTML = dimensions
      .map(function (d) {
        return (
          '<div class="dimension-row"><span>' +
          escape(d.label) +
          '</span><strong>' +
          escape(d.value) +
          '</strong></div>'
        );
      })
      .join('');
    buildControls();
    if (vector(oldPosition)) selectTarget(oldPosition, oldOrientation);
    else chooseCase(0);
    return true;
  }
  function chooseCase(index) {
    state.caseIndex = Math.max(0, Math.min(cases().length - 1, Math.floor(Number(index) || 0)));
    var c = current(),
      p = vector(c.position) ? c.position : [0, 0, 0],
      map = caseMaps[state.conceptId];
    $('case-slider').value = state.caseIndex;
    $('case-count').textContent = state.caseIndex + 1 + ' / ' + cases().length;
    map.axes.forEach(function (a, j) {
      var k = 0;
      for (var i = 1; i < a.length; i++) if (Math.abs(a[i] - p[j]) < Math.abs(a[k] - p[j])) k = i;
      $('target-' + j).value = k;
      $('target-value-' + j).textContent = fmt(p[j], 2) + ' m';
    });
    $('orientation').value = c.orientation_label || 'Orientation';
    $('pose-status').textContent = statusText(c);
    $('pose-status').classList.toggle('bad', !poseGood(c));
    $('coordinates').innerHTML = p
      .map(function (v, j) {
        return '<span>' + ['X', 'Y', 'Z'][j] + '<b>' + fmt(v, 3) + ' m</b></span>';
      })
      .join('');
    var pe = Number(c.position_error_m),
      oe = Number(c.orientation_error_rad);
    $('pose-error').textContent =
      c.reachable && Number.isFinite(pe) && Number.isFinite(oe)
        ? 'Pose residual: ' + fmt(pe * 1000, 3) + ' mm · ' + fmt((oe * 180) / Math.PI, 3) + '°'
        : c.reachable
        ? 'Demonstrated by the saved geometric solution.'
        : 'No solution is drawn for an unreachable test pose.';
    $('scene-status').textContent = (state.playing ? 'Playback · ' : '') + statusText(c);
    renderLoads();
    draw();
    return c;
  }
  function renderLoads() {
    var c = current(),
      t = concept().statics || {},
      values = c.torques_nm || [],
      caps = t.caps_nm || [13, 13, 13, 13, 13, 13],
      labels = t.joint_labels || [];
    $('load-heading').textContent = 'Static holding torque';
    $('load-tag').textContent =
      concept().id === 'serial_geared' ? 'Proposed caps' : 'Current motor caps';
    if (!c.reachable) {
      $('load-values').innerHTML =
        '<p class="microcopy">A valid pose is needed before its holding torque can be evaluated.</p>';
      $('load-note').textContent = 'Use another target or design to inspect a feasible pose.';
    } else {
      $('load-values').innerHTML =
        '<p class="microcopy">Required / available output torque, N·m</p>' +
        torqueRows(values, caps, labels);
      $('load-note').textContent =
        t.note ||
        'Absolute static torque, including the modeled arm and 2 kg payload. No acceleration, friction or safety factor is included.';
    }
    var q = c.q || [];
    $('joint-angles').hidden = !c.reachable || !q.length;
    $('angle-values').innerHTML = q
      .map(function (v, i) {
        return '<span>J' + (i + 1) + ' ' + fmt((v * 180) / Math.PI, 1) + '°</span>';
      })
      .join('');
  }
  function torqueRows(values, caps, labels) {
    return (
      values
        .map(function (v, j) {
          var a = Math.abs(v),
            cap = caps[j] || 13;
          return (
            '<div class="torque-row"><span title="' +
            escape(labels[j] || 'Joint ' + (j + 1)) +
            '">' +
            'J' +
            (j + 1) +
            '</span><div class="torque-track"><div class="torque-fill ' +
            (a > cap ? 'over' : '') +
            '" style="width:' +
            Math.min(100, (a / cap) * 100) +
            '%"></div></div><span class="torque-number">' +
            fmt(a, 1) +
            '</span><span class="torque-cap">/ ' +
            fmt(cap, 1) +
            '</span></div>'
          );
        })
        .join('') || '<p class="microcopy">Torque values are not available for this case.</p>'
    );
  }
  function project(p) {
    var x = p[0] - center[0],
      y = p[1] - center[1],
      z = p[2] - center[2],
      ca = Math.cos(state.azimuth),
      sa = Math.sin(state.azimuth),
      ce = Math.cos(state.elevation),
      se = Math.sin(state.elevation);
    var scale = Math.min(width / 3.0, height / 2.65) * state.zoom;
    return [
      width / 2 + (-sa * x + ca * y) * scale,
      height / 2 + 15 - (-se * ca * x - se * sa * y + ce * z) * scale,
      ce * ca * x + ce * sa * y + se * z,
    ];
  }
  function line(a, b, color, lineWidth, dash) {
    var p = project(a),
      q = project(b);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth || 1;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    ctx.moveTo(p[0], p[1]);
    ctx.lineTo(q[0], q[1]);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function dot(p, r, fill, stroke) {
    var xy = project(p);
    ctx.beginPath();
    ctx.arc(xy[0], xy[1], r, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  function label(p, text, color, dx, dy) {
    var a = project(p);
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.fillStyle = color || '#6f8798';
    ctx.textAlign = 'left';
    ctx.fillText(text, a[0] + (dx || 7), a[1] + (dy || -7));
  }
  function boxCorners(lo, hi) {
    var out = [];
    for (var i = 0; i < 8; i++)
      out.push([i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]]);
    return out;
  }
  function box(lo, hi, color, lineWidth, dash) {
    var p = boxCorners(lo, hi);
    for (var i = 0; i < 8; i++)
      for (var b = 0; b < 3; b++)
        if (!(i & (1 << b))) line(p[i], p[i | (1 << b)], color, lineWidth, dash);
  }
  function drawCube() {
    var p = boxCorners(cubeMin, cubeMax),
      faces = [
        [0, 1, 3, 2],
        [0, 2, 6, 4],
        [0, 1, 5, 4],
        [4, 5, 7, 6],
        [1, 3, 7, 5],
        [2, 3, 7, 6],
      ];
    faces.sort(function (a, b) {
      return (
        a.reduce(function (s, i) {
          return s + project(p[i])[2];
        }, 0) -
        b.reduce(function (s, i) {
          return s + project(p[i])[2];
        }, 0)
      );
    });
    faces.slice(0, 3).forEach(function (f) {
      ctx.beginPath();
      f.forEach(function (i, k) {
        var q = project(p[i]);
        if (k) ctx.lineTo(q[0], q[1]);
        else ctx.moveTo(q[0], q[1]);
      });
      ctx.closePath();
      ctx.fillStyle = '#b5cfe712';
      ctx.fill();
    });
    box(cubeMin, cubeMax, '#9ab9d3', 1.1);
    label([cubeMax[0], cubeMin[1], cubeMin[2]], '1 m build volume', '#7697b1', 7, 14);
  }
  function drawAxes() {
    var origin = [0, 0, 0];
    dot(origin, 4, '#566e80');
    [
      ['X', [0.25, 0, 0], '#b67468'],
      ['Y', [0, 0.25, 0], '#72a48a'],
      ['Z', [0, 0, 0.25], '#6f98bb'],
    ].forEach(function (a) {
      line(origin, a[1], a[2], 1.5);
      label(a[1], a[0], a[2]);
    });
    label([0, 0, -0.06], 'Fixed shoulder', '#8194a4', -28, 14);
  }
  function drawArm(c) {
    if (!c.reachable) return;
    var points = (c.points || []).filter(vector);
    if (points.length < 2) return;
    for (var i = 1; i < points.length; i++)
      line(
        points[i - 1],
        points[i],
        i === 2 ? '#be8249' : i === 3 ? '#4884b5' : '#314f63',
        i === points.length - 1 ? 5 : 7
      );
    points.forEach(function (p, j) {
      dot(p, j === 0 ? 8 : 5, j === 0 ? '#324c5e' : '#fff', '#36566c');
    });
    label(points[0], 'Base', '#718899', -20, 19);
  }
  function drawModule(c) {
    if (!$('show-module').checked || !vector(c.position)) return;
    var r = c.orientation_matrix,
      axis =
        Array.isArray(r) && r.length === 3
          ? r.map(function (row) {
              return row[0];
            })
          : [1, 0, 0];
    if (!vector(axis)) axis = [1, 0, 0];
    var length = Number(assumptions.module_length_m) || 0.5,
      half = length / 2,
      a = c.position.map(function (v, j) {
        return v - half * axis[j];
      }),
      b = c.position.map(function (v, j) {
        return v + half * axis[j];
      });
    line(a, b, c.reachable ? '#9160a2' : '#b496bb', 5, c.reachable ? [] : [5, 4]);
    dot(c.position, 5, '#fff', '#9160a2');
    label(b, '50 cm module', '#9876a1', 8, -7);
  }
  function draw() {
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    projectedSamples = [];
    drawCube();
    var c = current(),
      map = caseMaps[state.conceptId];
    if ($('show-samples').checked) {
      var orientation = c.orientation_label;
      map.positions.forEach(function (group) {
        var idx = group.indices.find(function (i) {
          return cases()[i].orientation_label === orientation;
        });
        if (idx == null) idx = group.indices[0];
        var v = cases()[idx],
          p = project(group.position);
        projectedSamples.push({ x: p[0], y: p[1], index: idx });
        dot(group.position, 2.5, poseColor(v));
      });
    }
    drawAxes();
    drawArm(c);
    if (vector(c.position)) {
      drawModule(c);
      var p = project(c.position);
      ctx.strokeStyle = poseColor(c);
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(p[0], p[1], 10, 0, Math.PI * 2);
      ctx.stroke();
      var r = c.orientation_matrix;
      if (Array.isArray(r) && r.length === 3) {
        var axis = r.map(function (row) {
          return row[2];
        });
        if (vector(axis))
          line(
            c.position,
            c.position.map(function (v, j) {
              return v + 0.11 * axis[j];
            }),
            '#a385b8',
            2
          );
      }
    }
  }
  function resize() {
    var rect = $('scene').getBoundingClientRect(),
      ratio = Math.min(window.devicePixelRatio || 1, 2);
    width = rect.width;
    height = rect.height;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    state.playing = false;
    $('play').textContent = '▶ Play tested poses';
    $('play').setAttribute('aria-pressed', 'false');
    if (cases().length) $('scene-status').textContent = statusText(current());
  }
  function play() {
    if (state.playing) return;
    state.playing = true;
    $('play').textContent = 'Ⅱ Pause poses';
    $('play').setAttribute('aria-pressed', 'true');
    chooseCase(state.caseIndex);
    timer = setInterval(function () {
      var map = caseMaps[state.conceptId],
        c = current(),
        pindex = map.positions.findIndex(function (g) {
          return g.indices.indexOf(state.caseIndex) >= 0;
        }),
        oindex = map.orientations.indexOf(c.orientation_label || 'Orientation');
      selectTarget(
        map.positions[(pindex + 1) % map.positions.length].position,
        map.orientations[(oindex + 1) % map.orientations.length]
      );
    }, 850);
  }
  function setView(view) {
    state.view = view;
    state.zoom = 1;
    if (view === 'top') {
      state.azimuth = -Math.PI / 2;
      state.elevation = Math.PI / 2 - 0.0001;
    } else if (view === 'front') {
      state.azimuth = -Math.PI / 2;
      state.elevation = 0;
    } else {
      state.azimuth = -0.87;
      state.elevation = 0.43;
    }
    document.querySelectorAll('[data-view]').forEach(function (b) {
      b.classList.toggle('active', b.dataset.view === view);
    });
    draw();
  }
  $('play').addEventListener('click', function () {
    state.playing ? stop() : play();
  });
  $('orientation').addEventListener('change', function () {
    stop();
    selectTarget(current().position, $('orientation').value);
  });
  $('case-slider').addEventListener('input', function () {
    stop();
    chooseCase(Number(this.value));
  });
  $('show-samples').addEventListener('change', draw);
  $('show-module').addEventListener('change', draw);
  $('reset-view').addEventListener('click', function () {
    setView('perspective');
  });
  document.querySelectorAll('[data-view]').forEach(function (b) {
    b.addEventListener('click', function () {
      setView(b.dataset.view);
    });
  });
  var drag = null;
  $('scene').addEventListener('pointerdown', function (e) {
    drag = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, id: e.pointerId };
    this.setPointerCapture(e.pointerId);
  });
  $('scene').addEventListener('pointermove', function (e) {
    if (!drag) return;
    var dx = e.clientX - drag.x,
      dy = e.clientY - drag.y;
    state.azimuth -= dx * 0.007;
    state.elevation = Math.max(-1.4, Math.min(1.55, state.elevation + dy * 0.007));
    drag.x = e.clientX;
    drag.y = e.clientY;
    state.view = 'custom';
    document.querySelectorAll('[data-view]').forEach(function (b) {
      b.classList.remove('active');
    });
    draw();
  });
  $('scene').addEventListener('pointerup', function (e) {
    if (!drag) return;
    var moved = Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY);
    drag = null;
    if (moved < 5) {
      var r = this.getBoundingClientRect(),
        x = e.clientX - r.left,
        y = e.clientY - r.top,
        best = null,
        distance = 14;
      projectedSamples.forEach(function (p) {
        var d = Math.hypot(p.x - x, p.y - y);
        if (d < distance) {
          best = p;
          distance = d;
        }
      });
      if (best) {
        stop();
        chooseCase(best.index);
      }
    }
  });
  $('scene').addEventListener('pointercancel', function () {
    drag = null;
  });
  $('scene').addEventListener(
    'wheel',
    function (e) {
      e.preventDefault();
      state.zoom = Math.max(0.6, Math.min(2.4, state.zoom * Math.exp(-e.deltaY * 0.001)));
      draw();
    },
    { passive: false }
  );
  $('scene').addEventListener('keydown', function (e) {
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].indexOf(e.key) < 0) return;
    e.preventDefault();
    state.azimuth += e.key === 'ArrowLeft' ? 0.1 : e.key === 'ArrowRight' ? -0.1 : 0;
    state.elevation = Math.max(
      -1.4,
      Math.min(
        1.55,
        state.elevation + (e.key === 'ArrowUp' ? 0.08 : e.key === 'ArrowDown' ? -0.08 : 0)
      )
    );
    draw();
  });
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stop();
  });
  window.addEventListener('resize', resize);
  $('assumptions').innerHTML =
    '<p><strong>1 m × 1 m × 1 m target volume</strong>Unconfirmed shoulder reference at the near face’s middle height, ' +
    fmt(cubeMin[0] * 100, 0) +
    ' cm outside it. Targets describe the grasp centre, not full object containment.</p><p><strong>2 kg, held near its centre of mass</strong>A 50 cm workpiece extends about 25 cm to either side. Thickness, grasp offset, collision clearance and assembly access remain to be specified.</p><p><strong>Ideal joints and static gravity</strong>Existing motors use 13 N·m caps. The reduction concept proposes 88.4 N·m at the shoulder and 44.2 N·m at the elbow, with added transmission mass. Dynamics, cable routing, thermal endurance and collision-free transfer are not established.</p>';
  var mounts = data.mounting_scenarios || [];
  $('mounting-scenarios').innerHTML = mounts
    .map(function (m, i) {
      return (
        '<div><span>' +
        escape(i === 0 ? 'Shoulder at mid-height' : 'Shoulder 50 mm above bottom') +
        '</span><strong>' +
        fmt(m.required_wrist_reach_m, 3) +
        ' m wrist reach required</strong><small>' +
        escape(
          i === 0
            ? '1.50 m combined span passes ideal geometry'
            : '1.50 m combined span is too short'
        ) +
        '</small></div>'
      );
    })
    .join('');
  var sources = data.sources || [];
  $('sources').innerHTML = sources
    .map(function (s) {
      return /^https?:\/\//.test(s.url || '')
        ? '<a href="' +
            escape(s.url) +
            '" target="_blank" rel="noopener noreferrer">' +
            escape(s.title) +
            '</a>'
        : '';
    })
    .join('');
  buildPicker();
  chooseConcept(state.conceptId);
  selectTarget(center, 'Approach −Z · roll 0°');
  resize();
  window.redesignLab = {
    state: state,
    data: data,
    chooseConcept: chooseConcept,
    chooseCase: chooseCase,
    selectTarget: selectTarget,
    draw: draw,
    play: play,
    stop: stop,
    setView: setView,
    current: current,
  };
})();
