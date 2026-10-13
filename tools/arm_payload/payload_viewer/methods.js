/* Worked gravity calculations. The source snapshots are verified in Python;
 * these pure functions keep the browser's signed interval calculation inspectable.
 */
(function () {
  'use strict';

  const EPSILON = 1e-10;

  function feasibleInterval(ownTorque, torquePerKg, caps) {
    let lower = 0;
    let upper = Infinity;
    let limitingJoint = -1;
    let feasible = true;

    for (let joint = 0; joint < caps.length; joint += 1) {
      const own = ownTorque[joint];
      const perKg = torquePerKg[joint];
      const cap = caps[joint];
      if (Math.abs(perKg) <= EPSILON) {
        if (Math.abs(own) > cap + EPSILON) feasible = false;
        continue;
      }
      const first = (-cap - own) / perKg;
      const second = (cap - own) / perKg;
      lower = Math.max(lower, Math.min(first, second));
      const jointUpper = Math.max(first, second);
      if (jointUpper < upper) {
        upper = jointUpper;
        limitingJoint = joint;
      }
    }
    feasible = feasible && upper >= lower - EPSILON && upper >= 0;
    return { lower, upper, feasible, limitingJoint };
  }

  function calculate(snapshot, mass, caps) {
    if (!Number.isFinite(mass) || mass < 0) throw new Error('Payload must be a nonnegative mass.');
    const own = snapshot.own_gravity_torque_nm;
    const perKg = snapshot.payload_torque_per_kg_nm;
    if (
      !Array.isArray(own) ||
      !Array.isArray(perKg) ||
      own.length !== caps.length ||
      perKg.length !== caps.length
    ) {
      throw new Error('Joint torque arrays must match the motor caps.');
    }
    const joints = caps.map(function (cap, joint) {
      if (![cap, own[joint], perKg[joint]].every(Number.isFinite) || cap <= 0) {
        throw new Error('Joint torques must be finite and caps must be positive.');
      }
      const payload = mass * perKg[joint];
      const total = own[joint] + payload;
      const remaining = cap - Math.abs(total);
      return {
        own: own[joint],
        payload,
        total,
        cap,
        remaining,
        usage: Math.abs(total) / cap,
        pass: remaining >= -EPSILON,
      };
    });
    const interval = feasibleInterval(own, perKg, caps);
    const peakJoint = joints.reduce(function (best, joint, index) {
      return joint.usage > joints[best].usage ? index : best;
    }, 0);
    return {
      joints,
      interval,
      peakJoint,
      pass: joints.every(function (joint) {
        return joint.pass;
      }),
    };
  }

  const api = { calculate, feasibleInterval };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (typeof window !== 'undefined') window.ArmPayloadMethods = api;
  if (typeof document === 'undefined') return;

  const byId = function (id) {
    return document.getElementById(id);
  };
  const section = byId('calculation');
  if (!section) return;
  const analysis = typeof META !== 'undefined' ? META.analysis : null;
  if (!analysis || !analysis.examples || !analysis.examples.zero || !analysis.examples.extended) {
    byId('method-load-status').textContent =
      'The verified example data is missing. Rebuild the viewer to restore the worked calculation.';
    byId('method-load-status').dataset.pass = 'false';
    byId('method-payload').disabled = true;
    byId('method-pose').disabled = true;
    return;
  }

  const descriptions = {
    zero: 'All six joint angles are zero. This is one posture, not a reach or payload rating for the whole arm.',
    extended:
      'A demanding extended posture found by the numerical search. It illustrates a low payload ceiling; it is not a proven global minimum or a collision-validated motion.',
  };
  const caps = analysis.motor_caps_nm;
  const presetButtons = Array.from(section.querySelectorAll('[data-method-load]'));

  function signed(value, decimals) {
    if (Math.abs(value) < Math.pow(10, -decimals) / 2) return (0).toFixed(decimals);
    return (value < 0 ? '−' : '+') + Math.abs(value).toFixed(decimals);
  }

  function render() {
    const key = byId('method-pose').value;
    const snapshot = analysis.examples[key];
    const mass = Number(byId('method-payload').value);
    const result = calculate(snapshot, mass, caps);
    const limiting = result.interval.limitingJoint;
    const capacity = result.interval.feasible ? result.interval.upper : null;
    const mostLoaded = result.joints[result.peakJoint];

    byId('method-payload-value').textContent = mass.toFixed(2) + ' kg';
    byId('method-payload').setAttribute('aria-valuetext', mass.toFixed(2) + ' kilograms');
    byId('method-pose-description').textContent = descriptions[key];
    byId('method-capacity').textContent = !result.interval.feasible
      ? 'No feasible payload'
      : Number.isFinite(capacity)
      ? capacity.toFixed(3) + ' kg'
      : 'No finite gravity limit';
    byId('method-limiting').textContent = limiting >= 0 ? 'Joint ' + (limiting + 1) : 'None';
    byId('method-moving-mass').textContent = analysis.moving_arm_mass_kg.toFixed(3) + ' kg';
    byId('method-reach-bound').textContent = analysis.reach_bound_m.toFixed(3) + ' m';
    presetButtons.forEach(function (button) {
      button.setAttribute(
        'aria-pressed',
        String(Math.abs(Number(button.dataset.methodLoad) - mass) < EPSILON)
      );
    });

    const status = byId('method-load-status');
    status.dataset.pass = String(result.pass);
    status.textContent = result.pass
      ? 'All six static torque checks pass at ' +
        mass.toFixed(2) +
        ' kg. Joint ' +
        (result.peakJoint + 1) +
        ' has the least remaining torque: ' +
        mostLoaded.remaining.toFixed(2) +
        ' N·m.'
      : 'Static torque limit exceeded at ' +
        mass.toFixed(2) +
        ' kg. Joint ' +
        (result.peakJoint + 1) +
        ' needs ' +
        Math.abs(mostLoaded.total).toFixed(2) +
        ' N·m against its ' +
        mostLoaded.cap.toFixed(0) +
        ' N·m cap.';

    const body = byId('method-joint-rows');
    body.textContent = '';
    result.joints.forEach(function (joint, index) {
      const row = document.createElement('tr');
      row.dataset.limiting = String(index === limiting);
      row.dataset.over = String(!joint.pass);
      const heading = document.createElement('th');
      heading.scope = 'row';
      heading.textContent = 'J' + (index + 1);
      row.appendChild(heading);
      [joint.own, joint.payload, joint.total, joint.cap, joint.remaining].forEach(function (
        value,
        column
      ) {
        const cell = document.createElement('td');
        cell.textContent = column === 3 ? value.toFixed(1) : signed(value, 3);
        if (column === 2) {
          cell.className = 'method-total';
          const track = document.createElement('span');
          track.className = 'method-util-track';
          track.setAttribute('aria-hidden', 'true');
          const fill = document.createElement('span');
          fill.className = 'method-util-fill';
          fill.style.width = Math.min(100, joint.usage * 100).toFixed(2) + '%';
          track.appendChild(fill);
          cell.appendChild(track);
        }
        if (column === 4) cell.className = 'method-remaining';
        row.appendChild(cell);
      });
      body.appendChild(row);
    });

    if (limiting >= 0 && result.interval.feasible) {
      const own = snapshot.own_gravity_torque_nm[limiting];
      const perKg = snapshot.payload_torque_per_kg_nm[limiting];
      byId('method-substitution-title').textContent =
        'Joint ' + (limiting + 1) + ' sets this posture’s ceiling';
      const sameDirection = own * perKg >= 0;
      byId('method-substitution-equation').textContent = sameDirection
        ? 'm ≤ (' +
          caps[limiting].toFixed(0) +
          ' − ' +
          Math.abs(own).toFixed(4) +
          ') ÷ ' +
          Math.abs(perKg).toFixed(4) +
          ' = ' +
          capacity.toFixed(3) +
          ' kg'
        : '|' +
          signed(own, 4) +
          ' + m × (' +
          signed(perKg, 4) +
          ')| ≤ ' +
          caps[limiting].toFixed(0) +
          ' N·m';
      byId('method-substitution-explanation').textContent = sameDirection
        ? 'The arm contributes ' +
          Math.abs(own).toFixed(4) +
          ' N·m and each kilogram adds ' +
          Math.abs(perKg).toFixed(4) +
          ' N·m in the same turning direction at this joint. The other five joints also satisfy their limits at this mass. This subtraction is valid for this posture; the general calculation keeps signed torques.'
        : 'The two torque contributions act in opposite directions, so the full signed interval calculation is needed. We intersect the allowed mass intervals from all six joints.';
    }
    byId('method-joint-angles').textContent =
      'J1–J6 angles: ' +
      snapshot.joint_angles_deg
        .map(function (angle) {
          return signed(angle, 2) + '°';
        })
        .join(', ') +
      '.';
    byId('method-tcp-location').textContent =
      'Grasp point relative to the mounted base frame: [' +
      snapshot.payload_center_world_m
        .map(function (value) {
          return value.toFixed(4);
        })
        .join(', ') +
      '] m. The pose geometry stays fixed as payload changes.';
    api.lastResult = { pose: key, mass, result };
  }

  byId('method-pose').addEventListener('change', render);
  byId('method-payload').addEventListener('input', render);
  presetButtons.forEach(function (button) {
    button.addEventListener('click', function () {
      byId('method-payload').value = button.dataset.methodLoad;
      render();
    });
  });
  api.render = render;
  render();
})();
