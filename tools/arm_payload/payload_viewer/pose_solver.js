/*
 * Six-axis ARM_REBOT pose and signed static-payload solver.
 *
 * Dependency-free: browser window.ArmPoseSolver or CommonJS module.exports.
 * Model: the existing viewer metadata {mount,origins,axes,centers,masses,grasp,caps}.
 * Matrices in that metadata are nested row-major 4x4 arrays. Public rotations
 * are nested row-major 3x3 arrays; R is the WORLD rotation of the final URDF
 * tool link, not a new frame constructed from the grasp direction.
 *
 * q is radians; positions/COMs are meters; torque is N*m; mass is kilograms.
 * Joints are continuous, with no invented joint limits. This local inverse
 * kinematics solver does not check collisions, paths, cable limits or dynamics.
 * Failure to converge is NOT proof that the target is globally unreachable.
 * Solving a pose does not establish that motors can hold the selected payload.
 */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ArmPoseSolver = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const GRAVITY = 9.81;
  const UNIT_TOLERANCE = 1e-10;
  const now =
    typeof performance !== 'undefined' && typeof performance.now === 'function'
      ? () => performance.now()
      : () => Date.now();

  function finiteVector(value, length, label) {
    if (!value || value.length !== length || !Array.from(value).every(Number.isFinite))
      throw new Error(label + ' must have ' + length + ' finite components.');
    return Array.from(value);
  }
  function flatRotation(value) {
    const a =
      value && value.length === 3 && value[0] && value[0].length === 3
        ? Array.from(value[0]).concat(Array.from(value[1]), Array.from(value[2]))
        : value;
    return finiteVector(a, 9, 'Rotation');
  }
  function nestedRotation(r) {
    return [r.slice(0, 3), r.slice(3, 6), r.slice(6, 9)];
  }
  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function subtract(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }
  function norm(a) {
    return Math.sqrt(dot(a, a));
  }
  function mul3(a, b) {
    const c = new Array(9);
    for (let row = 0; row < 3; row++) {
      const r = row * 3;
      for (let col = 0; col < 3; col++)
        c[r + col] = a[r] * b[col] + a[r + 1] * b[col + 3] + a[r + 2] * b[col + 6];
    }
    return c;
  }
  function transposeFlat(r) {
    return [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];
  }
  function rotateFlat(r, p) {
    return [
      r[0] * p[0] + r[1] * p[1] + r[2] * p[2],
      r[3] * p[0] + r[4] * p[1] + r[5] * p[2],
      r[6] * p[0] + r[7] * p[1] + r[8] * p[2],
    ];
  }
  function translated(r, p, translation) {
    const v = rotateFlat(r, p);
    return [v[0] + translation[0], v[1] + translation[1], v[2] + translation[2]];
  }
  function axisRotationFlat(axis, angle) {
    const x = axis[0],
      y = axis[1],
      z = axis[2],
      c = Math.cos(angle),
      s = Math.sin(angle),
      u = 1 - c;
    return [
      c + x * x * u,
      x * y * u - z * s,
      x * z * u + y * s,
      y * x * u + z * s,
      c + y * y * u,
      y * z * u - x * s,
      z * x * u - y * s,
      z * y * u + x * s,
      c + z * z * u,
    ];
  }

  /** Row-major 3x3 product. Pre-multiply for a world-axis tool rotation. */
  function multiply3(a, b) {
    return nestedRotation(mul3(flatRotation(a), flatRotation(b)));
  }
  function transpose3(a) {
    return nestedRotation(transposeFlat(flatRotation(a)));
  }
  /** Rodrigues rotation, with a normalized axis and angle in radians. */
  function axisRotation(axis, angle) {
    const a = finiteVector(axis, 3, 'Rotation axis'),
      length = norm(a);
    if (!Number.isFinite(angle) || !(length > 0))
      throw new Error('Rotation requires a nonzero axis and finite angle.');
    return nestedRotation(
      axisRotationFlat(
        a.map((x) => x / length),
        angle
      )
    );
  }

  /*
   * Quaternion-based SO(3) logarithm. The largest-diagonal extraction avoids
   * dividing by sin(theta), which is unstable for rotations near pi. Choosing
   * quaternion w >= 0 returns a shortest rotation with angle in [0, pi].
   * Exactly at pi either axis sign describes the same rotation.
   */
  function logFlat(r) {
    const trace = r[0] + r[4] + r[8];
    let w, x, y, z, s;
    if (trace > 0) {
      s = 2 * Math.sqrt(Math.max(0, trace + 1));
      w = s / 4;
      x = (r[7] - r[5]) / s;
      y = (r[2] - r[6]) / s;
      z = (r[3] - r[1]) / s;
    } else if (r[0] >= r[4] && r[0] >= r[8]) {
      s = 2 * Math.sqrt(Math.max(0, 1 + r[0] - r[4] - r[8]));
      w = (r[7] - r[5]) / s;
      x = s / 4;
      y = (r[1] + r[3]) / s;
      z = (r[2] + r[6]) / s;
    } else if (r[4] >= r[8]) {
      s = 2 * Math.sqrt(Math.max(0, 1 + r[4] - r[0] - r[8]));
      w = (r[2] - r[6]) / s;
      x = (r[1] + r[3]) / s;
      y = s / 4;
      z = (r[5] + r[7]) / s;
    } else {
      s = 2 * Math.sqrt(Math.max(0, 1 + r[8] - r[0] - r[4]));
      w = (r[3] - r[1]) / s;
      x = (r[2] + r[6]) / s;
      y = (r[5] + r[7]) / s;
      z = s / 4;
    }
    const magnitude = Math.hypot(w, x, y, z);
    if (!(magnitude > 0) || !Number.isFinite(magnitude))
      throw new Error('Cannot take the logarithm of this rotation.');
    const sign = w < 0 ? -1 : 1;
    w *= sign / magnitude;
    x *= sign / magnitude;
    y *= sign / magnitude;
    z *= sign / magnitude;
    const sineHalf = Math.hypot(x, y, z);
    const scale = sineHalf < 1e-12 ? 2 : (2 * Math.atan2(sineHalf, w)) / sineHalf;
    return [scale * x, scale * y, scale * z];
  }
  function rotationLog(rotation) {
    return logFlat(flatRotation(rotation));
  }
  function validateRotation(r) {
    const residual = mul3(r, transposeFlat(r));
    let error = 0;
    for (let i = 0; i < 9; i++)
      error = Math.max(error, Math.abs(residual[i] - (i % 4 === 0 ? 1 : 0)));
    const determinant =
      r[0] * (r[4] * r[8] - r[5] * r[7]) -
      r[1] * (r[3] * r[8] - r[5] * r[6]) +
      r[2] * (r[3] * r[7] - r[4] * r[6]);
    if (error > 1e-5 || Math.abs(determinant - 1) > 1e-5)
      throw new Error('Target rotation must be a proper orthonormal 3x3 matrix.');
  }

  function unpackTransform(t) {
    const flat =
      t && t.length === 4 && t[0] && t[0].length === 4
        ? Array.from(t[0]).concat(Array.from(t[1]), Array.from(t[2]), Array.from(t[3]))
        : t;
    const v = finiteVector(flat, 16, 'Transform');
    return { r: [v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]], p: [v[3], v[7], v[11]] };
  }
  function prepareModel(model) {
    if (
      !model ||
      !model.origins ||
      model.origins.length !== 6 ||
      !model.axes ||
      model.axes.length !== 6
    )
      throw new Error('The pose solver requires the six-joint viewer model.');
    return {
      mount: unpackTransform(model.mount),
      origins: model.origins.map(unpackTransform),
      axes: model.axes.map((axis) => finiteVector(axis, 3, 'Joint axis')),
      centers: model.centers.map((center) => finiteVector(center, 3, 'Body COM')),
      masses: finiteVector(model.masses, 6, 'Body masses'),
      grasp: finiteVector(model.grasp, 3, 'Grasp point'),
    };
  }
  /*
   * Prepared models are local to one solve/forward call. Nothing is cached
   * between calls, so in-place rod/COM edits cannot leave stale kinematics.
   */
  function frames(q, m, withGravity) {
    let r = m.mount.r,
      p = m.mount.p;
    const pivots = [],
      axes = [],
      centers = [];
    for (let joint = 0; joint < 6; joint++) {
      p = translated(r, m.origins[joint].p, p);
      r = mul3(r, m.origins[joint].r);
      pivots.push(p);
      axes.push(rotateFlat(r, m.axes[joint]));
      r = mul3(r, axisRotationFlat(m.axes[joint], q[joint]));
      if (withGravity) centers.push(translated(r, m.centers[joint], p));
    }
    const tcp = translated(r, m.grasp, p);
    const result = { tcp, r, pivots, axes };
    if (withGravity) {
      const own = [],
        unit = [];
      for (let joint = 0; joint < 6; joint++) {
        let torque = 0;
        for (let body = joint; body < 6; body++) {
          const lever = subtract(centers[body], pivots[joint]);
          const force = [0, 0, -GRAVITY * m.masses[body]];
          torque += dot(cross(lever, force), axes[joint]);
        }
        own.push(torque);
        unit.push(dot(cross(subtract(tcp, pivots[joint]), [0, 0, -GRAVITY]), axes[joint]));
      }
      result.centers = centers;
      result.own = own;
      result.unit = unit;
      result.transform = [
        r[0],
        r[1],
        r[2],
        p[0],
        r[3],
        r[4],
        r[5],
        p[1],
        r[6],
        r[7],
        r[8],
        p[2],
        0,
        0,
        0,
        1,
      ];
    }
    return result;
  }

  /** Full forward pose plus signed gravity torques, matching the Python model. */
  function forward(q, model) {
    const pose = frames(finiteVector(q, 6, 'Joint angles'), prepareModel(model), true);
    pose.R = nestedRotation(pose.r);
    delete pose.r;
    return pose;
  }

  function errorAt(pose, targetP, targetR, rotationWeight) {
    const translation = subtract(targetP, pose.tcp);
    const rotation = logFlat(mul3(targetR, transposeFlat(pose.r)));
    const positionError = norm(translation),
      orientationError = norm(rotation);
    return {
      positionError,
      orientationError,
      residual: translation.concat(rotation.map((x) => x * rotationWeight)),
      cost:
        positionError * positionError +
        rotationWeight * rotationWeight * orientationError * orientationError,
    };
  }
  function jacobian(pose, rotationWeight) {
    const j = Array.from({ length: 6 }, () => new Array(6));
    for (let joint = 0; joint < 6; joint++) {
      const linear = cross(pose.axes[joint], subtract(pose.tcp, pose.pivots[joint]));
      for (let row = 0; row < 3; row++) {
        j[row][joint] = linear[row];
        j[row + 3][joint] = rotationWeight * pose.axes[joint][row];
      }
    }
    return j;
  }
  /*
   * Solve the positive-definite DLS normal matrix by Cholesky. Damping makes
   * singular geometric Jacobians solvable; it does not create a missing DOF.
   */
  function choleskySolve(matrix, rhs) {
    const l = Array.from({ length: 6 }, () => Array(6).fill(0));
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j <= i; j++) {
        let value = matrix[i][j];
        for (let k = 0; k < j; k++) value -= l[i][k] * l[j][k];
        if (i === j) {
          if (!(value > 0) || !Number.isFinite(value)) return null;
          l[i][j] = Math.sqrt(value);
        } else l[i][j] = value / l[j][j];
      }
    }
    const y = new Array(6),
      x = new Array(6);
    for (let i = 0; i < 6; i++) {
      let value = rhs[i];
      for (let j = 0; j < i; j++) value -= l[i][j] * y[j];
      y[i] = value / l[i][i];
    }
    for (let i = 5; i >= 0; i--) {
      let value = y[i];
      for (let j = i + 1; j < 6; j++) value -= l[j][i] * x[j];
      x[i] = value / l[i][i];
    }
    return x;
  }
  function dampedStep(j, error, damping) {
    const a = Array.from({ length: 6 }, () => Array(6).fill(0));
    const b = Array(6).fill(0);
    for (let col = 0; col < 6; col++) {
      for (let row = 0; row < 6; row++) b[col] += j[row][col] * error[row];
      for (let other = 0; other <= col; other++) {
        let value = col === other ? damping * damping : 0;
        for (let row = 0; row < 6; row++) value += j[row][col] * j[row][other];
        a[col][other] = a[other][col] = value;
      }
    }
    return choleskySolve(a, b);
  }
  function halton(index, base) {
    let value = 0,
      factor = 1;
    while (index > 0) {
      factor /= base;
      value += factor * (index % base);
      index = Math.floor(index / base);
    }
    return value;
  }
  function numericOption(options, name, fallback, minimum, maximum) {
    const value = options[name] === undefined ? fallback : options[name];
    if (!Number.isFinite(value) || value < minimum || value > maximum)
      throw new Error(name + ' must be in [' + minimum + ', ' + maximum + '].');
    return value;
  }

  /**
   * Solve BOTH target position and orientation with damped least squares.
   *
   * Keep targetRotation equal to forward(seedQ, model).R when dragging TCP.
   * Keep targetPosition equal to forward(seedQ, model).tcp when rotating tool.
   * The caller must display residual errors if converged is false.
   *
   * options:
   *  maxIterations: 60 per start (1..250)
   *  positionTolerance: .001 m; orientationTolerance: pi/180 rad
   *  rotationWeight: .15 m/rad in the least-squares residual
   *  damping: .01 initially, adapted with monotone backtracking
   *  maxStep: .3 rad per joint per accepted iteration
   *  maxTimeMs: 25 across all starts; 0 disables the wall-clock deadline
   *  restarts: 0; optional deterministic extra seeds (up to 16)
   *
   * Result errors are computed at returned q, whose equivalent continuous
   * angles are placed nearest seedQ. iterations counts updates across starts.
   * The solver returns the best attempted pose, not a fabricated target pose.
   */
  function solve(targetPosition, targetRotation, seedQ, model, options) {
    options = options || {};
    const targetP = finiteVector(targetPosition, 3, 'Target position');
    const targetR = flatRotation(targetRotation);
    validateRotation(targetR);
    const seed = finiteVector(seedQ, 6, 'Seed joint angles');
    const m = prepareModel(model);
    const maxIterations = Math.floor(numericOption(options, 'maxIterations', 60, 1, 250));
    const positionTolerance = numericOption(options, 'positionTolerance', 0.001, 1e-12, 1);
    const orientationTolerance = numericOption(
      options,
      'orientationTolerance',
      Math.PI / 180,
      1e-12,
      Math.PI
    );
    const weight = numericOption(options, 'rotationWeight', 0.15, 1e-5, 10);
    const initialDamping = numericOption(options, 'damping', 0.01, 1e-8, 10);
    const maxStep = numericOption(options, 'maxStep', 0.3, 1e-5, Math.PI);
    const maxTimeMs = numericOption(options, 'maxTimeMs', 25, 0, 10000);
    const restarts = Math.floor(numericOption(options, 'restarts', 0, 0, 16));
    const deadline = maxTimeMs > 0 ? now() + maxTimeMs : Infinity;
    const toleranceMet = (error) =>
      error.positionError <= positionTolerance && error.orientationError <= orientationTolerance;
    let bestQ = seed.slice(),
      bestPose = frames(seed, m, false);
    let best = errorAt(bestPose, targetP, targetR, weight);
    let converged = toleranceMet(best),
      iterations = 0,
      attempts = 0;
    let reason = converged ? 'tolerance' : 'iteration_limit';
    const primes = [2, 3, 5, 7, 11, 13];

    for (let attempt = 0; attempt <= restarts && !converged; attempt++) {
      if (now() >= deadline) {
        reason = 'time_limit';
        break;
      }
      attempts++;
      reason = 'iteration_limit';
      let q =
        attempt === 0
          ? seed.slice()
          : seed.map((angle, i) => angle + (2 * halton(attempt, primes[i]) - 1) * Math.PI);
      let pose = attempt === 0 ? bestPose : frames(q, m, false);
      let error = errorAt(pose, targetP, targetR, weight);
      let damping = initialDamping,
        rejected = 0;
      if (error.cost < best.cost) {
        best = error;
        bestQ = q.slice();
        bestPose = pose;
      }
      if (toleranceMet(error)) {
        best = error;
        bestQ = q.slice();
        bestPose = pose;
        converged = true;
        reason = 'tolerance';
        break;
      }
      for (let iteration = 0; iteration < maxIterations; iteration++) {
        if (now() >= deadline) {
          reason = 'time_limit';
          break;
        }
        iterations++;
        const j = jacobian(pose, weight);
        const step = dampedStep(j, error.residual, damping);
        if (!step) {
          damping = Math.min(10, damping * 4);
          if (++rejected >= 8) {
            reason = 'stalled';
            break;
          }
          continue;
        }
        const largest = Math.max.apply(null, step.map(Math.abs));
        const clamp = largest > maxStep ? maxStep / largest : 1;
        let accepted = false;
        for (let reduction = 0; reduction < 5; reduction++) {
          const scale = clamp * Math.pow(0.5, reduction);
          const candidate = q.map((angle, i) => angle + scale * step[i]);
          const candidatePose = frames(candidate, m, false);
          const candidateError = errorAt(candidatePose, targetP, targetR, weight);
          if (candidateError.cost < error.cost || toleranceMet(candidateError)) {
            q = candidate;
            pose = candidatePose;
            error = candidateError;
            accepted = true;
            damping = Math.max(initialDamping * 0.01, damping * 0.65);
            rejected = 0;
            break;
          }
        }
        if (error.cost < best.cost || toleranceMet(error)) {
          best = error;
          bestQ = q.slice();
          bestPose = pose;
        }
        if (toleranceMet(error)) {
          converged = true;
          reason = 'tolerance';
          break;
        }
        if (!accepted) {
          damping = Math.min(10, damping * 4);
          if (++rejected >= 8) {
            reason = 'stalled';
            break;
          }
        }
      }
      if (reason === 'time_limit') break;
    }

    // Changing by complete revolutions does not constrain these continuous joints.
    const q = bestQ.map(
      (angle, i) => seed[i] + Math.atan2(Math.sin(angle - seed[i]), Math.cos(angle - seed[i]))
    );
    const finalError = errorAt(frames(q, m, false), targetP, targetR, weight);
    converged = toleranceMet(finalError);
    return {
      q,
      converged,
      positionError: finalError.positionError,
      orientationError: finalError.orientationError,
      iterations,
      attempts,
      reason: converged ? 'tolerance' : reason,
    };
  }

  /**
   * Signed static interval abs(own + payloadKg*unit) <= cap for every joint.
   * feasible:false is different from feasible [0,0]. A positive lower bound
   * means payload counterbalance is needed. upper:Infinity means no finite
   * gravity-only bound. limitingJoint is a zero-based upper-bound joint index,
   * or null when the interval is impossible/unbounded.
   */
  function capacity(own, unit, caps) {
    if (!own || !own.length) throw new Error('At least one joint is required.');
    const a = finiteVector(own, own.length, 'Own gravity torque');
    const b = finiteVector(unit, own.length, 'Payload torque per kilogram');
    const c = finiteVector(caps, own.length, 'Torque limits');
    if (c.some((value) => value < 0)) throw new Error('Torque limits must be nonnegative.');
    let lower = 0,
      upper = Infinity,
      feasible = true,
      limitingJoint = null;
    const emptyArmFeasible = a.every((value, i) => Math.abs(value) <= c[i]);
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(b[i]) <= UNIT_TOLERANCE) {
        if (Math.abs(a[i]) > c[i]) feasible = false;
      } else {
        const first = (-c[i] - a[i]) / b[i],
          second = (c[i] - a[i]) / b[i];
        lower = Math.max(lower, Math.min(first, second));
        const bound = Math.max(first, second);
        if (bound < upper) {
          upper = bound;
          limitingJoint = i;
        }
      }
    }
    feasible = feasible && upper >= lower;
    return {
      lower,
      upper,
      feasible,
      limitingJoint: feasible ? limitingJoint : null,
      emptyArmFeasible,
    };
  }

  return {
    forward,
    solve,
    capacity,
    multiply3,
    transpose3,
    axisRotation,
    rotationLog,
    cross,
    dot,
    subtract,
    constants: { gravity: GRAVITY, unitTorqueTolerance: UNIT_TOLERANCE },
  };
});
