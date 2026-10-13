"""Build the offline animated workspace viewer from the validated numerical maps."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from assess_rod_payload import RodVariants

from estimate_payload import ArmGravity
from payload_analysis.page_data import verify_map_assets, worked_examples, gripper_visual_frame, write_map_csv

HERE = Path(__file__).resolve().parent
from payload_analysis.paths import arm_urdf
MODES = ('any_orientation', 'downward_15deg')


def encode(array, dtype):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')


def neighbor_capacities(data, width):
    """Minimum of the six adjacent cells, used only to render a sampled surface."""
    n = round(2 / width)
    cells = np.rint((data['voxel_center_m'] + 1) / width - .5).astype(int)
    dense = np.full((n + 2,) * 3, -1.)
    indices = cells + 1
    dense[tuple(indices.T)] = data['max_static_payload_kg']
    values = []
    for axis in range(3):
        for step in (-1, 1):
            adjacent = indices.copy()
            adjacent[:, axis] += step
            values.append(dense[tuple(adjacent.T)])
    return np.min(values, axis=0)


def write_viewer(data, report, output):
    source = arm_urdf()
    model = ArmGravity(source)
    verify_map_assets(model, source)
    packed = {}
    for mode in MODES:
        d = data[mode]
        # Preserve full precision for capacities and joint angles. GPU display positions
        # use the same float32 values as the validated NPZ; no rounded CSV is reimported.
        packed[mode] = dict(count=len(d['max_static_payload_kg']),
                            xyz=encode(d['sampled_tcp_m'], '<f4'),
                            capacity=encode(d['max_static_payload_kg'], '<f8'),
                            angles=encode(d['joint_angles_rad'], '<f8'),
                            neighbor=encode(neighbor_capacities(d, report['voxel_size_m']), '<f4'))
    meta = dict(samples=report['samples'], voxel=report['voxel_size_m'],
                model={key: getattr(model, key).tolist() for key in
                       ('mount', 'origins', 'axes', 'masses', 'centers', 'grasp', 'caps')})
    meta['model']['gripper_visual_axes'] = gripper_visual_frame(source, model)
    meta['analysis'] = worked_examples(HERE, model)
    variants = RodVariants()
    design_rods = []
    for spec in variants.rods:
        rod = variants.parts[spec['tube']]
        distal_mass = sum(part['mass'] for part in variants.parts.values()
                          if part['body'] == spec['body'] and part['allocation'] in spec['distal'])
        design_rods.append(dict(joint=spec['joint'], direction=spec['direction'].tolist(),
                                rod_mass=rod['mass'], rod_center=rod['center'].tolist(), distal_mass=distal_mass))
    # Compact deterministic Sobol samples; evaluate the quantized angles themselves.
    sample_codes = np.floor(qmc.Sobol(6, scramble=True, seed=20261008).random_base2(18)*65536).astype('<u2')
    meta['design'] = dict(rods=design_rods, count=len(sample_codes), voxel=.05,
                          samples=encode(sample_codes, '<u2'))
    html = (HERE/'payload_viewer/template.html').read_text()
    assets = HERE/'payload_viewer'
    for name, text in {'__CSS__': (assets/'style.css').read_text()+'\n'+(assets/'shell.css').read_text(),
                       '__METHOD_HTML__': (assets/'methods.html').read_text(),
                       '__METHOD_CSS__': (assets/'methods.css').read_text(),
                       '__METHOD_JS__': (assets/'methods.js').read_text(),
                       '__ARM_HTML__': (assets/'manipulator.html').read_text(),
                       '__ARM_CSS__': (assets/'manipulator.css').read_text(),
                       '__ARM_JS__': (assets/'manipulator.js').read_text(),
                       '__SHELL_JS__': (assets/'shell.js').read_text(),
                       '__POSE_SOLVER_JS__': (assets/'pose_solver.js').read_text(),
                       '__JS__': (HERE/'payload_viewer/viewer.js').read_text(),
                       '__DESIGN_JS__': (HERE/'payload_viewer/rod_controls.js').read_text(),
                       '__WORKER__': (HERE/'payload_viewer/rod_worker.js').read_text(),
                       '__DATA__': json.dumps(packed, separators=(',', ':')),
                       '__META__': json.dumps(meta, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')}.items():
        if name not in html:
            raise ValueError(f'Missing page template slot: {name}')
        html = html.replace(name, text)
    output.write_text(html)
    print(f'Built {output} ({output.stat().st_size / 1e6:.1f} MB, self-contained)')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'workspace_payload')
    args = parser.parse_args()
    data = {mode: dict(np.load(args.input/f'{mode}.npz')) for mode in MODES}
    report = json.loads((args.input/'summary.json').read_text())
    write_viewer(data, report, args.input/'payload_workspace.html')
    # The page links these downloads; they are generated from the saved maps.
    for mode in MODES:
        write_map_csv(args.input/f'{mode}.csv', data[mode])
    from plot_payload_workspace import plot_chart
    plot_chart(data, report, args.input)
    print(f'Wrote {", ".join(f"{mode}.csv" for mode in MODES)} and payload_workspace.png/.pdf')
    if (HERE/'workspace_payload/redesign_validation.json').exists():
        from build_redesign_viewer import write_redesign_viewer
        write_redesign_viewer(args.input/'redesign_options.html')


if __name__ == '__main__':
    main()
