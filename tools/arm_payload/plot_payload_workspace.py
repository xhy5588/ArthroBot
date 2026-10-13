"""Render the sampled payload workspace as PNG/PDF and a standalone interactive HTML viewer."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
COLORS = ['#8795a5', '#347be0', '#15a18c', '#e5a92c', '#b15dc9']
LABELS = ['Below 0.5 kg', '0.5–1 kg', '1–1.5 kg', '1.5–2 kg', 'At least 2 kg']
MODES = ['any_orientation', 'downward_15deg']


def plot_chart(data, report, output):
    """Write the printable chart, payload_workspace.png and .pdf, into ``output``."""
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm([0, .5, 1, 1.5, 2, 1e20], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for col, mode in enumerate(MODES):
        d = data[mode]
        p, cap = d['sampled_tcp_m'], d['max_static_payload_kg']
        # A radius/height projection explicitly maximizes over azimuth as well as posture.
        r = np.linalg.norm(p[:, :2], axis=1)
        width = report['voxel_size_m']
        ri = np.floor(r/width).astype(int)
        zi = np.floor((p[:, 2]+1.)/width).astype(int)
        # Rows span z in [-1, 1) m and columns r in [0, 0.8) m, at the map's cell size.
        projection = np.full((round(2/width), math.ceil(.8/width - 1e-9)), np.nan)
        keep = (ri < projection.shape[1]) & (zi >= 0) & (zi < projection.shape[0])
        ri, zi, cap = ri[keep], zi[keep], cap[keep]
        temp = np.full_like(projection, -np.inf)
        np.maximum.at(temp, (zi, ri), cap)
        projection[np.isfinite(temp)] = temp[np.isfinite(temp)]
        ax = axes[0, col]
        ax.imshow(projection, origin='lower', extent=[0, projection.shape[1]*width, -1, 1], aspect='equal',
                  cmap=cmap, norm=norm, interpolation='nearest')
        ax.axhline(0, color='#526070', linewidth=.8, linestyle='--')
        ax.plot(0, 0, 'ko', ms=5)
        ax.set(xlim=(0, .7), ylim=(-.60, .80), xlabel='Horizontal distance from base axis (m)',
               ylabel='Height relative to mounting plane (m)',
               title=('Any gripper direction' if col == 0 else 'Gripper pointing down (within 15°)'))
        ax.grid(alpha=.15)
        # A genuine thin horizontal slice, not a projection through the full workspace.
        ax = axes[1, col]
        keep = (p[:, 2] >= .10) & (p[:, 2] < .125)
        ax.scatter(p[keep, 0], p[keep, 1], c=cap[keep], cmap=cmap, norm=norm, s=15, marker='s', linewidths=0)
        ax.plot(0, 0, 'ko', ms=5)
        for radius in (.2, .4, .6):
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, edgecolor='#bcc5ce', linewidth=.7))
        ax.set(xlim=(-.7, .7), ylim=(-.7, .7), xlabel='X (m)', ylabel='Y (m)',
               title='Top view: TCP height 10–12.5 cm above mounting plane', aspect='equal')
    fig.suptitle('ARM_REBOT · reachable positions and static payload capacity', fontsize=18, weight='bold', y=.985)
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS, LABELS)],
               loc='upper center', bbox_to_anchor=(.5, .954), ncol=5, frameon=False)
    fig.text(.5, .035, 'Upper panels: best sampled posture and azimuth at each radius/height. Lower panels: a thin height slice.\n'
             f'{report["samples"]:,} joint samples • {report["voxel_size_m"]*100:g} cm cells • 13 N·m per joint • self-collision screened with approximate spheres\n'
             'Base fixed upright; payload COM at TCP. No floor/table, load geometry, acceleration, grip or strength checks.',
             ha='center', va='center', fontsize=10, color='#455260')
    fig.tight_layout(rect=(0, .07, 1, .92))
    fig.savefig(output/'payload_workspace.png', dpi=160)
    fig.savefig(output/'payload_workspace.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE/'workspace_payload')
    args = parser.parse_args()
    report = json.loads((args.input/'summary.json').read_text())
    data = {mode: dict(np.load(args.input/f'{mode}.npz')) for mode in MODES}
    plot_chart(data, report, args.input)
    from build_payload_viewer import write_viewer
    write_viewer(data, report, args.input/'payload_workspace.html')
    print('Saved PNG, PDF and interactive HTML:', args.input)


if __name__ == '__main__':
    main()
