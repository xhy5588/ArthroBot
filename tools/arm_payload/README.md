# Arm payload calculator

A static payload calculator for the ArthroBot arm: for any posture, how much the
gripper can hold and which motor limits it. Try it online at
https://xhy5588.github.io/ArthroBot/. What it shows, why it matters, and how the
numbers are computed: [docs/arm_payload.md](../../docs/arm_payload.md).

Run from the repository root with any Python that has NumPy, SciPy, trimesh,
Matplotlib and the ArthroBot packages (Isaac Sim does not need to run):

```bash
python tools/arm_payload/build_payload_viewer.py                 # build the pages, CSVs and chart
xdg-open tools/arm_payload/workspace_payload/payload_workspace.html
python -m pytest tests/test_arm_payload.py                        # statics and the frozen reference
python tools/arm_payload/payload_viewer/check_manipulator.py      # one of the browser checks
```

The generated HTML is self-contained and works offline. Edit the sources below and
rebuild; never edit the generated HTML. The browser checks (`payload_viewer/check_*.py`)
need Chrome or Chromium and Python's `websockets`; they run headless with a temporary
profile.

## Files

| Location | Responsibility |
| --- | --- |
| `payload_analysis/paths.py` | Tool folder; ArthroBot's arm build (`build/arm/arm.urdf`, `model_report.json`) |
| `payload_analysis/model.py` | URDF transforms, body/COM aggregation, gravity model, pose descriptions |
| `payload_analysis/statics.py` | Pure torque and signed payload-interval calculations |
| `payload_analysis/page_data.py` | Model check against `model_fingerprint.json`, worked examples, map CSV export |
| `payload_analysis/verify.py` | Numerical checks (also run by `tests/test_arm_payload.py`) |
| `estimate_payload.py` | Command-line static estimate for reference poses; `ArmGravity` import |
| `map_payload_workspace.py` | Sobol sampling, cuRobo self-collision screening, voxel aggregation (GPU) |
| `assess_rod_payload.py` | Rod-length changes with the corresponding mass and COM shifts |
| `assess_build_volume.py`, `assess_redesigns.py`, `assess_cube_payload.py` | Build-volume and redesign studies |
| `build_payload_viewer.py`, `build_redesign_viewer.py`, `plot_payload_workspace.py` | Page, CSV and chart builders |
| `payload_viewer/template.html`, `style.css`, `shell.css`, `shell.js` | Page shell, rod-length dialog, sections, sliders |
| `payload_viewer/manipulator.html`, `manipulator.css`, `manipulator.js` | Arm studio: pose editor and live torque readout |
| `payload_viewer/pose_solver.js` | Forward kinematics, damped least-squares IK, signed payload interval |
| `payload_viewer/viewer.js` | Reach map: rendering, filters, pose inspection |
| `payload_viewer/rod_controls.js`, `rod_worker.js` | Rod-length controls and their worker calculation |
| `payload_viewer/methods.html`, `methods.css`, `methods.js` | "How it works": equations and worked examples |
| `payload_viewer/redesign_*.html`, `redesign.*` | Alternative-design page |
| `payload_viewer/browser_check.py`, `check_*.py` | Isolated browser harness and page checks |

## Data in `workspace_payload/`

Kept in git (inputs; the map takes a GPU run to regenerate):
- `any_orientation.npz`, `downward_15deg.npz` and `summary.json`: the reach map, 67,108,864
  sampled postures in 2.5 cm cells, with any gripper angle or pointing down within 15°.
- `model_fingerprint.json`: the arm model the map and studies were computed with; the
  build and tests compare ArthroBot's model with it.
- `rod_length_payload.json`, `cube_payload_limits.json`, `build_volume_feasibility.json`,
  `redesign_validation.json`, `sampling_convergence.json`, `original_position_example.json`:
  study results; the redesign page is built from `redesign_validation.json`.
- `../payload_static_current.json`: the frozen reference poses (`estimate_payload.py`).
- `../payload_2kg_requirements.json`: the joint torques a 2 kg payload needs at the reference
  poses and at the hardest postures found per joint (a study record; no script here writes it).

Generated and git-ignored: the two HTML pages, the map CSVs, the printable chart
(`payload_workspace.png`/`.pdf`), and the browser checks' reports and screenshots.

## Regenerating the map

`map_payload_workspace.py --power 26` needs cuRobo, PyTorch and CUDA (about 5 minutes on
a GPU). It screens self-collisions with ArthroBot's cuRobo sphere model
(`scripts/arm/build_curobo_robot.py`); the included map used the original ARM_REBOT
sphere model. Rebuild the page afterwards.
