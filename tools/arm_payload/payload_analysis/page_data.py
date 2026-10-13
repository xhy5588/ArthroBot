"""Build provenance-checked, current-model examples for the offline explainer."""
import csv
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

from .model import origin
from .paths import OUTPUT_DIR

FINGERPRINT = OUTPUT_DIR / 'model_fingerprint.json'


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_map_assets(model, urdf):
    """Reject maps made for a different arm model.

    The saved map, rod studies and examples were computed with the original ARM_REBOT
    files, whose hashes the reports keep. ArthroBot builds the same arm from the same
    CAD export, under other joint names, so the check is numerical: every mass, center
    of mass, joint origin and axis, the TCP, the mount and the motor caps must equal the
    recorded model (model_fingerprint.json). The map's self-collision screen used the
    original cuRobo sphere model, which has no counterpart to compare here.
    """
    expected = json.loads(FINGERPRINT.read_text())['model']
    for key in ('mount', 'origins', 'axes', 'masses', 'centers', 'grasp', 'caps'):
        np.testing.assert_allclose(np.asarray(getattr(model, key)), np.asarray(expected[key]), rtol=0, atol=1e-9,
                                   err_msg=f'The arm model ({key}) differs from the model the map was computed with; '
                                           'regenerate the workspace map first.')
    np.testing.assert_allclose(model.base_mass, expected['base_mass'], rtol=0, atol=1e-9)
    frame = gripper_visual_frame(urdf, model)
    for axis, vector in expected['gripper_visual_axes'].items():
        np.testing.assert_allclose(frame[axis], vector, rtol=0, atol=1e-9,
                                   err_msg=f'The gripper jaw frame ({axis}) differs from the recorded model.')


def worked_examples(root, model):
    """Use saved witness angles, but recompute every displayed torque and limit."""
    report = json.loads((root/'payload_static_current.json').read_text())
    examples = {}
    for label, key in (('zero', 'cad_pose'), ('extended', 'most_demanding_found')):
        previous = report['mounts']['vertical'][key]
        current = model.describe(np.deg2rad(previous['joint_angles_deg']))
        for field in ('payload_kg', 'own_gravity_torque_nm', 'payload_torque_per_kg_nm'):
            np.testing.assert_allclose(current[field], previous[field], rtol=1e-9, atol=1e-9,
                                       err_msg=f'The saved {label} example no longer matches the calculation.')
        examples[label] = current
    return dict(
        rod_lengths_mm=[150, 150], motor_caps_nm=model.caps.tolist(),
        total_arm_mass_kg=float(model.base_mass+model.masses.sum()),
        moving_arm_mass_kg=float(model.masses.sum()),
        reach_bound_m=float(np.linalg.norm(model.origins[1:, :3, 3], axis=1).sum()+np.linalg.norm(model.grasp)),
        examples=examples,
    )


def gripper_visual_frame(urdf, model):
    """Orient schematic fingers using the actual tool-frame jaw slide direction.

    Shape and dimensions remain schematic. The finger closing axis comes from
    the jaw's URDF joint frame; it is not one of the tool link's principal axes.
    """
    root = ET.parse(urdf).getroot()
    slides = []
    for name in ('gripper_jaw_0', 'gripper_jaw_1'):
        joint = root.find(f"joint[@name='{name}']")
        if joint is None or joint.find('parent').get('link') != 'tool_output':
            raise ValueError(f'{name} must be a direct child of tool_output.')
        axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        slide = origin(joint.find('origin'))[:3, :3] @ axis
        slides.append(slide / np.linalg.norm(slide))
    if not np.allclose(slides[0], -slides[1], atol=1e-8):
        raise ValueError('The schematic expects parallel, opposing jaw travel.')
    forward = model.grasp / np.linalg.norm(model.grasp)
    side = slides[0] - forward * np.dot(forward, slides[0])
    side /= np.linalg.norm(side)
    top = np.cross(forward, side)
    return {name: vector.tolist() for name, vector in
            (('forward', forward), ('side', side), ('top', top))}


def write_map_csv(path, data):
    """The sampled map as CSV (one row per cell), from the arrays the map script saves in its NPZ."""
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['voxel_x_m', 'voxel_y_m', 'voxel_z_m', 'tcp_x_m', 'tcp_y_m', 'tcp_z_m',
                         'max_static_payload_kg', 'samples_in_cell'] + [f'joint_{i}_deg' for i in range(1, 7)])
        for c, p, m, n, q in zip(data['voxel_center_m'], data['sampled_tcp_m'], data['max_static_payload_kg'],
                                 data['sample_count'], data['joint_angles_rad']):
            writer.writerow([*np.round(c, 6), *np.round(p, 7), round(m, 6), n, *np.round(np.rad2deg(q), 5)])
