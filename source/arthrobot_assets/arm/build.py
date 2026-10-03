"""Build the arm's simulation URDF from its Onshape export.

The raw export cannot be simulated as an arm:

- 562 of its 617 part meshes hang on the base link;
- each of its six revolute joints turns one gear or planet carrier inside a motor;
- the first jaw slider makes the moving jaw the parent of its rail;
- 82 meshes belong to a detached spare motor;
- its link masses total about 1 g.

This builder assigns every mesh to one of ten rigid bodies (seven arm segments,
two jaws and the gripper pinion), rebuilds the joints on the original shaft
axes, assigns masses from the motor and part catalogs, and checks that every
mesh keeps its CAD pose. The export is pinned by SHA-256: a new export stops the
build until the part grouping below has been reviewed against it.

Usage: ``python -m arthrobot_assets.arm.build`` writes ``build/arm/``.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from arthrobot import paths
from arthrobot.gripper import DM4310_GRIPPER
from arthrobot.mass import MassProperties, combine, distribute_by_volume, is_physical_inertia
from arthrobot.motors import MotorSpec, load_motor
from arthrobot.parts import load_parts
from arthrobot.urdf import (forward_kinematics, joint_child, joint_parent, mesh_scale, origin_matrix,
                            resolve_package_path, set_origin, write_inertial, write_urdf)
from arthrobot_assets.arm import (ARM_BODIES, ARM_JOINTS, BODIES, CAD_DIR, JAW_JOINTS, PINION_JOINT,
                                  SOURCE_URDF, TOOL_BODY)

SOURCE_SHA256 = 'f28849727f310ffb695cfa0ebcd7f57c0f2dde1386cc67e03521916d9602e72c'
DEFAULT_MOTOR = 'mg5010'

# ------------------------------------------------------------------ reviewed part grouping
# The export stores each motor's internal parts as one contiguous block of 80 meshes on
# its root link. A block moves with the body holding that motor's housing. Joint n is
# driven by motor n; for joints 3-6 the housing sits on the downstream body.
MOTOR_BLOCKS = (
    # joint, body holding the housing, root-mesh block, housing link, output link
    (1, 'base', range(80, 160), 'mg4010_fc_1', '圆柱齿轮12_0_4_1'),
    (2, 'shoulder_output', range(482, 562), 'mg4010_fc_6', 'mg4010_pc_行星支架_6'),
    (3, 'elbow_output', range(240, 320), 'mg4010_fc_3', 'mg4010_pc_行星支架_3'),
    (4, 'forearm_output', range(402, 482), 'mg4010_fc_5', 'mg4010_pc_行星支架_5'),
    (5, 'wrist_output', range(160, 240), 'mg4010_fc_2', 'mg4010_pc_行星支架_2'),
    (6, 'tool_output', range(0, 80), 'mg4010_fc', 'mg4010_pc_行星支架'),
)
# Motor 1's output carrier lies in the base block but turns with joint 1.
JOINT_1_OUTPUT_CARRIER_MESH = 111
# A detached, unused seventh MG4010 near (-0.008, 0.172, -0.418) m.
SPARE_MOTOR_MESHES = range(320, 402)

# Named export links fastened to each arm body. Names missing from this export
# (parts of the previous gripper) are skipped.
BODY_LINKS = {
    'base': {'90deg_holder_3', 'mg4010_fc_1', '圆柱齿轮12_0_4_1'},
    'shoulder_output': {'90deg_holder_2', 'fixhex_1', 'mg4010_fc_6'},
    'upper_arm_output': {'150', 'center_connecter_1', 'rod_90deg_1', 'mg4010_pc_行星支架_6',
                         'mg4010_pc_行星支架_3'},
    'elbow_output': {'fixhex', '90deg_holder_4', 'mg4010_fc_3', 'mg4010_pc_行星支架_5'},
    'forearm_output': {'center_connecter', '150_1', 'rod_90deg', '90deg_holder_1', 'mg4010_fc_5',
                       'mg4010_pc_行星支架_2'},
    'wrist_output': {'90deg_holder_5', 'mg4010_fc_2', 'fixhex_2', 'mg4010_pc_行星支架'},
    'tool_output': {'90deg_holder', 'mg4010_fc', '90deg_holder_6', 'mg4010_fc_4', 'mr104zz_4', 'part_10',
                    'part_2', 'part_7', 'part_8', 'part_9', 'rack__20_teeth_', 'rack__20_teeth____rail',
                    'spur_gear__40_teeth_'},
}

SOURCE_ARM_JOINTS = tuple(f'revolute_{number}' for number in range(1, 7))
SOURCE_JAW_SLIDERS = ('dof_gripper_jaw_0_joint', 'dof_gripper_jaw_1_joint')
JAW_FASTENER_PREFIX = 'group_gripper_jaw_'      # fixed joints inside each jaw assembly
PINION_GEAR_PREFIX = '5_gear_1m16c_'
PINION_COUPLER_PREFIX = '2_m7_rotor_'
GRIPPER_MOTOR_PREFIX = '9_m_rs00_'              # CAD envelope of the gripper motor
JAW_CLIP_PREFIX = '1_clip_2'                    # clips on both jaws; their midpoint is the grasp center

EXPECTED_LINKS, EXPECTED_VISUALS, EXPECTED_ROOT_VISUALS = 61, 617, 562


@dataclass
class PlacedMesh:
    """One CAD part mesh, assigned to a body."""
    body: str
    mesh: trimesh.Trimesh
    in_world: np.ndarray          # pose at the CAD zero configuration
    in_body: np.ndarray           # pose in its body frame
    visual: ET.Element            # the output <visual> element
    allocation: str = ''
    mass_properties: MassProperties | None = None


def build(motor_name: str = DEFAULT_MOTOR, gripper_mass_kg: float | None = None) -> tuple[ET.Element, dict]:
    """Return the arm URDF tree and a build report.

    ``gripper_mass_kg`` is the measured mass of the whole gripper including its motor;
    by default the 300 g motor plus a 160 g mechanism estimate is used.
    """
    if hashlib.sha256(SOURCE_URDF.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('The CAD export changed; review the part grouping in arm/build.py first')
    motor = load_motor(motor_name)
    gripper_mass_is_estimate = gripper_mass_kg is None
    if gripper_mass_kg is None:
        gripper_mass_kg = DM4310_GRIPPER.total_mass_estimate_kg
    if gripper_mass_kg <= DM4310_GRIPPER.motor_mass_kg:
        raise ValueError('The whole gripper must weigh more than its 300 g motor')

    source = ET.parse(SOURCE_URDF).getroot()
    source_frames = forward_kinematics(source)
    links = {link.get('name'): link for link in source.findall('link')}
    assert len(links) == EXPECTED_LINKS and len(source.findall('.//visual')) == EXPECTED_VISUALS
    assert len(links['root'].findall('visual')) == EXPECTED_ROOT_VISUALS

    root_mesh_body = _root_mesh_bodies(links['root'])
    link_body, jaw_links, jaw_seeds, pinion_links = _assign_links_to_bodies(source, links)
    body_frames = _body_frames(source, source_frames, links, jaw_seeds, pinion_links[0])

    robot = ET.Element('robot', name='arthrobot_arm')
    body_elements = {body: ET.SubElement(robot, 'link', name=body) for body in BODIES}
    placed, excluded = _place_meshes(links, source_frames, body_frames, root_mesh_body, link_body, body_elements)
    allocations = _allocate_masses(placed, motor, gripper_mass_kg, gripper_mass_is_estimate)
    bodies = _write_body_inertials(placed, body_elements)
    closing_axes = _add_joints(robot, source, source_frames, body_frames, motor)
    _verify_geometry(robot, placed, closing_axes)

    tool_from_world = np.linalg.inv(body_frames[TOOL_BODY])
    clip_centers = [record.in_world[:3, :3] @ record.mesh.centroid + record.in_world[:3, 3]
                    for key, record in placed.items() if key.startswith(JAW_CLIP_PREFIX)]
    grasp_center = (tool_from_world @ np.r_[np.mean(clip_centers, axis=0), 1.])[:3]
    report = dict(
        source_sha256=SOURCE_SHA256, motor=motor.summary(),
        total_mass_kg=sum(body['mass_kg'] for body in bodies.values()), bodies=bodies, allocations=allocations,
        included_visuals=len(placed), excluded_visuals=excluded,
        excluded_reason='82 meshes of a detached, unused seventh MG4010 assembly.',
        jaw_parts=[sorted(names) for names in jaw_links], pinion_parts=pinion_links,
        gripper=DM4310_GRIPPER.summary(), gripper_mass_kg=gripper_mass_kg,
        gripper_mass_is_estimate=gripper_mass_is_estimate,
        pinion_frame_in_tool=(tool_from_world @ body_frames['gripper_pinion']).tolist(),
        grasp_center_in_tool_m=grasp_center.tolist(),
        jaw_closing_axis_in_tool=(body_frames[TOOL_BODY][:3, :3].T @ closing_axes[1]).tolist(),
        warnings=[*DM4310_GRIPPER.record['limitations'],
                  'Three silicone pad links in the export have no geometry; only one pad mesh exists.',
                  'Jaw travel is 0-50 mm per jaw (user-confirmed); pad contact stops closure slightly earlier.',
                  'MG5010 modules use the MG4010 CAD meshes as a geometry and mass-distribution placeholder.'],
        visual_manifest=[dict(source=key, body=record.body, name=record.visual.get('name'),
                              allocation=record.allocation) for key, record in placed.items()])
    return robot, report


def write(output_dir: Path | None = None, motor_name: str = DEFAULT_MOTOR,
          gripper_mass_kg: float | None = None) -> tuple[Path, dict]:
    """Build and write ``arm.urdf`` and ``model_report.json``; return the URDF path and report."""
    output_dir = output_dir or paths.build_dir('arm')
    robot, report = build(motor_name, gripper_mass_kg)
    urdf_path = write_urdf(robot, output_dir / 'arm.urdf')
    (output_dir / 'model_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return urdf_path, report


# ------------------------------------------------------------------ grouping
def _root_mesh_bodies(root_link: ET.Element) -> dict[int, str]:
    """Body of every root-link mesh index, excluding the spare motor."""
    visuals = root_link.findall('visual')
    # Spot-check that the export still has the spare motor where expected.
    assert visuals[323].find('geometry/mesh').get('filename').endswith('/MG4010_FC.stl')
    assert visuals[361].find('geometry/mesh').get('filename').endswith('/MG4010_PC_____.stl')
    bodies = {index: body for _, body, block, _, _ in MOTOR_BLOCKS for index in block}
    bodies[JOINT_1_OUTPUT_CARRIER_MESH] = 'shoulder_output'
    assert set(bodies) | set(SPARE_MOTOR_MESHES) == set(range(len(visuals)))
    assert not set(bodies) & set(SPARE_MOTOR_MESHES)
    return bodies


def _assign_links_to_bodies(source: ET.Element, links: dict[str, ET.Element]):
    """Map every named export link to its body; also return jaw link sets, jaw seed links, pinion links."""
    link_body = {name: body for body, names in BODY_LINKS.items() for name in names if name in links}
    # Each jaw is the connected component of its slider-side link over the jaw's fixed joints.
    neighbors = {name: set() for name in links}
    for joint in source.findall('joint'):
        if joint.get('type') == 'fixed' and joint.get('name').startswith(JAW_FASTENER_PREFIX):
            parent, child = joint_parent(joint), joint_child(joint)
            neighbors[parent].add(child)
            neighbors[child].add(parent)
    sliders = [source.find(f"joint[@name='{name}']") for name in SOURCE_JAW_SLIDERS]
    # The first slider is exported with the moving jaw as the parent of the rail.
    jaw_seeds = [joint_parent(sliders[0]), joint_child(sliders[1])]
    jaw_links = []
    for index, seed in enumerate(jaw_seeds):
        component, stack = set(), [seed]
        while stack:
            name = stack.pop()
            if name not in component:
                component.add(name)
                stack.extend(neighbors[name] - component)
        assert not component & set(link_body), 'A jaw part is already assigned to an arm body'
        jaw_links.append(component)
        link_body.update({name: f'jaw_{index}' for name in component})
    assert not jaw_links[0] & jaw_links[1]
    # Remaining new-gripper parts (rail, motor, brackets) are fastened to the tool body.
    reviewed = {name for names in BODY_LINKS.values() for name in names}
    for name in set(links) - {'root'} - reviewed:
        link_body.setdefault(name, TOOL_BODY)
    pinion_links = [next(name for name in links if name.startswith(PINION_GEAR_PREFIX)),
                    next(name for name in links if name.startswith(PINION_COUPLER_PREFIX))]
    link_body.update({name: 'gripper_pinion' for name in pinion_links})
    assert set(link_body) == set(links) - {'root'}
    return link_body, jaw_links, jaw_seeds, pinion_links


def _body_frames(source, source_frames, links, jaw_seeds, pinion_gear_link) -> dict[str, np.ndarray]:
    """World frame of every body at the CAD zero pose."""
    frames = {'base': np.eye(4)}
    for body, joint_name in zip(ARM_BODIES[1:], SOURCE_ARM_JOINTS):
        frames[body] = source_frames[joint_child(source.find(f"joint[@name='{joint_name}']"))]
    for index, seed in enumerate(jaw_seeds):
        frames[f'jaw_{index}'] = source_frames[seed]
    # The gear mesh origin is offset from its axis: recenter the pinion frame on the gear
    # axis (the mesh's local X), keeping the export's orientation.
    gear_visual = links[pinion_gear_link].find('visual')
    gear_mesh = trimesh.load_mesh(resolve_package_path(gear_visual.find('geometry/mesh').get('filename'), CAD_DIR))
    gear_center = np.mean(gear_mesh.bounds, axis=0)
    gear_center[1:] = 0.
    pinion_frame = source_frames[pinion_gear_link] @ origin_matrix(gear_visual.find('origin'))
    pinion_frame[:3, 3] += pinion_frame[:3, :3] @ gear_center
    frames['gripper_pinion'] = pinion_frame
    return frames


def _place_meshes(links, source_frames, body_frames, root_mesh_body, link_body, body_elements):
    """Copy every included visual into its body with its CAD pose preserved."""
    mesh_cache: dict[tuple, trimesh.Trimesh] = {}
    placed: dict[str, PlacedMesh] = {}
    excluded = []
    for link_name, link in links.items():
        for index, visual in enumerate(link.findall('visual')):
            key = f'{link_name}:{index}'
            if link_name == 'root' and index in SPARE_MOTOR_MESHES:
                excluded.append(key)
                continue
            body = root_mesh_body[index] if link_name == 'root' else link_body[link_name]
            mesh_element = visual.find('geometry/mesh')
            mesh_path = resolve_package_path(mesh_element.get('filename'), CAD_DIR).resolve()
            mesh = _load_solid(mesh_path, mesh_scale(mesh_element), mesh_cache)
            in_world = source_frames[link_name] @ origin_matrix(visual.find('origin'))
            in_body = np.linalg.inv(body_frames[body]) @ in_world
            output = copy.deepcopy(visual)
            output.set('name', f'part_{len(placed):03d}')
            set_origin(output, in_body)
            output.find('geometry/mesh').set('filename', str(mesh_path))
            for material in output.findall('material'):
                material.set('name', 'cad_material')
            body_elements[body].append(output)
            placed[key] = PlacedMesh(body, mesh, in_world, in_body, output)
    return placed, excluded


def _load_solid(path: Path, scale: np.ndarray, cache: dict) -> trimesh.Trimesh:
    key = (str(path), tuple(scale))
    if key not in cache:
        mesh = trimesh.load_mesh(path)
        mesh.apply_scale(scale)
        if not mesh.is_volume:
            raise ValueError(f'Mesh is not a closed solid: {path}')
        cache[key] = mesh
    return cache[key]


# ------------------------------------------------------------------ masses
def _allocate_masses(placed: dict[str, PlacedMesh], motor: MotorSpec, gripper_mass_kg: float,
                     gripper_mass_is_estimate: bool) -> list[dict]:
    """Give every mesh a share of a recorded component mass (uniform density per component)."""
    allocations = []

    def allocate(label: str, keys: list[str], mass_kg: float) -> None:
        assert keys and all(key in placed and not placed[key].allocation for key in keys), label
        shares = distribute_by_volume([(placed[key].mesh, placed[key].in_body) for key in keys], mass_kg)
        for key, share in zip(keys, shares):
            placed[key].mass_properties, placed[key].allocation = share, label
        allocations.append(dict(component=label, mass_kg=mass_kg, visuals=len(keys)))

    for joint, _, block, housing, output in MOTOR_BLOCKS:
        keys = [f'root:{index}' for index in block] + [f'{housing}:0', f'{output}:0']
        allocate(f'joint_{joint}_motor', keys, motor.mass_kg)
    catalog = load_parts()
    for key, record in placed.items():
        part = catalog.get(Path(record.visual.find('geometry/mesh').get('filename')).name)
        if part is not None and not record.allocation:
            allocate(key.split(':')[0], [key], part.mass_kg)
    gripper_motor = [key for key in placed if key.startswith(GRIPPER_MOTOR_PREFIX)]
    assert len(gripper_motor) == 1
    allocate('gripper_DM4310_motor', gripper_motor, DM4310_GRIPPER.motor_mass_kg)
    mechanism = sorted(key for key, record in placed.items() if not record.allocation)
    allocate('gripper_mechanism_estimated' if gripper_mass_is_estimate else 'gripper_mechanism_from_total',
             mechanism, gripper_mass_kg - DM4310_GRIPPER.motor_mass_kg)
    return allocations


def _write_body_inertials(placed: dict[str, PlacedMesh], body_elements: dict[str, ET.Element]) -> dict:
    bodies = {}
    for body, element in body_elements.items():
        properties = combine([record.mass_properties for record in placed.values() if record.body == body])
        assert is_physical_inertia(properties.inertia), body
        bodies[body] = dict(mass_kg=properties.mass, center_of_mass_m=properties.center.tolist(),
                            inertia_kg_m2=properties.inertia.tolist())
        write_inertial(element, properties)
    return bodies


# ------------------------------------------------------------------ joints
def _add_joints(robot, source, source_frames, body_frames, motor: MotorSpec) -> list[np.ndarray]:
    """Six continuous arm joints, two prismatic jaws and the coupled pinion; returns jaw closing axes."""
    def add_joint(name, joint_type, parent, child):
        joint = ET.SubElement(robot, 'joint', name=name, type=joint_type)
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        set_origin(joint, np.linalg.inv(body_frames[parent]) @ body_frames[child])
        return joint

    for index, (name, source_name) in enumerate(zip(ARM_JOINTS, SOURCE_ARM_JOINTS)):
        joint = add_joint(name, 'continuous', ARM_BODIES[index], ARM_BODIES[index + 1])
        # The body frame equals the source joint's child frame, so its axis carries over unchanged.
        joint.append(copy.deepcopy(source.find(f"joint[@name='{source_name}']/axis")))
        ET.SubElement(joint, 'limit', effort=str(motor.rated_torque_nm), velocity=str(motor.max_speed_rad_s))

    gripper = DM4310_GRIPPER
    closing_axes = []
    for index, (name, slider_name) in enumerate(zip(JAW_JOINTS, SOURCE_JAW_SLIDERS)):
        body = f'jaw_{index}'
        joint = add_joint(name, 'prismatic', TOOL_BODY, body)
        slider = source.find(f"joint[@name='{slider_name}']")
        slider_frame = source_frames[joint_parent(slider)] @ origin_matrix(slider.find('origin'))
        closing_axis = slider_frame[:3, :3] @ np.fromstring(slider.find('axis').get('xyz'), sep=' ')
        if index == 0:
            closing_axis = -closing_axis    # parent and child were swapped; keep the physical motion
        closing_axes.append(closing_axis)
        ET.SubElement(joint, 'axis', xyz=' '.join(map(str, body_frames[body][:3, :3].T @ closing_axis)))
        driven = index == 0
        ET.SubElement(joint, 'limit', lower='0', upper=str(gripper.jaw_stroke_m),
                      effort=str(gripper.jaw_force_limit_n if driven else 0.),
                      velocity=str(gripper.pinion_radius_m * gripper.max_speed_rad_s))
        if not driven:
            ET.SubElement(joint, 'mimic', joint=JAW_JOINTS[0], multiplier='1', offset='0')
    np.testing.assert_allclose(closing_axes[0], -closing_axes[1], atol=2e-5)

    joint = add_joint(PINION_JOINT, 'revolute', TOOL_BODY, 'gripper_pinion')
    # Jaw 0's rack is at +Y of the pinion and closes along -Z, so closing turns the pinion about -X.
    ET.SubElement(joint, 'axis', xyz='-1 0 0')
    ET.SubElement(joint, 'limit', lower='0', upper=str(gripper.full_stroke_pinion_angle_rad), effort='0',
                  velocity=str(gripper.max_speed_rad_s))
    ET.SubElement(joint, 'mimic', joint=JAW_JOINTS[0], multiplier=str(1 / gripper.pinion_radius_m), offset='0')
    return closing_axes


def _verify_geometry(robot: ET.Element, placed: dict[str, PlacedMesh], closing_axes) -> None:
    """Every mesh keeps its CAD pose; closing 10 mm moves both jaws equally and oppositely."""
    zero_pose = forward_kinematics(robot)
    for record in placed.values():
        np.testing.assert_allclose(zero_pose[record.body] @ origin_matrix(record.visual.find('origin')),
                                   record.in_world, atol=1e-12)
    closed = forward_kinematics(robot, {JAW_JOINTS[0]: .01})
    np.testing.assert_allclose(closed[TOOL_BODY], zero_pose[TOOL_BODY], atol=1e-12)
    for index, axis in enumerate(closing_axes):
        np.testing.assert_allclose(closed[f'jaw_{index}'][:3, 3] - zero_pose[f'jaw_{index}'][:3, 3],
                                   .01 * axis, atol=1e-12)


if __name__ == '__main__':
    urdf_path, build_report = write()
    print(json.dumps(dict(urdf=str(urdf_path), visuals=build_report['included_visuals'],
                          total_mass_kg=build_report['total_mass_kg']), indent=2))
