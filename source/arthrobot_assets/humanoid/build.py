"""Build the humanoid's simulation URDF from its Onshape export.

Like the arm export, the humanoid export describes the CAD assembly rather than
a moving robot: most motor parts hang on the root link and each exported
revolute joint turns one internal motor part. This builder:

1. finds the fastened groups of parts (fixed-joint components) and assigns each
   loose motor-internal mesh to its nearest motor;
2. rebuilds the 20 motor joints on the original shaft axes, walking each limb
   chain out from the torso (4 chains: two arms, two legs ending in wheels);
3. adds both rack-and-pinion grippers (driven jaw, mimic jaw, mimic pinion);
4. assigns masses from the motor and part catalogs, replaces hidden motor
   internals by one convex exterior collider per motor half, and checks that
   every mesh keeps its CAD pose.

The export is pinned by SHA-256. Z is up; the CAD zero pose is kept, which is
not a standing posture (see :mod:`.training_model` and :mod:`.nominal_pose`).

Usage: ``python -m arthrobot_assets.humanoid.build`` writes ``build/humanoid/``.
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
from arthrobot.mass import MassProperties, combine, distribute_by_volume
from arthrobot.motors import load_motor
from arthrobot.parts import load_parts
from arthrobot.urdf import (forward_kinematics, joint_child, joint_parent, mesh_scale, origin_matrix,
                            resolve_package_path, set_origin, write_inertial, write_urdf)
from arthrobot_assets.humanoid import CAD_DIR, HARDWARE_FILE, SIDES, SOURCE_URDF

SOURCE_SHA256 = '542efd40efe05ed82665124a083bb08ebfcd4d2aa161ac7f50300eb228e371d2'
EXPECTED_LINKS, EXPECTED_VISUALS = 178, 2419

# Exported joint names along each chain, from the torso outwards (reviewed).
CHAINS = {
    'left_arm': ('revolute_1_2', 'revolute_2_2', 'revolute_3_1', 'revolute_4_1', 'revolute_5_2', 'revolute_6_2'),
    'right_arm': ('revolute_1', 'revolute_2', 'revolute_3', 'revolute_4', 'revolute_5', 'revolute_6'),
    'left_leg': ('revolute_1_1', 'revolute_4_2', 'revolute_5_1', 'revolute_7'),
    'right_leg': ('revolute_2_1', 'revolute_3_2', 'revolute_6_1', 'revolute_8'),
}
LEG_JOINT_PARTS = ('hip_1', 'hip_2', 'knee', 'wheel')

TORSO_LINK = 'centerboard_w_battery'
MOTOR_HOUSING_PREFIX = 'mg4010_fc'           # housing link of each motor
MOTOR_OUTPUT_PREFIXES = ('mg4010_pc', '圆柱齿轮')  # planet carrier, or the output gear of a hip motor
MOTOR_HOUSING_MESH = 'MG4010_FC.stl'
MOTOR_OUTPUT_CARRIER_MESH = 'MG4010_PC_____.stl'
LOOSE_WHEEL_MESH = 'Part_1.stl'
MAX_MOTOR_PART_DISTANCE_M = .045             # a loose motor part lies within this of its motor center
EXPECTED_MOTORS_IN_EXPORT = 28               # 20 jointed + 8 unused in the spare assembly
TORSO_BOARD_MESH, WHEEL_MESH = 'CenterBoard_w_battery.stl', 'Part_1.stl'
GRIPPER_MOTOR_PREFIX = '9_m_rs00'
RAIL_PREFIX, PINION_GEAR_PREFIX, PINION_COUPLER_PREFIX, RACK_PREFIX = '6_rail_170', '5_gear_', '2_m7_rotor', '2_rack'

EXCLUDED_REASON = ('Disconnected spare assembly: a third torso module, six unjointed motors with their '
                   'brackets and rods, two loose wheels and two motors from the previous gripper.')


@dataclass
class MotorJoint:
    """An exported motor joint, re-expressed as housing and output sides."""
    housing: str            # housing link name (mg4010_fc...)
    output: str             # link on the output side
    marker: str             # the motor-internal output link the export jointed
    housing_component: int
    output_component: int
    frame: np.ndarray       # joint frame in the world at the CAD pose
    axis: np.ndarray        # joint axis in the world
    shaft_alignment: float
    shaft_offset_m: float


@dataclass
class PlacedMesh:
    body: str
    mesh: trimesh.Trimesh
    in_world: np.ndarray
    in_body: np.ndarray
    visual: ET.Element
    allocation: str = ''
    mass_properties: MassProperties | None = None


def build(collision_dir: Path | None = None) -> tuple[ET.Element, dict]:
    """Return the humanoid URDF tree and a build report.

    Motor exterior collision hulls are written as STL files into ``collision_dir``
    (default ``build/humanoid/collision_meshes``).
    """
    if hashlib.sha256(SOURCE_URDF.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('The CAD export changed; review the chains and grouping in humanoid/build.py first')
    hardware = json.loads(HARDWARE_FILE.read_text())
    motor = load_motor(hardware['motor'])
    collision_dir = collision_dir or paths.build_dir('humanoid') / 'collision_meshes'

    source = ET.parse(SOURCE_URDF).getroot()
    source_frames = forward_kinematics(source)
    links = {link.get('name'): link for link in source.findall('link')}
    assert len(links) == EXPECTED_LINKS and len(source.findall('.//visual')) == EXPECTED_VISUALS
    meshes = _MeshLibrary()

    components, component_of = _fastened_components(source, links)
    root_visuals = links['root'].findall('visual')
    motor_blocks = _group_loose_motor_parts(links, root_visuals, source_frames, meshes)
    motor_joints = _motor_joints(source, source_frames, component_of)

    # ---- bodies along each chain
    body_of_component = {component_of[TORSO_LINK]: 'torso'}
    body_frames = {'torso': np.eye(4)}
    joints = []
    for chain, source_names in CHAINS.items():
        current_component, parent_body = component_of[TORSO_LINK], 'torso'
        for number, source_name in enumerate(source_names, 1):
            spec = motor_joints[source_name]
            assert current_component in (spec.housing_component, spec.output_component)
            child_component = (spec.output_component if current_component == spec.housing_component
                               else spec.housing_component)
            assert child_component not in body_of_component
            side, limb = chain.split('_')
            name = f'{chain}_{number}' if limb == 'arm' else f'{side}_{LEG_JOINT_PARTS[number - 1]}'
            body = name + '_body'
            body_of_component[child_component] = body
            body_frames[body] = spec.frame
            joints.append(dict(name=name, source=source_name, parent=parent_body, child=body, type='continuous',
                               axis_world=spec.axis.tolist(), motor_housing=spec.housing,
                               shaft_alignment=spec.shaft_alignment, shaft_offset_m=spec.shaft_offset_m))
            parent_body, current_component = body, child_component
    assert len(body_of_component) == 21 and len(joints) == 20

    link_body = {name: body for component, body in body_of_component.items() for name in components[component]}
    root_visual_body = {}
    for spec in motor_joints.values():
        housing_body = body_of_component[spec.housing_component]
        output_body = body_of_component[spec.output_component]
        root_visual_body.update({index: housing_body for index in motor_blocks[spec.housing]})
        link_body[spec.marker] = housing_body if spec.marker.startswith('圆柱齿轮') else output_body
        for index in motor_blocks[spec.housing]:
            if _mesh_name(root_visuals[index]) == MOTOR_OUTPUT_CARRIER_MESH:
                root_visual_body[index] = output_body

    grippers = {side: _add_gripper(side, source, source_frames, links, components, component_of, link_body,
                                   body_frames, joints, meshes) for side in SIDES}

    # ---- meshes, masses, colliders
    robot = ET.Element('robot', name='arthrobot_humanoid')
    body_elements = {body: ET.SubElement(robot, 'link', name=body) for body in body_frames}
    placed, excluded = {}, []
    for link_name, link in links.items():
        for index, visual in enumerate(link.findall('visual')):
            key = f'{link_name}:{index}'
            body = root_visual_body.get(index) if link_name == 'root' else link_body.get(link_name)
            if body is None:
                excluded.append(key)
                continue
            in_world = source_frames[link_name] @ origin_matrix(visual.find('origin'))
            in_body = np.linalg.inv(body_frames[body]) @ in_world
            output = copy.deepcopy(visual)
            output.set('name', f'part_{len(placed):04d}')
            set_origin(output, in_body)
            mesh_element = output.find('geometry/mesh')
            mesh_element.set('filename', str(meshes.path(visual)))
            for material in output.findall('material'):
                material.set('name', 'cad_' + material.get('name').replace('.', '_'))
            body_elements[body].append(output)
            placed[key] = PlacedMesh(body, meshes.solid(visual), in_world, in_body, output)

    allocations = _allocate_masses(placed, motor_joints, motor_blocks, grippers, motor, hardware)
    collision_count, motor_exterior_count = _add_colliders(placed, body_elements, collision_dir)
    bodies = {}
    for body, element in body_elements.items():
        properties = combine([record.mass_properties for record in placed.values() if record.body == body])
        assert np.linalg.eigvalsh(properties.inertia).min() > 0
        bodies[body] = dict(mass_kg=properties.mass, center_of_mass_m=properties.center.tolist(),
                            inertia_kg_m2=properties.inertia.tolist())
        write_inertial(element, properties)
    _add_joint_elements(robot, joints, body_frames, grippers, motor)

    zero_pose = forward_kinematics(robot)
    max_error = max(float(np.abs(zero_pose[record.body] @ origin_matrix(record.visual.find('origin'))
                                 - record.in_world).max()) for record in placed.values())
    assert max_error < 1e-10
    assert len(bodies) == 27 and len(joints) == 26
    report = dict(
        source_sha256=SOURCE_SHA256, source_visuals=EXPECTED_VISUALS, included_visuals=len(placed),
        collision_mesh_count=collision_count, motor_exterior_colliders=motor_exterior_count,
        excluded_visuals=excluded, excluded_reason=EXCLUDED_REASON, joints=joints, grippers=grippers,
        bodies=bodies, allocations=allocations, total_mass_kg=sum(body['mass_kg'] for body in bodies.values()),
        motor=dict(**motor.summary(), count=20), hardware=hardware, gripper=DM4310_GRIPPER.summary(),
        geometry_preservation_max_error=max_error,
        part_ownership={key: dict(body=record.body, allocation=record.allocation,
                                  mesh=Path(record.visual.find('geometry/mesh').get('filename')).name)
                        for key, record in placed.items()},
        motor_blocks=motor_blocks,
        coordinate_note='Z up; left is +X in CAD as seen from the front (-Y). The CAD zero pose is kept; '
                        'zero joint angles are not a standing posture.')
    return robot, report


def write(output_dir: Path | None = None) -> tuple[Path, dict]:
    """Build and write ``humanoid.urdf``, ``model_report.json`` and the collision hulls."""
    output_dir = output_dir or paths.build_dir('humanoid')
    robot, report = build(output_dir / 'collision_meshes')
    urdf_path = write_urdf(robot, output_dir / 'humanoid.urdf')
    (output_dir / 'model_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    return urdf_path, report


# ------------------------------------------------------------------ helpers
class _MeshLibrary:
    """Loads each export mesh once; meshes must be closed solids for mass properties."""

    def __init__(self):
        self._cache: dict[tuple, trimesh.Trimesh] = {}

    @staticmethod
    def path(visual: ET.Element) -> Path:
        return resolve_package_path(visual.find('geometry/mesh').get('filename'), CAD_DIR).resolve()

    def solid(self, visual: ET.Element) -> trimesh.Trimesh:
        scale = mesh_scale(visual.find('geometry/mesh'))
        key = (self.path(visual).name, tuple(scale))
        if key not in self._cache:
            mesh = trimesh.load_mesh(self.path(visual))
            mesh.apply_scale(scale)
            assert mesh.is_volume and mesh.volume > 0, key[0]
            self._cache[key] = mesh
        return self._cache[key]

    def center(self, visual: ET.Element, placement: np.ndarray) -> np.ndarray:
        """World center of mass of a placed mesh (mesh origins are often far from the part)."""
        return (placement @ np.r_[self.solid(visual).center_mass, 1])[:3]


def _mesh_name(visual: ET.Element) -> str:
    return Path(visual.find('geometry/mesh').get('filename')).name


def _fastened_components(source: ET.Element, links: dict) -> tuple[list[set], dict[str, int]]:
    """Groups of links joined by fixed joints. Onshape's "hanging node" edges attach parts to
    the root for export only, so fixed joints whose parent is ``root`` are ignored."""
    neighbors = {name: set() for name in links}
    for joint in source.findall('joint'):
        parent, child = joint_parent(joint), joint_child(joint)
        if joint.get('type') == 'fixed' and parent != 'root':
            neighbors[parent].add(child)
            neighbors[child].add(parent)
    components, component_of = [], {}
    for name in links:
        if name in component_of:
            continue
        component, stack = set(), [name]
        while stack:
            current = stack.pop()
            if current not in component:
                component.add(current)
                stack.extend(neighbors[current] - component)
        component_of.update({member: len(components) for member in component})
        components.append(component)
    return components, component_of


def _group_loose_motor_parts(links, root_visuals, source_frames, meshes) -> dict[str, list[int]]:
    """Assign every loose root mesh to its nearest motor; each motor must be one contiguous block."""
    motor_centers = {name: meshes.center(link.find('visual'),
                                         source_frames[name] @ origin_matrix(link.find('visual/origin')))
                     for name, link in links.items() if name.startswith(MOTOR_HOUSING_PREFIX)}
    for index, visual in enumerate(root_visuals):
        if _mesh_name(visual) == MOTOR_HOUSING_MESH:
            motor_centers[f'root:{index}'] = meshes.center(visual, origin_matrix(visual.find('origin')))
    assert len(motor_centers) == EXPECTED_MOTORS_IN_EXPORT
    names, centers = list(motor_centers), np.array(list(motor_centers.values()))
    blocks = {name: [] for name in names}
    for index, visual in enumerate(root_visuals):
        if _mesh_name(visual) == LOOSE_WHEEL_MESH:
            continue
        distances = np.linalg.norm(centers - meshes.center(visual, origin_matrix(visual.find('origin'))), axis=1)
        assert distances.min() < MAX_MOTOR_PART_DISTANCE_M
        blocks[names[distances.argmin()]].append(index)
    # Independent check of the nearest-motor rule: the export lists each motor contiguously.
    for indices in blocks.values():
        assert len(indices) == max(indices) - min(indices) + 1
    return blocks


def _motor_joints(source, source_frames, component_of) -> dict[str, MotorJoint]:
    """Every exported continuous joint, checked to lie on its motor housing's shaft axis."""
    motor_joints = {}
    for joint in source.findall('joint'):
        if joint.get('type') != 'continuous':
            continue
        parent, child = joint_parent(joint), joint_child(joint)
        marker = next(name for name in (parent, child) if name.startswith(MOTOR_OUTPUT_PREFIXES))
        suffix = (marker.removeprefix('mg4010_pc_行星支架') if marker.startswith('mg4010_pc')
                  else marker.removeprefix('圆柱齿轮12_0_4'))
        housing = MOTOR_HOUSING_PREFIX + suffix
        output = child if marker == parent else parent
        frame = source_frames[parent] @ origin_matrix(joint.find('origin'))
        axis = frame[:3, :3] @ np.fromstring(joint.find('axis').get('xyz'), sep=' ')
        housing_frame = source_frames[housing]
        alignment = abs(float(axis @ housing_frame[:3, 0]))
        offset = float(np.linalg.norm(np.cross(housing_frame[:3, 3] - frame[:3, 3], axis)))
        assert alignment > .99999 and offset < .0001, (joint.get('name'), alignment, offset)
        motor_joints[joint.get('name')] = MotorJoint(housing, output, marker, component_of[housing],
                                                     component_of[output], frame, axis, alignment, offset)
    return motor_joints


def _add_gripper(side, source, source_frames, links, components, component_of, link_body, body_frames,
                 joints, meshes) -> dict:
    """Jaw bodies, prismatic jaw joints and the coupled pinion of one gripper."""
    tool = f'{side}_arm_6_body'
    tool_links = {name for name, body in link_body.items() if body == tool}
    rail = next(name for name in tool_links if name.startswith(RAIL_PREFIX))
    sliders = sorted((joint for joint in source.findall('joint') if joint.get('type') == 'prismatic'
                      and rail in (joint_parent(joint), joint_child(joint))), key=lambda joint: joint.get('name'))
    assert len(sliders) == 2
    jaws = []
    for index, slider in enumerate(sliders):
        parent, child = joint_parent(slider), joint_child(slider)
        seed = child if parent == rail else parent
        body = f'{side}_jaw_{index}'
        body_frames[body] = source_frames[seed]
        link_body.update({name: body for name in components[component_of[seed]]})
        frame = source_frames[parent] @ origin_matrix(slider.find('origin'))
        axis = frame[:3, :3] @ np.fromstring(slider.find('axis').get('xyz'), sep=' ')
        if child == rail:
            axis = -axis    # parent and child were swapped; keep the physical closing direction
        name = f'{side}_gripper_jaw_{index}'
        joints.append(dict(name=name, source=slider.get('name'), parent=tool, child=body, type='prismatic',
                           axis_world=axis.tolist(), index=index))
        jaws.append((name, body, axis))
    np.testing.assert_allclose(jaws[0][2], -jaws[1][2], atol=3e-5)

    gear = next(name for name in tool_links if name.startswith(PINION_GEAR_PREFIX))
    coupler = next(name for name in tool_links if name.startswith(PINION_COUPLER_PREFIX))
    gear_visual = links[gear].find('visual')
    gear_center = meshes.solid(gear_visual).bounds.mean(axis=0)
    gear_center[1:] = 0      # recenter on the gear axis (its local X), as for the arm
    frame = source_frames[gear] @ origin_matrix(gear_visual.find('origin'))
    frame[:3, 3] += frame[:3, :3] @ gear_center
    pinion = f'{side}_gripper_pinion'
    body_frames[pinion] = frame
    link_body[gear] = link_body[coupler] = pinion
    # Orient the pinion axis so that closing jaw 0 turns it positively.
    rack = next(name for name, body in link_body.items() if body == jaws[0][1] and name.startswith(RACK_PREFIX))
    rack_visual = links[rack].find('visual')
    radial = meshes.center(rack_visual, source_frames[rack] @ origin_matrix(rack_visual.find('origin'))) - frame[:3, 3]
    axis = frame[:3, 0].copy()
    axis *= np.sign(np.dot(np.cross(axis, radial), jaws[0][2]))
    pinion_joint = f'{side}_gripper_pinion_joint'
    joints.append(dict(name=pinion_joint, source='added rack-and-pinion coupling', parent=tool, child=pinion,
                       type='revolute', axis_world=axis.tolist()))
    return dict(tool=tool, jaws=[body for _, body, _ in jaws], jaw_joints=[name for name, _, _ in jaws],
                pinion=pinion, pinion_joint=pinion_joint)


def _allocate_masses(placed, motor_joints, motor_blocks, grippers, motor, hardware) -> list[dict]:
    allocations = []

    def allocate(label, keys, mass_kg, estimated=False):
        assert keys and all(key in placed and not placed[key].allocation for key in keys), label
        assert mass_kg > 0
        shares = distribute_by_volume([(placed[key].mesh, placed[key].in_body) for key in keys], mass_kg)
        for key, share in zip(keys, shares):
            placed[key].mass_properties, placed[key].allocation = share, label
        allocations.append(dict(component=label, mass_kg=mass_kg, estimated=estimated, visuals=len(keys)))

    for spec in motor_joints.values():
        keys = [f'root:{index}' for index in motor_blocks[spec.housing]] + [f'{spec.housing}:0', f'{spec.marker}:0']
        allocate(spec.housing, keys, motor.mass_kg)
    catalog = load_parts()
    for key, record in placed.items():
        if record.allocation:
            continue
        mesh = _mesh_name(record.visual)
        if mesh in catalog:
            allocate(key, [key], catalog[mesh].mass_kg, catalog[mesh].estimated)
        elif mesh in (TORSO_BOARD_MESH, WHEEL_MESH):
            measured = hardware['torso_board_mass_kg' if mesh == TORSO_BOARD_MESH else 'wheel_mass_kg']
            mass = measured if measured is not None else record.mesh.volume * hardware['unknown_structure_density_kg_m3']
            allocate(key, [key], mass, measured is None)
    for side, spec in grippers.items():
        gripper_bodies = {spec['tool'], spec['pinion'], *spec['jaws']}
        keys = [key for key, record in placed.items() if record.body in gripper_bodies and not record.allocation]
        motor_keys = [key for key in keys if key.startswith(GRIPPER_MOTOR_PREFIX)]
        assert len(motor_keys) == 1
        allocate(f'{side}_DM4310', motor_keys, DM4310_GRIPPER.motor_mass_kg)
        allocate(f'{side}_gripper_mechanism', [key for key in keys if key not in motor_keys],
                 hardware['gripper_mass_kg'] - DM4310_GRIPPER.motor_mass_kg, True)
    unassigned = [key for key, record in placed.items() if not record.allocation]
    assert not unassigned, unassigned
    return allocations


def _add_colliders(placed, body_elements, collision_dir: Path) -> tuple[int, int]:
    """Detailed colliders for structure and jaws; one convex exterior per motor half.

    Motor internals (gears, fasteners) are hidden inside the housing, so each motor's
    parts on one body are replaced by the convex hull of their vertices.
    """
    collision_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    motor_halves: dict[tuple[str, str], list[PlacedMesh]] = {}
    for record in placed.values():
        if record.allocation.startswith(MOTOR_HOUSING_PREFIX):
            motor_halves.setdefault((record.allocation, record.body), []).append(record)
            continue
        collision = copy.deepcopy(record.visual)
        collision.tag = 'collision'
        for material in collision.findall('material'):
            collision.remove(material)
        body_elements[record.body].append(collision)
        count += 1
    for (motor_housing, body), parts in motor_halves.items():
        vertices = np.concatenate([trimesh.transform_points(part.mesh.vertices, part.in_body) for part in parts])
        hull_path = collision_dir / f'{motor_housing}_{body}.stl'
        hull_path.write_bytes(trimesh.exchange.stl.export_stl(trimesh.convex.convex_hull(vertices)))
        collision = ET.SubElement(body_elements[body], 'collision', name=f'motor_exterior_{motor_housing}_{body}')
        ET.SubElement(collision, 'origin', xyz='0 0 0', rpy='0 0 0')
        geometry = ET.SubElement(collision, 'geometry')
        ET.SubElement(geometry, 'mesh', filename=str(hull_path), scale='1 1 1')
        count += 1
    return count, len(motor_halves)


def _add_joint_elements(robot, joints, body_frames, grippers, motor) -> None:
    gripper = DM4310_GRIPPER
    for spec in joints:
        joint = ET.SubElement(robot, 'joint', name=spec['name'], type=spec['type'])
        parent, child = spec['parent'], spec['child']
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        set_origin(joint, np.linalg.inv(body_frames[parent]) @ body_frames[child])
        axis = body_frames[child][:3, :3].T @ spec['axis_world']
        ET.SubElement(joint, 'axis', xyz=' '.join(map(str, axis)))
        if spec['type'] == 'continuous':
            ET.SubElement(joint, 'limit', effort=str(motor.rated_torque_nm), velocity=str(motor.max_speed_rad_s))
            continue
        side = spec['name'].split('_')[0]
        is_jaw = spec['type'] == 'prismatic'
        driven = is_jaw and spec['index'] == 0
        ET.SubElement(joint, 'limit', lower='0',
                      upper=str(gripper.jaw_stroke_m if is_jaw else gripper.full_stroke_pinion_angle_rad),
                      effort=str(gripper.jaw_force_limit_n if driven else 0.),
                      velocity=str(gripper.pinion_radius_m * gripper.max_speed_rad_s if is_jaw
                                   else gripper.max_speed_rad_s))
        if not driven:
            ET.SubElement(joint, 'mimic', joint=grippers[side]['jaw_joints'][0],
                          multiplier=str(1. if is_jaw else 1. / gripper.pinion_radius_m), offset='0')


if __name__ == '__main__':
    urdf_path, build_report = write()
    print(json.dumps(dict(urdf=str(urdf_path), bodies=len(build_report['bodies']), joints=len(build_report['joints']),
                          visuals=build_report['included_visuals'], mass_kg=build_report['total_mass_kg']), indent=2))
