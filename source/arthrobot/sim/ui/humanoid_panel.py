"""Limb, wheel and gripper controls for the humanoid viewer."""
import math

import omni.ui as ui

from arthrobot.gripper import DM4310_GRIPPER

CHAINS = ('left_arm', 'right_arm', 'left_leg', 'right_leg')


def _chain_joints(report, chain):
    side, limb = chain.split('_')
    names = [joint['name'] for joint in report['joints'] if joint['type'] == 'continuous']
    if limb == 'arm':
        return [name for name in names if name.startswith(chain)]
    return [name for name in names if name.startswith(side + '_') and '_arm_' not in name]


class HumanoidPanel:
    def __init__(self, report, demo=False, fixed_torso=True, max_wheel_rpm=74.):
        self.report = report
        self.max_wheel_rpm = max_wheel_rpm
        self.demo = ui.SimpleBoolModel(demo)
        self.motors_enabled = ui.SimpleBoolModel(True)
        self.angle_models, self.wheel_models, self.opening_models = {}, {}, {}
        self.measured_labels, self.jaw_labels = {}, {}
        self._updating_display = False
        self.window = ui.Window('Humanoid Motor Controls', width=500, height=790, position_x=1080, position_y=30,
                                visible=True)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label('MG5010: 13 N m  |  DM4310 grippers: 3 N m', height=22)
                ui.Label(('Fixed-torso motor rig' if fixed_torso else 'Floating body; no balance controller')
                         + '; CAD zero pose', height=22)
                with ui.HStack(height=25):
                    ui.CheckBox(self.motors_enabled, width=24)
                    ui.Label('Motors enabled', width=170)
                    ui.CheckBox(self.demo, width=24)
                    ui.Label('One motor at a time demo')
                with ui.HStack(height=28, spacing=5):
                    ui.Button('Return to CAD pose', clicked_fn=self.reset)
                    ui.Button('Stop wheels', clicked_fn=self.stop_wheels)
                    ui.Button('Robot view', clicked_fn=self.use_robot_camera)
                self.status = ui.Label('Manual control', height=22)
                with ui.ScrollingFrame():
                    with ui.VStack(spacing=5, height=0):
                        for chain in CHAINS:
                            self._chain_controls(chain)
                        for side in report['grippers']:
                            self._gripper_controls(side)
                ui.Label('Motor off releases drive forces; gravity and joints stay active.\n'
                         'Self-collision can stop motion. Angle sliders are not joint limits.', height=36, word_wrap=True)
                ui.Label(f"Total mass: {report['total_mass_kg']:.2f} kg (partly estimated).\n"
                         'Torso/battery and wheel masses need calibration.', height=36)
        self.motors_enabled.add_value_changed_fn(
            lambda model: self.demo.set_value(False) if not model.get_value_as_bool() else None)

    def _chain_controls(self, chain):
        side = chain.split('_')[0]
        with ui.CollapsableFrame(chain.replace('_', ' ').title(), height=0, collapsed=False):
            with ui.VStack(spacing=3, height=0):
                for name in _chain_joints(self.report, chain):
                    is_wheel = name.endswith('_wheel')
                    model = ui.SimpleFloatModel(0.)
                    (self.wheel_models if is_wheel else self.angle_models)[name] = model
                    with ui.HStack(height=27, spacing=4):
                        ui.Label(name.removeprefix(side + '_') + (' rpm' if is_wheel else ' deg'), width=120)
                        limit = 30 if is_wheel else 180
                        ui.FloatSlider(model, min=-limit, max=limit, precision=1)
                        ui.FloatField(model, width=52, precision=1)
                        self.measured_labels[name] = ui.Label('0.0', width=55)
                    model.add_value_changed_fn(self._on_manual_edit)

    def _gripper_controls(self, side):
        with ui.CollapsableFrame(side.title() + ' gripper', height=0):
            with ui.VStack(spacing=4, height=0):
                ui.Label('Opening: 100% open / 0% close target', height=20)
                model = ui.SimpleFloatModel(100.)
                self.opening_models[side] = model
                ui.FloatSlider(model, min=0, max=100, precision=0, height=26)
                with ui.HStack(height=26):
                    ui.Button('Open', clicked_fn=lambda: model.set_value(100.))
                    ui.Button('Close', clicked_fn=lambda: model.set_value(0.))
                self.jaw_labels[side] = ui.Label('Jaw travel: 0.0 / 0.0 mm', height=20)
                model.add_value_changed_fn(self._on_manual_edit)

    def _on_manual_edit(self, _model):
        if not self._updating_display:
            self.demo.set_value(False)

    def reset(self):
        self.demo.set_value(False)
        for model in [*self.angle_models.values(), *self.wheel_models.values()]:
            model.set_value(0.)
        for model in self.opening_models.values():
            model.set_value(100.)

    def stop_wheels(self):
        self.demo.set_value(False)
        for model in self.wheel_models.values():
            model.set_value(0.)

    @staticmethod
    def use_robot_camera():
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().set_active_camera('/World/Camera')

    def joint_targets_rad(self) -> dict[str, float]:
        return {name: math.radians(model.get_value_as_float()) for name, model in self.angle_models.items()}

    def wheel_speeds_rad_s(self) -> dict[str, float]:
        return {name: max(-self.max_wheel_rpm, min(self.max_wheel_rpm, model.get_value_as_float())) * math.pi / 30
                for name, model in self.wheel_models.items()}

    def closing_m(self, side) -> float:
        opening = max(0., min(100., self.opening_models[side].get_value_as_float()))
        return DM4310_GRIPPER.jaw_stroke_m * (1 - opening / 100)

    def hold_measured(self, dof_names, positions):
        """Make the sliders show the measured pose (used when motors are switched back on)."""
        self._updating_display = True
        try:
            for name, model in self.angle_models.items():
                model.set_value(math.degrees(float(positions[dof_names.index(name)])))
            for side, spec in self.report['grippers'].items():
                jaw = float(positions[dof_names.index(spec['jaw_joints'][0])])
                self.opening_models[side].set_value(100 * (1 - jaw / DM4310_GRIPPER.jaw_stroke_m))
            for model in self.wheel_models.values():
                model.set_value(0.)
        finally:
            self._updating_display = False

    def update(self, dof_index, positions, velocities, motors_enabled, demo):
        for name, label in self.measured_labels.items():
            value = (float(velocities[dof_index[name]]) * 30 / math.pi if name.endswith('_wheel')
                     else math.degrees(float(positions[dof_index[name]])))
            label.text = f'{value:.1f}'
        for side, spec in self.report['grippers'].items():
            first, second = [positions[dof_index[name]] * 1000 for name in spec['jaw_joints']]
            self.jaw_labels[side].text = f'Jaw closing travel: {first:.1f} / {second:.1f} mm'
        if not motors_enabled:
            self.status.text = 'Motors off: gravity-driven motion'
        elif not demo:
            self.status.text = 'Manual control (actual deg / wheel rpm at right)'
