"""Opening and shared motor-torque controls for one DM4310 gripper."""
import math

import omni.ui as ui

from arthrobot.gripper import DM4310_GRIPPER

GRIPPER = DM4310_GRIPPER


class GripperPanel:
    def __init__(self, demo=False, mass_kg=GRIPPER.total_mass_estimate_kg, mass_is_estimate=True, camera_paths=()):
        self._updating_display = False
        self.opening_percent = ui.SimpleFloatModel(100.)
        self.demo = ui.SimpleBoolModel(demo)
        self.torque_limit_model = ui.SimpleFloatModel(GRIPPER.rated_torque_nm)
        self.window = ui.Window('DM4310 Gripper Controls', width=460, height=435, position_x=1030, position_y=5,
                                visible=True)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label('Opening: 100% open / 0% closed target', height=22)
                ui.FloatSlider(self.opening_percent, min=0, max=100, step=1, precision=0, height=28)
                with ui.HStack(height=30):
                    ui.Button('Open', clicked_fn=lambda: self.opening_percent.set_value(100.))
                    ui.Button('Close', clicked_fn=lambda: self.opening_percent.set_value(0.))
                with ui.HStack(height=28):
                    for label, path in camera_paths:
                        ui.Button(label, clicked_fn=lambda path=path: self.use_camera(path))
                with ui.HStack(height=24):
                    ui.CheckBox(self.demo, width=24)
                    ui.Label('Gripper demo')
                self.travel_label = ui.Label('Jaw travel: 0.0 / 0.0 mm', height=22)
                self.angle_label = ui.Label('Motor output: 0.0 degrees', height=22)
                ui.Label('Travel: 50 mm per jaw. Contact can stop closure.', height=22)
                ui.Label('Shared motor torque limit (N m)', height=22)
                ui.FloatSlider(self.torque_limit_model, min=0, max=GRIPPER.rated_torque_nm, step=.05, precision=2,
                               height=28)
                self.force_label = ui.Label('', height=22)
                self.status = ui.Label('Ready', height=22)
                qualifier = 'estimated' if mass_is_estimate else 'provided'
                ui.Label(f'Gripper mass: {mass_kg:.3f} kg ({qualifier}); motor: {GRIPPER.motor_mass_kg:.3f} kg.\n'
                         'Ideal transmission; real gripping force is uncalibrated.', height=38, word_wrap=True)
        self.opening_percent.add_value_changed_fn(self._on_manual_edit)

    @staticmethod
    def use_camera(path):
        from omni.kit.viewport.utility import get_active_viewport
        get_active_viewport().set_active_camera(path)

    def _on_manual_edit(self, _model):
        if not self._updating_display:
            self.demo.set_value(False)

    def closing_m(self) -> float:
        """Commanded closing travel per jaw."""
        return GRIPPER.jaw_stroke_m * (1 - self.opening_percent.get_value_as_float() / 100)

    def torque_limit_nm(self) -> float:
        return max(0., min(GRIPPER.rated_torque_nm, self.torque_limit_model.get_value_as_float()))

    def show_closing(self, closing_m):
        self._updating_display = True
        try:
            self.opening_percent.set_value(100 * (1 - closing_m / GRIPPER.jaw_stroke_m))
        finally:
            self._updating_display = False

    def update(self, jaw_positions_m, pinion_angle_rad, tracking_error_m):
        self.travel_label.text = (f'Jaw closing travel: {jaw_positions_m[0] * 1000:.1f} / '
                                  f'{jaw_positions_m[1] * 1000:.1f} mm')
        self.angle_label.text = f'Motor output: {math.degrees(pinion_angle_rad):.1f} degrees'
        equal_load_force = self.torque_limit_nm() / (2 * GRIPPER.pinion_radius_m)
        self.force_label.text = f'Ideal equal-load force at limit: {equal_load_force:.1f} N/jaw'
        self.status.text = ('Tracking / contact or load limiting motion' if tracking_error_m > .0005
                            else 'Following target')
