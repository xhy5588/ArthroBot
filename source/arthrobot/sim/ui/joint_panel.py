"""Sliders for joint targets, with the measured angles shown alongside."""
import numpy as np
import omni.ui as ui


class JointPanel:
    def __init__(self, joint_names, title='Arm Joint Controls', mass_kg=None, torque_cap_nm=None, motor_model=None):
        self._updating_display = False
        self.target_models = [ui.SimpleFloatModel(0.) for _ in joint_names]
        self.demo = ui.SimpleBoolModel(False)
        self.measured_labels = {}
        self.window = ui.Window(title, width=460, height=450, position_x=1030, position_y=460, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=6):
                if motor_model:
                    ui.Label(motor_model, height=22)
                ui.Label('Drag a slider or type an angle in degrees.', height=22)
                with ui.HStack(height=22):
                    ui.Label('Joint', width=85)
                    ui.Label('Target angle')
                    ui.Label('Actual', width=65)
                for index, name in enumerate(joint_names):
                    with ui.HStack(height=28, spacing=6):
                        ui.Label(name, width=85)
                        ui.FloatSlider(self.target_models[index], min=-180, max=180, step=1, precision=1)
                        ui.FloatField(self.target_models[index], width=52, precision=1)
                        self.measured_labels[index] = ui.Label('0.0', width=55)
                    self.target_models[index].add_value_changed_fn(self._on_manual_edit)
                with ui.HStack(height=28, spacing=8):
                    ui.Button('Reset pose', clicked_fn=self.reset)
                    ui.CheckBox(self.demo, width=22)
                    ui.Label('Auto demo', width=120)
                self.status = ui.Label('Running', height=22)
                ui.Label('Self-collision: non-neighboring sections.\nSliders: +/-180 deg; no physics angle limits.',
                         height=36, word_wrap=True)
                if mass_kg is not None and torque_cap_nm is not None:
                    ui.Label(f'Mass: {mass_kg:.3f} kg | Motor torque cap: {torque_cap_nm:.1f} N m', height=22)

    def _on_manual_edit(self, _model):
        if not self._updating_display:
            self.demo.set_value(False)

    def reset(self):
        self.demo.set_value(False)
        for model in self.target_models:
            model.set_value(0.)

    def targets(self) -> np.ndarray:
        """Target angles in radians, shape (1, joints)."""
        return np.deg2rad([model.get_value_as_float() for model in self.target_models]).astype(np.float32)[None, :]

    def update(self, positions, tracking_error=0.):
        for index, label in self.measured_labels.items():
            label.text = f'{np.rad2deg(positions[0, index]):.1f}'
        if tracking_error > .1:
            self.status.text = 'Target not reached: contact or load may be blocking motion.'
        else:
            self.status.text = 'Auto demo' if self.demo.get_value_as_bool() else 'Running: manual joint control'
