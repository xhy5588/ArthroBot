"""Runtime checks of the hybrid standing environment (``train_standing.py --mode check``)."""
import torch

from arthrobot_tasks.humanoid.standing.control import LIMB_JOINTS, TORQUE_LIMIT_NM


def check_hybrid_control(env) -> dict:
    """Every action drives only its own motor with the expected torque; saturation; a zero-action hold; fall reset.

    The zero-action hold uses joint PD only, with no wheel balance law: it checks that the
    nominal pose holds up for half a second, not that the robot can stand.
    """
    env.cfg.randomize_resets = False
    env.reset()
    assert bool((env.arm_clearance() > .03).all())
    for motor in range(len(env.motor_ids)):
        action = torch.zeros_like(env.actions)
        action[:, motor] = .1
        env._pre_physics_step(action)
        env._apply_action()
        env.robot.write_data_to_sim()
        gain = TORQUE_LIMIT_NM if motor >= LIMB_JOINTS else env.cfg.position_scale * env.cfg.position_kp
        expected = torch.zeros_like(env.robot.data.applied_torque)
        expected[:, env.motor_ids[motor]] = .1 * gain
        torch.testing.assert_close(env.robot.data.applied_torque, expected, atol=2e-4, rtol=1e-4)
    assert bool((env.commanded_torques(torch.full_like(env.actions, 100.)) == TORQUE_LIMIT_NM).all())

    env.reset()
    heights, peak_torques, clearances, tilts = [], [], [], []
    resets = 0
    for _ in range(120):
        observation, reward, terminated, _, _ = env.step(torch.zeros_like(env.actions))
        assert torch.isfinite(observation['policy']).all() and torch.isfinite(reward).all()
        assert env.robot.data.applied_torque.abs().max() <= TORQUE_LIMIT_NM + .001
        resets += int(terminated.sum())
        heights.append(float(env.torso_height().min()))
        peak_torques.append(float(env.robot.data.applied_torque.abs().max()))
        clearances.append(float(env.arm_clearance().min()))
        tilts.append(float(torch.acos((-env.robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)).max() * 180 / torch.pi))
    floor = env.floor_forces()
    assert floor.shape == (env.num_envs, 21, 3) and torch.isfinite(floor).all()
    first_half_second = slice(0, 30)
    assert min(heights[first_half_second]) > .46, 'The nominal pose loses height in its first half second'
    assert min(clearances[first_half_second]) > 0., 'The nominal pose arms touch the floor immediately'
    assert max(peak_torques) > .1, 'The PD controller must support the limbs against gravity'

    env.reset()
    upside_down = env.robot.data.root_state_w[:1, :7].clone()
    upside_down[:, 3:] = torch.tensor([0., 1., 0., 0.], device=env.device)
    env.robot.write_root_pose_to_sim(upside_down, env_ids=torch.tensor([0], device=env.device))
    _, _, terminated, _, _ = env.step(torch.zeros_like(env.actions))
    assert bool(terminated[0]) and int(env.episode_length_buf[0]) == 0
    return dict(passed=True, all_20_action_mappings_passed=True, fall_reset_passed=True,
                floor_contact_filter_passed=bool((floor[:, env.wheel_sensor_ids, 2] > 3.).any()),
                zero_action_test_seconds=2., zero_action_resets=resets, zero_action_minimum_height_m=min(heights),
                zero_action_max_tilt_deg=max(tilts), zero_action_minimum_arm_clearance_m=min(clearances),
                zero_action_max_torque_nm=max(peak_torques), first_half_second_minimum_height_m=min(heights[:30]),
                note='Joint PD only; wheel actions are zero. A physics check, not an RL result.')
