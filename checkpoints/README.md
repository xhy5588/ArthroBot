# Included policies

| Folder | Policy | Trained with | Load with |
| --- | --- | --- | --- |
| `arm_reach/` | arm TCP reaching, update 550 (`model_550.pt`; `policy.pt` / `policy.onnx` are exported actors) | rsl_rl PPO, `ArthroBot-Arm-Reach-v0` | `scripts/rsl_rl/play.py --task ArthroBot-Arm-Reach-Play-v0 --checkpoint checkpoints/arm_reach/model_550.pt` |
| `humanoid_standing/` | hybrid standing, update 4,999 | rsl_rl PPO, `scripts/humanoid/train_standing.py` | `train_standing.py --mode evaluate` or `--mode preview` |
| `humanoid_getup/` | get-up from lying, update 10,000 of a from-scratch run with the ending pose (model and optimizer state; resumable) | multi-critic PPO, `scripts/humanoid/train_getup.py` | `scripts/humanoid/evaluate_getup.py`, `handoff.py`, `record_getup.py` |

`settings.json` next to a humanoid checkpoint records how it was trained. The
standing settings also hold the nominal standing pose, which the get-up and hand-over
code read from there. `humanoid_getup/evaluation_history.json` lists, for that
run, the held-out results every 250 updates, the pull-force changes and the hand-over tests.

Some paths inside `settings.json` and the `.pt` files point to the original
development repository. They are informational only and nothing loads them.
