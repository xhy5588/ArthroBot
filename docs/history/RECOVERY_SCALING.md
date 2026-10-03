# Recovery training: 16, 32 and 64 robot benchmarks

Measured September 24, 2026 on the local RTX 4080 SUPER / 32 GB RAM workstation.
32 and 64 robots improved throughput without rendering. In the current live-window setup, both larger counts exhausted the safe host-memory budget before training began.

| Robots, headless | Seconds / PPO update | Transitions / second | Measured updates |
|---|---:|---:|---:|
| 16 | 7.79 | 98.5 | 10 |
| 32 | 9.20 | 167.0 | 10 |
| 64 | 9.85 | 311.9 | 10 |

64 robots provided **3.17×** the throughput of 16 headless robots and **1.87×** that of 32. Each update takes longer but contains more experience: 768, 1,536 and 3,072 transitions respectively.

The pre-test live 16-robot run averaged 87.8 transitions/s over its latest 30 updates. That is a contextual baseline, not a matched headless comparison.

## Method

Each isolated trial resumed model_249.pt from recovery_runs/20260924_142336, including optimizer and curriculum. It ran 12 PPO updates, discarded the first two timings, and disabled scheduled evaluations. The physics, motors, reset distribution, policy and 48 steps per environment stayed the same. Trials changed only environment count and whether the GUI/rendering was enabled. Four PPO minibatches were retained, so minibatch size increases with environment count.

Trials used systemd user-service limits of 23 GiB MemoryHigh, 24 GiB MemoryMax and 2 GiB MemorySwapMax. A separate guard stopped the service if host available RAM dropped below 1 GiB. Both visible trials were stopped during startup; there is no valid visible 32/64 speed measurement. The headless trials completed without using their cgroup swap allowance and kept over 20 GiB host RAM available.

This is a short throughput test, not a convergence comparison or a proof of successful get-up behavior. Timing includes experience collection and PPO optimization, excludes startup and evaluation, and varies with robot states and other workstation activity. Per-trial metrics and source snapshots are retained under recovery_benchmarks/.

## Training state when the benchmark finished

The main run was saved after 250 completed updates before testing. Benchmark checkpoints never replace the main checkpoint pointer. The original policy and optimizer were resumed with 16 robots and the live window, with 9,750 updates remaining to reach the original 10,000-update target. The benchmark policies were not promoted.

The user subsequently selected **64 robots without rendering**. That continuation
starts from the main run's checkpoint after 275 completed updates, with 9,725
updates remaining. See [RECOVERY.md](RECOVERY.md) for the updated launcher defaults.

Reproduce after pausing the active trainer:

```bash
/home/eric/env_isaaclab/bin/python onshape/modular_humanoid_bipad/training/benchmark_recovery.py \
  --checkpoint /home/eric/AnotherIsaacSim/onshape/modular_humanoid_bipad/training/recovery_runs/20260924_142336/model_249.pt \
  --counts 16 32 64 --headless
```

Omit --headless to test the live-window setup. The script rejects running alongside the tracked main trainer. Raw aggregate results are in recovery_32_64_report.json.
