# Tilt-recovery experiments (archived)

These notes record the humanoid experiments that came before the get-up task.
Each one trained a policy to recover from a tilted, upright start. None of them
could recover from leans of 25–40°, for three reasons:

- the wheels can catch a lean of only about 20° (0.59 m/s top speed);
- joint targets of ±0.25 rad around the standing pose cannot reach the floor;
- the actor learning rate was fixed at 3e-6.

The [get-up task](../humanoid.md#get-up-from-lying) replaced this line of work.

The code for these experiments is **not** in this repository. The notes are kept
because they explain design decisions in the current tasks. File names, run folders
and commands in them refer to the original development repository.

| Note | Experiment |
| --- | --- |
| [RECOVERY.md](RECOVERY.md) | v1: randomized get-up and balance from tilted starts |
| [RECOVERY_V2.md](RECOVERY_V2.md) | v2: sustained balance plus a directional recovery curriculum |
| [RECOVERY_V3.md](RECOVERY_V3.md) | v3: fine-tuning that protected the standing policy with a frozen teacher |
| [RECOVERY_PHASED.md](RECOVERY_PHASED.md) | phased lean curriculum after standing (stopped at the 25–40° phase) |
| [RECOVERY_SCALING.md](RECOVERY_SCALING.md) | throughput with 16, 32 and 64 robots before the simulation speed-up |
