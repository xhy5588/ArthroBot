"""Make start-pose banks for the get-up task (no simulator needed).

The banks in source/arthrobot_tasks/humanoid/getup/data/pose_banks/ were made with:

    python scripts/humanoid/make_pose_banks.py lying --seed 42 --per-family 400     # training_seed42_400.json
    python scripts/humanoid/make_pose_banks.py lying --seed 1042 --per-family 32    # held_out_seed1042_32.json
    python scripts/humanoid/make_pose_banks.py lying --seed 2042 --per-family 32    # fresh_seed2042_32.json
    python scripts/humanoid/make_pose_banks.py standing --seed 7 --count 800 \\
        --end-states source/arthrobot_tasks/humanoid/getup/data/pose_banks/getup_end_states_u20500.json

Re-make them after changing the robot geometry or the joint limits.
"""
import argparse
import json
from pathlib import Path

from arthrobot_tasks.humanoid.getup.poses import POSE_BANK_DIR, make_lying_bank, make_standing_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=['lying', 'standing'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--per-family', type=int, default=400, help='Lying poses per family.')
    parser.add_argument('--count', type=int, default=800, help='Standing poses.')
    parser.add_argument('--end-states', type=Path, help='Recorded get-up end states (standing banks).')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', type=Path, help='Default: data/pose_banks/<kind>_seed<seed>_<count>.json.')
    args = parser.parse_args()
    if args.kind == 'lying':
        bank = make_lying_bank(args.seed, args.per_family, args.workers)
        output = args.output or POSE_BANK_DIR / f'lying_seed{args.seed}_{args.per_family}.json'
    else:
        bank = make_standing_bank(args.seed, args.count, args.end_states, args.workers)
        output = args.output or POSE_BANK_DIR / f'standing_seed{args.seed}_{args.count}.json'
    output.write_text(json.dumps(bank) + '\n')
    print(f'Wrote {len(bank["poses"])} poses to {output}')


if __name__ == '__main__':
    main()
