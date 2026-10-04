"""Record rollout videos (and optionally the hand-over test) for new checkpoints of a get-up run.

Plain Python, no Isaac Sim in this process: it polls the run folder and, for every new
``model_<N>.pt`` with N divisible by --every, runs ``record_getup.py`` (the five start
families: an MP4, plus an animated GIF and scalars in TensorBoard under ``<run>/rollouts``).
With --handover it also runs ``handoff.py`` and logs ``Handover/*`` scalars there. Jobs run
one at a time next to training. It exits once the run has finished or its process is gone.
Progress is kept in ``<run>/rollouts/recorded.json``, so a restarted watcher continues.

Rendering needs the RTX workaround on NVIDIA 595.x (see the README).

    python scripts/humanoid/watch_getup_run.py --run-dir logs/humanoid_getup/<run> --every 500 --handover
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

SCRIPTS_DIR = Path(__file__).resolve().parent
MAX_ATTEMPTS = 2
JOB_TIMEOUT_S = 1200


def checkpoints(run_dir: Path, every: int) -> list[tuple[int, Path]]:
    found = []
    for path in run_dir.glob('model_*.pt'):
        match = re.fullmatch(r'model_(\d+)\.pt', path.name)
        if match and int(match[1]) % every == 0:
            found.append((int(match[1]), path))
    return sorted(found)


def training_ended(run_dir: Path) -> bool:
    status_path = run_dir / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    process_gone = 'pid' in status and not Path(f'/proc/{status["pid"]}').exists()
    return status.get('state') in ('finished', 'stopped') or process_gone


def run_job(command: list[str], log_path: Path, title: str) -> str:
    """Run a recording job; append its key output lines to the log and return its full output."""
    try:
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                timeout=JOB_TIMEOUT_S).stdout
    except subprocess.TimeoutExpired as error:
        output = (error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or '') + '\nTIMEOUT'
    key_lines = [line for line in output.splitlines()
                 if line.startswith(('GETUP_', 'Traceback', 'TIMEOUT')) or 'Error' in line[:80]]
    with log_path.open('a') as log:
        log.write(f'\n=== {title} {time.ctime()} ===\n' + '\n'.join(key_lines[-20:]) + '\n')
    return output


def log_handover(report_path: Path, rollouts: Path, update: int) -> float | None:
    from torch.utils.tensorboard import SummaryWriter
    report = json.loads(report_path.read_text())
    writer = SummaryWriter(str(rollouts))
    for group in ('handover', 'getup_policy_only'):
        if report[group]['stayed_standing'] is not None:
            writer.add_scalar(f'Handover/{group}_stayed_standing', report[group]['stayed_standing'], update)
    writer.add_scalar('Handover/arms_outside_standing_range', report['outside_standing_policy_range_fraction'], update)
    writer.close()
    return report['handover']['stayed_standing']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--every', type=int, default=500, help='Record checkpoints whose update is a multiple of this.')
    parser.add_argument('--handover', action='store_true', help='Also run the hand-over test (320 robots, 20 s).')
    parser.add_argument('--poll', type=float, default=30., help='Seconds between checks for new checkpoints.')
    args = parser.parse_args()
    rollouts = args.run_dir / 'rollouts'
    rollouts.mkdir(parents=True, exist_ok=True)
    ledger_path = rollouts / 'recorded.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    log_path = rollouts / 'watcher.log'
    python = sys.executable
    while True:
        todo = [(update, path) for update, path in checkpoints(args.run_dir, args.every)
                if ledger.get(str(update), {}).get('status') != 'ok'
                and ledger.get(str(update), {}).get('attempts', 0) < MAX_ATTEMPTS]
        if not todo:
            if training_ended(args.run_dir):
                print('GETUP_WATCHER: training ended; every checkpoint is recorded', flush=True)
                return
            time.sleep(args.poll)
            continue
        update, checkpoint = todo[0]
        entry = ledger.setdefault(str(update), {'attempts': 0})
        entry['attempts'] += 1
        started = time.time()
        output = run_job([python, '-u', str(SCRIPTS_DIR / 'record_getup.py'), '--checkpoint', str(checkpoint),
                          '--output-dir', str(rollouts), '--tensorboard-dir', str(rollouts)],
                         log_path, f'video, update {update}')
        recorded = 'GETUP_ROLLOUT' in output
        if recorded and args.handover:
            report_path = rollouts / f'handoff_{update:06d}.json'
            run_job([python, '-u', str(SCRIPTS_DIR / 'handoff.py'), '--checkpoint', str(checkpoint),
                     '--num-envs', '320', '--seconds', '20', '--report', str(report_path)],
                    log_path, f'hand-over, update {update}')
            if report_path.exists():
                entry['handover_stayed_standing'] = log_handover(report_path, rollouts, update)
        entry.update(status='ok' if recorded else 'failed', seconds=round(time.time() - started, 1))
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
        print(f'GETUP_WATCHER: update {update} {entry["status"]} in {entry["seconds"]} s', flush=True)


if __name__ == '__main__':
    main()
