"""Load the trained standing actor outside rsl_rl (for the get-up hand-over and physics checks)."""
import json
from pathlib import Path

import torch

from arthrobot import paths
from arthrobot_tasks.humanoid.standing.observation import OBSERVATION_SIZE

CHECKPOINT = paths.CHECKPOINTS_DIR / 'humanoid_standing/model_4999.pt'
HIDDEN_LAYERS = (256, 256, 128)
ACTIONS = 20


def standing_actor(inputs: int = OBSERVATION_SIZE) -> torch.nn.Sequential:
    """The actor network of the standing policy: ELU MLP 96 -> 256 -> 256 -> 128 -> 20."""
    layers, width = [], inputs
    for hidden in HIDDEN_LAYERS:
        layers += [torch.nn.Linear(width, hidden), torch.nn.ELU()]
        width = hidden
    layers.append(torch.nn.Linear(width, ACTIONS))
    return torch.nn.Sequential(*layers)


def load_standing_actor(checkpoint: Path = CHECKPOINT, device: str = 'cpu') -> torch.nn.Sequential:
    """Deterministic actor (action mean) from an rsl_rl standing checkpoint."""
    actor = standing_actor().to(device)
    weights = torch.load(checkpoint, map_location=device, weights_only=False)['model_state_dict']
    actor.load_state_dict({key.removeprefix('actor.'): value for key, value in weights.items() if key.startswith('actor.')})
    return actor


def load_standing_settings(checkpoint: Path = CHECKPOINT) -> dict:
    """Training settings saved next to a standing checkpoint (includes the nominal pose)."""
    return json.loads((Path(checkpoint).parent / 'settings.json').read_text())
