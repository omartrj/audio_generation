import random
import numpy as np


def generate_envelope(num_samples: int, fs: int, sim_duration: float, scenario: str) -> np.ndarray:
    """Generate a siren activation envelope with smooth cosine fade-in/fade-out.

    Scenarios:
        always_on   - siren active for the entire simulation
        always_off  - siren silent for the entire simulation
        turn_on     . siren starts silent, then activates mid-way
        turn_off    . siren active at first, then deactivates mid-way
        interrupted . siren is briefly silenced in the middle
    """
    envelope = np.ones(num_samples)
    fade_samples = int(0.3 * fs)

    def apply_fade(start_idx: int, end_idx: int, fade_type: str) -> None:
        start_idx = max(0, start_idx)
        end_idx = min(num_samples, end_idx)
        length = end_idx - start_idx
        if length <= 0:
            return
        t = np.linspace(0, 1, length)
        curve = (1 - np.cos(t * np.pi)) / 2 if fade_type == "in" else (1 + np.cos(t * np.pi)) / 2
        envelope[start_idx:end_idx] = curve

    if scenario == "always_on":
        pass

    elif scenario == "always_off":
        envelope[:] = 0.0

    elif scenario == "turn_on":
        # siren activates between 20% and 65% of the sequence
        t_switch = random.uniform(0.20 * sim_duration, 0.65 * sim_duration)
        idx = int(t_switch * fs)
        envelope[:idx] = 0.0
        apply_fade(idx, idx + fade_samples, "in")

    elif scenario == "turn_off":
        # siren deactivates between 35% and 80% of the sequence
        t_switch = random.uniform(0.35 * sim_duration, 0.80 * sim_duration)
        idx = int(t_switch * fs)
        apply_fade(idx - fade_samples, idx, "out")
        envelope[idx:] = 0.0

    elif scenario == "interrupted":
        # silence a chunk in the middle: off at 25-40%, back on at 55-75%
        t_off = random.uniform(0.25 * sim_duration, 0.40 * sim_duration)
        t_on  = random.uniform(0.55 * sim_duration, 0.75 * sim_duration)
        idx_off = int(t_off * fs)
        idx_on  = int(t_on * fs)
        apply_fade(idx_off - fade_samples, idx_off, "out")
        envelope[idx_off:idx_on] = 0.0
        apply_fade(idx_on, idx_on + fade_samples, "in")

    return envelope
