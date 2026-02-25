import csv
import json
import os

import numpy as np
from scipy.io import wavfile


def save_simulation_data(
    output_dir: str,
    signals: np.ndarray,
    fs: int,
    positions: list,
    mic_positions: np.ndarray,
    dt: float,
    sim_params: dict,
    envelope: np.ndarray,
) -> None:
    """Persist all outputs for a single simulation run.

    Creates *output_dir* and writes:
        sound/microphone_N.wav  - one WAV per microphone channel
        gt.csv                  - ground-truth trajectory
        microphones.csv         - microphone positions used in this run
        params.json             - environmental / scenario parameters
    """
    os.makedirs(output_dir, exist_ok=True)
    _save_audio(output_dir, signals, fs)
    _save_ground_truth(output_dir, positions, dt, fs, envelope)
    _save_microphone_positions(output_dir, mic_positions)
    _save_params(output_dir, sim_params)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_audio(output_dir: str, signals: np.ndarray, fs: int) -> None:
    audio_dir = os.path.join(output_dir, "sound")
    os.makedirs(audio_dir, exist_ok=True)
    signals_clipped = np.clip(signals, -32767, 32767).astype(np.int16)
    for i in range(signals.shape[0]):
        path = os.path.join(audio_dir, f"microphone_{i + 1}.wav")
        wavfile.write(path, fs, signals_clipped[i])


def _save_ground_truth(
    output_dir: str,
    positions: list,
    dt: float,
    fs: int,
    envelope: np.ndarray,
) -> None:
    gt_file = os.path.join(output_dir, "gt.csv")
    with open(gt_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "sx", "sy", "dist", "angle", "is_active"])
        for step, pos in enumerate(positions):
            if step == 0:
                continue
            time_s = step * dt
            sx, sy = pos[0], pos[1]
            dist = np.hypot(sx, sy)
            angle_deg = np.degrees(np.arctan2(sy, sx))
            sample_idx = min(int(time_s * fs), len(envelope) - 1)
            is_active = int(envelope[sample_idx] > 0.1)
            writer.writerow([
                f"{time_s:.2f}", f"{sx:.3f}", f"{sy:.3f}",
                f"{dist:.2f}", f"{angle_deg:.2f}", is_active,
            ])


def _save_microphone_positions(output_dir: str, mic_positions: np.ndarray) -> None:
    mic_file = os.path.join(output_dir, "microphones.csv")
    with open(mic_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mic_id", "mx", "my", "mz"])
        for i, pos in enumerate(mic_positions):
            writer.writerow([f"mic_{i + 1}", f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}"])


def _save_params(output_dir: str, sim_params: dict) -> None:
    params_file = os.path.join(output_dir, "params.json")
    rounded = {k: round(v, 6) if isinstance(v, float) else v for k, v in sim_params.items()}
    with open(params_file, "w") as f:
        json.dump(rounded, f, indent=2)
