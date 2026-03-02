"""Worker module for parallel simulation execution.

The top-level function ``run_single_simulation`` must live at module scope so
that ``ProcessPoolExecutor`` can pickle it correctly across processes.
"""
import os
import random

import numpy as np
import scipy.io.wavfile as wavfile
from pyroadacoustics.pyroadacoustics.environment import Environment as SoundEnv

from core.envelope import generate_envelope
from core.io import save_simulation_data


def run_single_simulation(
    sim_index: int,
    sim_config: dict,
    base_mic_positions: np.ndarray,
    random_offsets: dict,
    positions: np.ndarray,
    scenario: str,
    src_signal: np.ndarray,
    fs: int,
    distractors: dict,
) -> None:
    """Run one acoustic simulation and write outputs to disk.

    Seeds both ``numpy`` and ``random`` from *sim_index* + the global seed so
    that every run is fully reproducible yet independent.
    """
    seed = sim_index + sim_config["random_seed"]
    np.random.seed(seed)
    random.seed(seed)

    # ---- output directory --------------------------------------------------
    num_digits = len(str(sim_config["num_simulations"] - 1))
    seq_name = f"seq{sim_index:0{num_digits}d}"
    sim_output_dir = os.path.join(sim_config["output_dir"], seq_name)

    # ---- random environmental parameters -----------------------------------
    temperature = random.uniform(0.0, 35.0)
    pressure    = random.uniform(0.95, 1.05)
    humidity    = random.uniform(20.0, 90.0)
    
    # A single SNR controls the overall volume of the background noise.
    # We use a higher range (e.g., 20.0 to 35.0) to ensure the noise doesn't completely mask the siren.
    snr         = random.uniform(20.0, 35.0)

    mic_positions = _jitter_mic_positions(base_mic_positions, random_offsets)

    sim_params = {
        "scenario":    scenario,
        "temperature": temperature,
        "pressure":    pressure,
        "humidity":    humidity,
        "snr":         snr,
    }

    # ---- prepare source signal ---------------------------------------------
    dt = 1.0 / sim_config["fs_control"]
    total_samples   = len(src_signal)
    simulation_time = total_samples / fs
    signal_interval = int(fs / sim_config["fs_control"])

    envelope      = generate_envelope(total_samples, fs, simulation_time, scenario)
    masked_signal = np.concatenate([src_signal * envelope, np.zeros(signal_interval)])

    # ---- acoustic environment ----------------------------------------------
    env = SoundEnv(
        fs=fs,
        fs_update=sim_config["fs_control"],
        temperature=temperature,
        pressure=pressure,
        rel_humidity=humidity,
    )
    env.set_simulation_params(
        interp_method="Sinc",
        include_reflection=False,
        include_air_absorption=True,
    )
    env.add_source(position=positions[0], signal=masked_signal)
    env.add_microphone_array(mic_positions)
    
    distractor_signal = None
    if distractors:
        mixed_distractors = []
        
        # Load one track from each category
        for category in ['street_pedestrian', 'street_traffic']:
            # Check safely if the user has provided tracks for this category yet
            if category in distractors and len(distractors[category]) > 0:
                chosen_path = random.choice(distractors[category])
                dist_fs, dist_data = wavfile.read(chosen_path)
                
                # Use only if sample rate matches to avoid artifacts
                if dist_fs == fs:
                    if dist_data.ndim > 1:
                        dist_mono = dist_data[:, 0]
                    else:
                        dist_mono = dist_data
                    
                    # Ensure distractor covers the simulation length
                    target_length = len(masked_signal)
                    if len(dist_mono) < target_length:
                        # Tile short tracks to fill the whole clip
                        repeats = int(np.ceil(target_length / len(dist_mono)))
                        dist_mono = np.tile(dist_mono, repeats)
                    
                    # Trim down if too long
                    dist_mono = dist_mono[:target_length]
                    mixed_distractors.append(dist_mono)
        
        if len(mixed_distractors) > 0:
            # Sum the pedestrian and traffic tracks together
            dist_combined = np.sum(mixed_distractors, axis=0)
            
            # Mix white gaussian noise with the combined distractors
            # The environment scaling will adjust this combined signal's power for the req SNR
            power_dist = np.mean(dist_combined.astype(float) ** 2)
            if power_dist > 0:
                # Create white noise with slightly lower power to mix evenly
                #white_noise = np.random.randn(len(dist_combined)) * np.sqrt(power_dist) * 0.5
                #distractor_signal = dist_combined + white_noise
                # just now
                distractor_signal = dist_combined
            else:
                white_noise = np.random.randn(len(masked_signal)) * 10.0
                distractor_signal = white_noise

    # Set background noise (uses custom signal if provided, else pure white noise by env logic)
    env.set_background_noise(signal=distractor_signal, SNR=snr)

    # ---- step-by-step simulation loop --------------------------------------
    num_steps = int(simulation_time / dt)
    signals_list     = [env.simulate(init=True)]
    position_history = [positions[0].copy()]

    for step in range(num_steps):
        env.move_source(positions[step + 1])
        signals_list.append(env.simulate())
        position_history.append(positions[step + 1])

    # ---- save results ------------------------------------------------------
    full_signal = np.concatenate(signals_list, axis=1)
    
    # Normalize volume while preserving relative distance regression scale.
    # Instead of dynamically peaking each sequence independently (which destroys distance scale),
    # we apply a strong static gain multiplier to the entire array.
    STATIC_GAIN = 10.0
    full_signal = full_signal * STATIC_GAIN
    # Ensure values stay safely inside int16 bounds before saving
    full_signal = np.clip(full_signal, -32767, 32767)
    
    save_simulation_data(
        sim_output_dir, full_signal, fs,
        position_history, mic_positions, dt, sim_params, envelope,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _jitter_mic_positions(base_positions: np.ndarray, offsets: dict) -> np.ndarray:
    """Return four symmetric mic positions with small random jitter applied."""
    base_x, base_y, base_z = np.abs(base_positions[0])
    mic_x = base_x + np.random.uniform(-offsets["x"], offsets["x"])
    mic_y = base_y + np.random.uniform(-offsets["y"], offsets["y"])
    mic_z = max(0.0, base_z + np.random.uniform(-offsets["z"], offsets["z"]))
    return np.array([
        [ mic_x,  mic_y, mic_z],
        [-mic_x,  mic_y, mic_z],
        [ mic_x, -mic_y, mic_z],
        [-mic_x, -mic_y, mic_z],
    ])
