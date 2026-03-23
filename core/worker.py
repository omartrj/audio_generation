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
from core.trajectories import RandomWalkTrajectory


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
    enable_distractions: bool = True,
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
    snr         = random.uniform(20.0, 35.0)

    # Random gain for the source signal
    #gain        = random.uniform(0.8, 5.0)
    # Per ora gain fisso
    gain        = 3.0

    mic_positions = _jitter_mic_positions(base_mic_positions, random_offsets)

    sim_params = {
        "scenario":    scenario,
        "temperature": temperature,
        "pressure":    pressure,
        "humidity":    humidity,
        "snr":         snr,
        "gain":        gain,
    }

    # ---- prepare source signal ---------------------------------------------
    dt = 1.0 / sim_config["fs_control"]
    total_samples   = len(src_signal)
    simulation_time = total_samples / fs
    signal_interval = int(fs / sim_config["fs_control"])

    envelope      = generate_envelope(total_samples, fs, simulation_time, scenario)
    src_signal = src_signal * gain # apply random gain to the source signal before masking
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
    
    env_speech = None
    speech_traj = None

    if enable_distractions:
        target_length = len(masked_signal)

        # 1. Traffic noise → background noise on the main environment
        if distractors.get('traffic_noise'):
            tr_path = random.choice(distractors['traffic_noise'])
            tr_fs, tr_data = wavfile.read(tr_path)
            if tr_fs == fs:
                tr_mono = tr_data[:, 0].astype(float) if tr_data.ndim > 1 else tr_data.astype(float)
                if len(tr_mono) < target_length:
                    tr_mono = np.tile(tr_mono, int(np.ceil(target_length / len(tr_mono))))
                tr_mono = tr_mono[:target_length]
                env.set_background_noise(signal=tr_mono, SNR=snr)

        # 2. Speech → second acoustic environment with a random walk near the mics
        if distractors.get('speech'):
            sp_path = random.choice(distractors['speech'])
            sp_fs, sp_data = wavfile.read(sp_path)
            if sp_fs == fs:
                sp_mono = sp_data[:, 0].astype(float) if sp_data.ndim > 1 else sp_data.astype(float)
                if len(sp_mono) < target_length:
                    sp_mono = np.tile(sp_mono, int(np.ceil(target_length / len(sp_mono))))
                sp_mono = sp_mono[:target_length]
                sp_signal_padded = np.concatenate([sp_mono, np.zeros(signal_interval)])

                # Random walk confined to a square of side 2×speech_max_dist_m
                # centred on the mic array, but never entering the array footprint.
                _r = sim_config.get("speech_max_dist_m", 5.0)
                speech_room = {'width': 2.0 * _r, 'length': 2.0 * _r}
                speech_traj = RandomWalkTrajectory(
                    room_config=speech_room,
                    min_speed_kmh=3.0,
                    max_speed_kmh=6.0,
                    person_height=1.7,
                    exclude_center_radius=sim_config.get("array_radius", 1.5),
                ).generate(simulation_time, dt)

                env_speech = SoundEnv(
                    fs=fs,
                    fs_update=sim_config["fs_control"],
                    temperature=temperature,
                    pressure=pressure,
                    rel_humidity=humidity,
                )
                env_speech.set_simulation_params(
                    interp_method="Sinc",
                    include_reflection=False,
                    include_air_absorption=True,
                )
                env_speech.add_source(position=speech_traj[0], signal=sp_signal_padded)
                env_speech.add_microphone_array(mic_positions)

    # ---- step-by-step simulation loop (main source) -----------------------
    num_steps = int(simulation_time / dt)
    signals_list     = [env.simulate(init=True)]
    position_history = [positions[0].copy()]

    for step in range(num_steps):
        env.move_source(positions[step + 1])
        signals_list.append(env.simulate())
        position_history.append(positions[step + 1])

    # ---- step-by-step simulation loop (speech distractor) -----------------
    speech_full = None
    if env_speech is not None:
        speech_signals_list = [env_speech.simulate(init=True)]
        for step in range(num_steps):
            env_speech.move_source(speech_traj[step + 1])
            speech_signals_list.append(env_speech.simulate())
        speech_full = np.concatenate(speech_signals_list, axis=1)

    # ---- save results ------------------------------------------------------
    full_signal = np.concatenate(signals_list, axis=1)
    if speech_full is not None:
        full_signal = full_signal + speech_full
    # Ensure values stay safely inside int16 bounds before saving
    full_signal = np.clip(full_signal, -32767, 32767)
    
    save_simulation_data(
        sim_output_dir, full_signal, fs,
        position_history, mic_positions, dt, sim_params, envelope,
        speech_positions=speech_traj,
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
