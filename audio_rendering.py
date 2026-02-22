import numpy as np
import random
import os
import csv
import json
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pyroadacoustics.pyroadacoustics.environment import Environment as SoundEnv

# Configuration
RANDOM_SEED = 42069
NUM_SIMULATIONS = 10
MIN_DISTANCE_METERS = 1.0
GLOBAL_MAX_PEAK = 25000.0 # Osservato empiricamente
MIN_SPEED_KMH = 10.0
MAX_SPEED_KMH = 80.0

# Simulation Params
SIMULATION_TIME = 20.0
INTERP_MODE = "Sinc"
INCLUDE_REFLECTED_PATH = False
INCLUDE_AIR_ABSORPTION = True

# Limiti delle coordinate dei microfoni (in metri)
X_RANGE = (0.5, 2.0)  # Distanza minima e massima in x
Y_RANGE = (0.5, 2.0)  # Distanza minima e massima in y
Z_RANGE = (0.0, 2.5)  # Altezza minima e massima in z

# Distribuzione degli scenari della sirena (devono sommare a 1.0)
SCENARIO_WEIGHTS = {
    'always_on':   0.35,  # Sirena sempre accesa
    'turn_on':     0.25,  # Parte spenta, si accende in un istante random tra 5s e 15s
    'turn_off':    0.25,  # Parte accesa, si spegne in un istante random tra 5s e 15s
    'interrupted': 0.15,  # Accesa -> Spenta -> Accesa
}


def build_scenario_list(n, weights, seed):
    """Costruisce una lista di n scenari distribuita deterministicamente
    in base ai pesi forniti, poi la mescola con il seed dato."""
    names = list(weights.keys())
    counts = {}
    assigned = 0
    for name in names[:-1]:
        c = round(weights[name] * n)
        counts[name] = c
        assigned += c
    counts[names[-1]] = n - assigned

    scenarios = []
    for name, c in counts.items():
        scenarios.extend([name] * c)

    rng = random.Random(seed)
    rng.shuffle(scenarios)
    return scenarios


def generate_siren_envelope(num_samples, fs, sim_duration, scenario):
    envelope = np.ones(num_samples)
    fade_samples = int(0.5 * fs)  # 500ms fade

    def apply_fade(env, start_idx, end_idx, fade_type):
        start_idx = max(0, start_idx)
        end_idx = min(num_samples, end_idx)
        length = end_idx - start_idx
        if length <= 0: return
        t = np.linspace(0, 1, length)
        curve = (1 - np.cos(t * np.pi)) / 2 if fade_type == 'in' else (1 + np.cos(t * np.pi)) / 2
        env[start_idx:end_idx] = curve

    if scenario == 'always_on':
        pass
    elif scenario == 'turn_on':
        t_switch = random.uniform(5.0, sim_duration - 5.0)
        idx_switch = int(t_switch * fs)
        envelope[:idx_switch] = 0.0
        apply_fade(envelope, idx_switch, idx_switch + fade_samples, 'in')
    elif scenario == 'turn_off':
        t_switch = random.uniform(5.0, sim_duration - 5.0)
        idx_switch = int(t_switch * fs)
        apply_fade(envelope, idx_switch - fade_samples, idx_switch, 'out')
        envelope[idx_switch:] = 0.0
    elif scenario == 'interrupted':
        t_off = random.uniform(4.0, 8.0)
        t_on = random.uniform(12.0, 16.0)
        idx_off = int(t_off * fs)
        idx_on = int(t_on * fs)
        apply_fade(envelope, idx_off - fade_samples, idx_off, 'out')
        envelope[idx_off:idx_on] = 0.0
        apply_fade(envelope, idx_on, idx_on + fade_samples, 'in')

    return envelope

def get_valid_trajectory_params(duration):
    # Resolution for check
    t_check = np.linspace(0, duration, 1000)
    
    while True:
        # Realistic speed scenario
        target_speed_kmh = random.uniform(MIN_SPEED_KMH, MAX_SPEED_KMH) # 10-80 km/h
        target_speed_ms = target_speed_kmh / 3.6

        # Amplitude based on speed
        if target_speed_kmh > 40:
             amp_range = (30.0, 60.0)
        else:
             amp_range = (15.0, 30.0)

        amp_x = random.uniform(*amp_range)
        amp_y = random.uniform(*amp_range)

        # Omega = V / A
        omega_x = (target_speed_ms / amp_x) * random.uniform(0.8, 1.2)
        omega_y = (target_speed_ms / amp_y) * random.uniform(0.8, 1.2)
        
        omega_x = np.clip(omega_x, 0.05, 0.8)
        omega_y = np.clip(omega_y, 0.05, 0.8)

        phi_x = random.uniform(0, 2 * np.pi)
        phi_y = random.uniform(0, 2 * np.pi)

        # Z-axis static
        amp_z = 0.0
        omega_z = 0.1
        phi_z = 0.0

        center_x = random.uniform(-15, 15)
        center_y = random.uniform(-15, 15)
        center_z = 1.5

        # Trajectory trial
        path_x = center_x + amp_x * np.sin(omega_x * t_check + phi_x)
        path_y = center_y + amp_y * np.sin(omega_y * t_check + phi_y)
        path_z = center_z + amp_z * np.sin(omega_z * t_check + phi_z)

        # 2D Distance Check
        dists_2d = np.hypot(path_x, path_y)
        min_dist_2d = np.min(dists_2d)

        if min_dist_2d >= MIN_DISTANCE_METERS:
            start_pos = np.array([path_x[0], path_y[0], path_z[0]])
            
            params = {
                'ax': amp_x, 'ox': omega_x, 'px': phi_x, 'cx': center_x,
                'ay': amp_y, 'oy': omega_y, 'py': phi_y, 'cy': center_y,
                'az': amp_z, 'oz': omega_z, 'pz': phi_z, 'cz': center_z
            }
            return params, start_pos

def save_simulation_data(output_dir, signals, fs, positions, mic_pos, dt, sim_index, sim_params, envelope):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Audio
    audio_dir = os.path.join(output_dir, 'sound')
    os.makedirs(audio_dir, exist_ok=True)

    # Taglia i picchi che escono dal range int16
    signals_clipped = np.clip(signals, -32767, 32767)

    for i in range(signals.shape[0]):
        output_file = os.path.join(audio_dir, f'microphone_{i+1}.wav')
        wavfile.write(output_file, fs, np.int16(signals_clipped[i]))
    # Save Ground Truth Trajectory
    gt_file = os.path.join(output_dir, 'gt.csv')
    with open(gt_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s', 'sx', 'sy', 'dist', 'angle', 'is_active'])
        for step, pos in enumerate(positions):
            if step == 0: continue 
            
            time_s = step * dt
            sx, sy = pos[0], pos[1]
            dist = np.hypot(sx, sy)
            angle_deg = np.degrees(np.arctan2(sy, sx))
            
            sample_idx = int(time_s * fs)
            # Ensure sample_idx doesn't go out of bounds for the very last sample
            sample_idx = min(sample_idx, len(envelope) - 1)
            is_active = 1 if envelope[sample_idx] > 0.1 else 0
            
            writer.writerow([f"{time_s:.2f}", f"{sx:.2f}", f"{sy:.2f}", f"{dist:.2f}", f"{angle_deg:.2f}", f"{is_active}"])

    # Save Microphone Positions
    mic_file = os.path.join(output_dir, 'microphones.csv')
    with open(mic_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mic_id', 'mx', 'my', 'mz'])
        for i, pos in enumerate(mic_pos):
            writer.writerow([f"mic_{i+1}", f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}"])

    # Save simulation parameters as JSON
    params_file = os.path.join(output_dir, 'params.json')
    with open(params_file, 'w') as f:
        json.dump(
            {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in sim_params.items()},
            f, indent=2
        )

def run_single_simulation(sim_index, base_output_dir, src_signal, fs, config, scenario):
    np.random.seed(sim_index + RANDOM_SEED)
    random.seed(sim_index + RANDOM_SEED)

    num_digits = len(str(NUM_SIMULATIONS - 1))
    seq_name = f"seq{sim_index:0{num_digits}d}"
    sim_output_dir = os.path.join(base_output_dir, seq_name)

    # Generate random simulation parameters
    temperature = random.uniform(0.0, 35.0)  # °C
    pressure = random.uniform(0.95, 1.05)    # atm
    humidity = random.uniform(20.0, 90.0)    # %
    snr = random.uniform(20.0, 40.0)         # dB
    
    # Geometria random dei microfoni (simmetrica)
    mic_x = random.uniform(*X_RANGE)
    mic_y = random.uniform(*Y_RANGE)
    mic_z = random.uniform(*Z_RANGE)
    
    microphone_positions = np.array([
        [ mic_x,  mic_y, mic_z],   # Front Right
        [-mic_x,  mic_y, mic_z],   # Front Left
        [ mic_x, -mic_y, mic_z],   # Back Right
        [-mic_x, -mic_y, mic_z]    # Back Left
    ])
    
    sim_params = {
        'scenario': scenario,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'snr': snr,
    }

    traj_params, start_pos = get_valid_trajectory_params(config['simulation_time'])

    env = SoundEnv(
        fs=fs,
        fs_update=config['fs_control'],
        temperature=temperature,
        pressure=pressure,
        rel_humidity=humidity
    )

    env.set_simulation_params(
        interp_method=INTERP_MODE,
        include_reflection=INCLUDE_REFLECTED_PATH,
        include_air_absorption=INCLUDE_AIR_ABSORPTION
    )
    
    # Ensure the source signal is exactly as long as the simulation
    total_samples_needed = int(config['simulation_time'] * fs)
    if len(src_signal) < total_samples_needed:
        repeats = int(np.ceil(total_samples_needed / len(src_signal)))
        extended_signal = np.tile(src_signal, repeats)[:total_samples_needed]
    else:
        extended_signal = src_signal[:total_samples_needed]

    envelope = generate_siren_envelope(total_samples_needed, fs, config['simulation_time'], scenario)
    masked_signal = extended_signal * envelope
    
    env.add_source(position=start_pos, signal=masked_signal)
    env.add_microphone_array(microphone_positions)

    # Add Noise
    env.set_background_noise(SNR=snr)

    # Simulation Loop
    dt = 1.0 / config['fs_control']
    num_steps = int(config['simulation_time'] / dt)
    
    signals_list = []
    position_history = [start_pos.copy()]

    signals = env.simulate(init=True)
    signals_list.append(signals)

    p = traj_params
    for step in range(num_steps):
        t_current = step * dt
        
        new_x = p['cx'] + p['ax'] * np.sin(p['ox'] * t_current + p['px'])
        new_y = p['cy'] + p['ay'] * np.sin(p['oy'] * t_current + p['py'])
        new_z = p['cz'] + p['az'] * np.sin(p['oz'] * t_current + p['pz'])
        
        new_pos = np.array([new_x, new_y, new_z])
        
        env.move_source(new_pos)
        signals = env.simulate()
        
        signals_list.append(signals)
        position_history.append(new_pos)

    full_signal = np.concatenate(signals_list, axis=1)
    save_simulation_data(sim_output_dir, full_signal, fs, position_history, microphone_positions, dt, sim_index, sim_params, envelope)

    return f"Simulation {sim_index} completed."

if __name__ == '__main__':
    BASE_OUTPUT_DIR = os.path.join(os.getcwd(), 'data')
    
    config = {
        'fs_control': 20,
        'simulation_time': SIMULATION_TIME
    }

    current_path = os.getcwd()
    path_sound = os.path.join(current_path, 'sound', 'siren_mono.wav')

    if not os.path.exists(path_sound):
        raise FileNotFoundError(f"File not found: {path_sound}")

    samplerate, data = wavfile.read(path_sound)
    src_signal = data.astype(float)
    
    max_workers = os.cpu_count()
    scenarios = build_scenario_list(NUM_SIMULATIONS, SCENARIO_WEIGHTS, RANDOM_SEED)

    dist_str = ', '.join(f"{n}: {scenarios.count(n)}" for n in SCENARIO_WEIGHTS)
    print(f"Starting {NUM_SIMULATIONS} simulations on {max_workers} workers [{dist_str}]")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_simulation, i, BASE_OUTPUT_DIR, src_signal, samplerate, config, scenarios[i]): i
            for i in range(NUM_SIMULATIONS)
        }

        with tqdm(total=NUM_SIMULATIONS, unit='sim') as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)