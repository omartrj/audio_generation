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
MIN_DISTANCE_METERS = 0.0 # L'ambulanza può passare "dentro" l'array
MIN_SPEED_KMH = 3.0
MAX_SPEED_KMH = 6.0

# Simulation Params
SIMULATION_TIME = 20.0
INTERP_MODE = "Sinc"
INCLUDE_REFLECTED_PATH = False
INCLUDE_AIR_ABSORPTION = True

# Limiti delle coordinate dei microfoni (in metri)
X_RANGE = (0.45, 0.55)  # Distanza minima e massima in x
Y_RANGE = (0.95, 1.05)  # Distanza minima e massima in y
Z_RANGE = (0.00, 0.10)  # Altezza minima e massima in z

# Stanza 9x9m, listener al centro (0,0)
ROOM_HALF_SIZE = 4.5    # metà lato stanza (m)
PERSON_HEIGHT  = 1.5    # altezza fissa della sorgente (m)

# Distribuzione degli scenari della sirena (devono sommare a 1.0)
# Il dataset sarà costruito deterministicamente con queste proporzioni esatte.
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
    for i, name in enumerate(names[:-1]):
        c = round(weights[name] * n)
        counts[name] = c
        assigned += c
    counts[names[-1]] = n - assigned  # il resto all'ultimo per evitare errori di arrotondamento

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

def generate_random_walk_trajectory(duration, dt):
    """Simula una persona che cammina in modo naturale in una stanza ROOM_HALF_SIZE*2 m.
    Usa un approccio a waypoint: la persona sceglie un punto casuale nella stanza
    e vi cammina verso, poi sceglie un nuovo punto, e così via.
    La direzione cambia gradualmente per un movimento fluido."""
    num_steps = int(duration / dt)

    speed_ms = random.uniform(MIN_SPEED_KMH, MAX_SPEED_KMH) / 3.6
    step_size = speed_ms * dt  # metri per step

    margin = 0.3  # distanza minima dai muri per i waypoint

    def new_waypoint():
        return np.array([
            random.uniform(-ROOM_HALF_SIZE + margin, ROOM_HALF_SIZE - margin),
            random.uniform(-ROOM_HALF_SIZE + margin, ROOM_HALF_SIZE - margin),
        ])

    # Posizione e direzione iniziali
    x = random.uniform(-ROOM_HALF_SIZE * 0.6, ROOM_HALF_SIZE * 0.6)
    y = random.uniform(-ROOM_HALF_SIZE * 0.6, ROOM_HALF_SIZE * 0.6)
    heading = random.uniform(0, 2 * np.pi)
    target = new_waypoint()

    max_turn = np.radians(12)  # max rotazione per step verso il target

    positions = np.zeros((num_steps + 1, 3))
    positions[0] = [x, y, PERSON_HEIGHT]

    for i in range(1, num_steps + 1):
        # Direzione verso il waypoint corrente
        dx = target[0] - x
        dy = target[1] - y
        dist_to_target = np.hypot(dx, dy)

        # Se siamo vicini al target, ne scegliamo uno nuovo
        if dist_to_target < 0.4:
            target = new_waypoint()
            dx = target[0] - x
            dy = target[1] - y

        desired_heading = np.arctan2(dy, dx)

        # Ruota heading gradualmente verso la direzione desiderata
        angle_diff = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
        turn = np.clip(angle_diff, -max_turn, max_turn)
        heading += turn

        x += step_size * np.cos(heading)
        y += step_size * np.sin(heading)

        # Hard clamp di sicurezza (non dovrebbe mai scattare con i waypoint dentro i muri)
        x = np.clip(x, -ROOM_HALF_SIZE, ROOM_HALF_SIZE)
        y = np.clip(y, -ROOM_HALF_SIZE, ROOM_HALF_SIZE)

        positions[i] = [x, y, PERSON_HEIGHT]

    return positions

def save_simulation_data(output_dir, signals, fs, positions, mic_pos, dt, sim_index, sim_params, envelope):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save Audio
    audio_dir = os.path.join(output_dir, 'sound')
    os.makedirs(audio_dir, exist_ok=True)

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

    # Simulation control
    dt = 1.0 / config['fs_control']
    num_steps = int(config['simulation_time'] / dt)

    trajectory = generate_random_walk_trajectory(config['simulation_time'], dt)
    start_pos = trajectory[0]

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

    # The simulation runs num_steps+1 iterations, each consuming signal_interval = int(fs/fs_update)
    # samples, which slightly exceeds total_samples_needed. Without padding the signal index wraps
    # around to the beginning, causing a brief siren bleed-in at the very end in 'turn_off' scenarios.
    signal_interval = int(fs / config['fs_control'])
    masked_signal = np.concatenate([masked_signal, np.zeros(signal_interval)])

    env.add_source(position=start_pos, signal=masked_signal)
    env.add_microphone_array(microphone_positions)

    # Add Noise
    env.set_background_noise(SNR=snr)

    # Simulation Loop
    signals_list = []
    position_history = [trajectory[0].copy()]

    signals = env.simulate(init=True)
    signals_list.append(signals)

    for step in range(num_steps):
        new_pos = trajectory[step + 1]
        
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