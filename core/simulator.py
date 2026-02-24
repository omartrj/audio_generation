import os
import csv
import json
import random
import numpy as np
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pyroadacoustics.pyroadacoustics.environment import Environment as SoundEnv

def _build_scenario_list(num_simulations, weights, seed):
    # Distribute scenarios based on provided weights
    names = list(weights.keys())
    counts = {}
    assigned = 0
    for name in names[:-1]:
        c = round(weights[name] * num_simulations)
        counts[name] = c
        assigned += c
    counts[names[-1]] = num_simulations - assigned

    scenarios = []
    for name, c in counts.items():
        scenarios.extend([name] * c)

    rng = random.Random(seed)
    rng.shuffle(scenarios)
    return scenarios

def _generate_envelope(num_samples, fs, sim_duration, scenario):
    # Generate the siren activation envelope with fade in/out effects
    envelope = np.ones(num_samples)
    fade_samples = int(0.5 * fs)

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

def _save_simulation_data(output_dir, signals, fs, positions, mic_pos, dt, sim_params, envelope):
    # Save audio files, ground truth trajectory, and simulation metadata
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio signals for each microphone
    audio_dir = os.path.join(output_dir, 'sound')
    os.makedirs(audio_dir, exist_ok=True)
    signals_clipped = np.clip(signals, -32767, 32767)

    for i in range(signals.shape[0]):
        output_file = os.path.join(audio_dir, f'microphone_{i+1}.wav')
        wavfile.write(output_file, fs, np.int16(signals_clipped[i]))

    # Save ground truth trajectory and metadata
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
            
            sample_idx = min(int(time_s * fs), len(envelope) - 1)
            is_active = 1 if envelope[sample_idx] > 0.1 else 0
            
            writer.writerow([f"{time_s:.2f}", f"{sx:.2f}", f"{sy:.2f}", f"{dist:.2f}", f"{angle_deg:.2f}", f"{is_active}"])

    # Save microphone positions
    mic_file = os.path.join(output_dir, 'microphones.csv')
    with open(mic_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mic_id', 'mx', 'my', 'mz'])
        for i, pos in enumerate(mic_pos):
            writer.writerow([f"mic_{i+1}", f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}"])

    # Save simulation parameters (ie temperature, humidity, etc)
    params_file = os.path.join(output_dir, 'params.json')
    with open(params_file, 'w') as f:
        json.dump(
            {k: (v if not isinstance(v, float) else round(v, 6)) for k, v in sim_params.items()},
            f, indent=2
        )

def _run_single_simulation(sim_index, sim_config, base_mic_positions, random_offsets, positions, scenario, src_signal, fs):
    # Setup random seeds and generate environmental parameters
    np.random.seed(sim_index + sim_config['random_seed'])
    random.seed(sim_index + sim_config['random_seed'])

    num_digits = len(str(sim_config['num_simulations'] - 1))
    seq_name = f"seq{sim_index:0{num_digits}d}"
    sim_output_dir = os.path.join(sim_config['output_dir'], seq_name)

    temperature = random.uniform(0.0, 35.0)
    pressure = random.uniform(0.95, 1.05)
    humidity = random.uniform(20.0, 90.0)
    snr = random.uniform(20.0, 40.0)
    
    # Apply jitter to microphone positions
    base_x, base_y, base_z = np.abs(base_mic_positions[0]) 
    jitter_x = np.random.uniform(-random_offsets['x'], random_offsets['x'])
    jitter_y = np.random.uniform(-random_offsets['y'], random_offsets['y'])
    jitter_z = np.random.uniform(-random_offsets['z'], random_offsets['z'])

    mic_x = base_x + jitter_x
    mic_y = base_y + jitter_y
    mic_z = max(0.0, base_z + jitter_z) # Ensure microphones don't go below the floor

    mic_positions = np.array([
        [mic_x, mic_y, mic_z],
        [-mic_x, mic_y, mic_z],
        [mic_x, -mic_y, mic_z],
        [-mic_x, -mic_y, mic_z]
    ])

    sim_params = {
        'scenario': scenario,
        'temperature': temperature,
        'pressure': pressure,
        'humidity': humidity,
        'snr': snr,
    }

    dt = 1.0 / sim_config['fs_control']
    num_steps = int(sim_config['simulation_time'] / dt)

    # Initialize acoustic environment
    env = SoundEnv(
        fs=fs,
        fs_update=sim_config['fs_control'],
        temperature=temperature,
        pressure=pressure,
        rel_humidity=humidity
    )

    env.set_simulation_params(
        interp_method="Sinc",
        include_reflection=False,
        include_air_absorption=True
    )
    
    # Prepare source signal and apply envelope
    total_samples_needed = int(sim_config['simulation_time'] * fs)
    if len(src_signal) < total_samples_needed:
        repeats = int(np.ceil(total_samples_needed / len(src_signal)))
        extended_signal = np.tile(src_signal, repeats)[:total_samples_needed]
    else:
        extended_signal = src_signal[:total_samples_needed]

    envelope = _generate_envelope(total_samples_needed, fs, sim_config['simulation_time'], scenario)
    masked_signal = extended_signal * envelope

    signal_interval = int(fs / sim_config['fs_control'])
    masked_signal = np.concatenate([masked_signal, np.zeros(signal_interval)])

    env.add_source(position=positions[0], signal=masked_signal)
    env.add_microphone_array(mic_positions)
    env.set_background_noise(SNR=snr)

    signals_list = []
    position_history = [positions[0].copy()]

    # Run the simulation loop step by step
    signals = env.simulate(init=True)
    signals_list.append(signals)

    for step in range(num_steps):
        new_pos = positions[step + 1]
        env.move_source(new_pos)
        signals = env.simulate()
        signals_list.append(signals)
        position_history.append(new_pos)

    full_signal = np.concatenate(signals_list, axis=1)
    _save_simulation_data(sim_output_dir, full_signal, fs, position_history, mic_positions, dt, sim_params, envelope)

class Simulator:
    def __init__(self, sim_config: dict, mic_config: dict, room_config: dict, trajectory, scenario_weights: dict, source_audio_path: str):
        self.sim_config = sim_config
        self.mic_config = mic_config
        self.room_config = room_config
        self.trajectory = trajectory
        self.scenario_weights = scenario_weights
        self.source_audio_path = source_audio_path

    def run(self):
        # Load audio source
        if not os.path.exists(self.source_audio_path):
            raise FileNotFoundError(f"File not found: {self.source_audio_path}")

        samplerate, data = wavfile.read(self.source_audio_path)
        src_signal = data.astype(float)
        
        # Load microphone configuration
        with open(self.mic_config['config_path'], 'r') as f:
            mic_data = json.load(f)
        base_mic_positions = np.array(list(mic_data.values()))
        
        scenarios = _build_scenario_list(
            self.sim_config['num_simulations'], 
            self.scenario_weights, 
            self.sim_config['random_seed']
        )

        max_workers = os.cpu_count()
        dist_str = ', '.join(f"{n}: {scenarios.count(n)}" for n in self.scenario_weights)
        print(f"Starting {self.sim_config['num_simulations']} simulations on {max_workers} workers [{dist_str}]")

        dt = 1.0 / self.sim_config['fs_control']

        # Execute simulations in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(self.sim_config['num_simulations']):
                positions = self.trajectory.generate(self.sim_config['simulation_time'], dt)
                
                future = executor.submit(
                    _run_single_simulation, 
                    i, 
                    self.sim_config, 
                    base_mic_positions,
                    self.mic_config['random_offsets'],
                    positions, 
                    scenarios[i], 
                    src_signal, 
                    samplerate
                )
                futures[future] = i

            with tqdm(total=self.sim_config['num_simulations'], unit='sim') as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)