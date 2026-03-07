import csv
import json
import os
import random

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import wavfile
from tqdm import tqdm

from core.worker import run_single_simulation


def _build_scenario_list(num_simulations: int, weights: dict, seed: int) -> list:
    """Distribute scenarios by weight, then shuffle with a fixed seed."""
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


class Simulator:
    """Orchestrates parallel acoustic simulation runs."""

    def __init__(
        self,
        sim_config: dict,
        mic_config: dict,
        room_config: dict,
        trajectory,
        scenario_weights: dict,
        source_audio_path: str,
        enable_distractions: bool = True,
    ) -> None:
        self.sim_config = sim_config
        self.mic_config = mic_config
        self.room_config = room_config
        self.trajectory = trajectory
        self.scenario_weights = scenario_weights
        self.source_audio_path = source_audio_path
        self.enable_distractions = enable_distractions

        if enable_distractions:
            distractions_dir = os.path.join(
                os.path.dirname(self.source_audio_path), "distractions"
            )
            self.distractors = self._load_distractors(distractions_dir)
        else:
            self.distractors = {}

    def _load_distractors(self, base_dir: str) -> dict:
        distractors = {'traffic_noise': [], 'speech': []}
        
        for category in distractors.keys():
            metadata_path = os.path.join(base_dir, category, "metadata.csv")
            if not os.path.exists(metadata_path):
                print(f"Warning: Distractor metadata not found at {metadata_path}")
                continue
                
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)  # Defaults to comma
                for row in reader:
                    audio_path = os.path.join(base_dir, category, row['filename'])
                    if os.path.exists(audio_path):
                        distractors[category].append(audio_path)
            
            print(f"Loaded {len(distractors[category])} {category} tracks.")
            
        return distractors

    def run(self) -> None:
        src_signal, samplerate, simulation_time = self._load_audio()
        self.sim_config["simulation_time"] = simulation_time
        base_mic_positions = self._load_mic_config()
        total_sims = self.sim_config["num_simulations"]
        scenarios = _build_scenario_list(
            total_sims,
            self.scenario_weights,
            self.sim_config["random_seed"],
        )

        num_digits = len(str(total_sims - 1))
        output_dir = self.sim_config["output_dir"]
        missing_indices = []
        for i in range(total_sims):
            seq_name = f"seq{i:0{num_digits}d}"
            gt_path = os.path.join(output_dir, seq_name, "gt.csv")
            if not os.path.exists(gt_path):
                missing_indices.append(i)

        if not missing_indices:
            print(f"All {total_sims} simulations are already present. Nothing to do.")
            return

        dt = 1.0 / self.sim_config["fs_control"]
        max_workers = os.cpu_count()
        dist_str = ", ".join(f"{n}: {scenarios.count(n)}" for n in self.scenario_weights)
        print(f"Starting/resuming {total_sims} simulations on {max_workers} workers [{dist_str}]")
        print(f"Missing simulations to generate: {len(missing_indices)}")
        print(f"Simulation duration: {simulation_time:.2f}s (from audio file)")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_single_simulation,
                    i,
                    self.sim_config,
                    base_mic_positions,
                    self.mic_config["random_offsets"],
                    self.trajectory.generate(self.sim_config["simulation_time"], dt),
                    scenarios[i],
                    src_signal,
                    samplerate,
                    self.distractors,
                    self.enable_distractions,
                ): i
                for i in missing_indices
            }

            with tqdm(total=len(missing_indices), unit="sim") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    def _load_audio(self) -> tuple:
        if not os.path.exists(self.source_audio_path):
            raise FileNotFoundError(f"Audio file not found: {self.source_audio_path}")
        samplerate, data = wavfile.read(self.source_audio_path)
        src_signal = data.astype(float)
        simulation_time = len(src_signal) / samplerate
        return src_signal, samplerate, simulation_time

    def _load_mic_config(self) -> np.ndarray:
        with open(self.mic_config["config_path"], "r") as f:
            mic_data = json.load(f)
        return np.array(list(mic_data.values()))
