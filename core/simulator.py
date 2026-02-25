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
    ) -> None:
        self.sim_config = sim_config
        self.mic_config = mic_config
        self.room_config = room_config
        self.trajectory = trajectory
        self.scenario_weights = scenario_weights
        self.source_audio_path = source_audio_path

    def run(self) -> None:
        src_signal, samplerate, simulation_time = self._load_audio()
        self.sim_config["simulation_time"] = simulation_time
        base_mic_positions = self._load_mic_config()
        scenarios = _build_scenario_list(
            self.sim_config["num_simulations"],
            self.scenario_weights,
            self.sim_config["random_seed"],
        )

        dt = 1.0 / self.sim_config["fs_control"]
        max_workers = os.cpu_count()
        dist_str = ", ".join(f"{n}: {scenarios.count(n)}" for n in self.scenario_weights)
        print(f"Starting {self.sim_config['num_simulations']} simulations on {max_workers} workers [{dist_str}]")
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
                ): i
                for i in range(self.sim_config["num_simulations"])
            }

            with tqdm(total=self.sim_config["num_simulations"], unit="sim") as pbar:
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
