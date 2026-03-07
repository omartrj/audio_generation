import os
from core import Simulator, RandomWalkTrajectory, SinWaveTrajectory

NUM_SIMULATIONS = 20
FS_CONTROL = 20  # Hz
RANDOM_SEED = 420
ENABLE_DISTRACTIONS = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
MIC_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "warthog.json")
SOUND_SOURCE_PATH = os.path.join(BASE_DIR, "sound", "siren_mono.wav")

SCENARIO_WEIGHTS = {
    'always_on': 0.35,
    'turn_on': 0.25,
    'turn_off': 0.25,
    'interrupted': 0.15
}

def main():
    sim_config = {
        'num_simulations': NUM_SIMULATIONS,
        'fs_control': FS_CONTROL,
        'random_seed': RANDOM_SEED,
        'output_dir': OUTPUT_DIR
    }

    mic_config = {
        "config_path": MIC_CONFIG_PATH,
        "random_offsets": {
            "x": 0.10,
            "y": 0.10,
            "z": 0.05
        }
    }

    # Active configuration: Close-range human walking trajectory
    # room_config = {
    #     'width': 6.0,
    #     'length': 8.0,
    # }
    # trajectory = RandomWalkTrajectory(
    #     room_config=room_config,
    #     min_speed_kmh=3.0,
    #     max_speed_kmh=6.0,
    # )

    # Alternative configuration: Ambulance on a open area
    room_config = {
        'width': 40.0,
        'length': 40.0,
    }
    
    trajectory = SinWaveTrajectory(
        room_config=room_config,
        min_speed_kmh=10.0,
        max_speed_kmh=70.0,
        min_distance_m=3.0
    )

    simulator = Simulator(
        sim_config=sim_config,
        mic_config=mic_config,
        room_config=room_config,
        trajectory=trajectory,
        scenario_weights=SCENARIO_WEIGHTS,
        source_audio_path=SOUND_SOURCE_PATH,
        enable_distractions=ENABLE_DISTRACTIONS,
    )

    simulator.run()

if __name__ == "__main__":
    main()