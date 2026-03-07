import json
import os
from core import Simulator, AmbulanceTrajectory, RandomWalkTrajectory, SinWaveTrajectory

NUM_SIMULATIONS = 10
FS_CONTROL = 20  # Hz
RANDOM_SEED = 420
ENABLE_DISTRACTIONS = True
SPEECH_MAX_DIST_M = 10.0   # max distance of the speech walker from the mic array center (m)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
MIC_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "warthog.json")
SOUND_SOURCE_PATH = os.path.join(BASE_DIR, "sound", "siren_mono.wav")

SCENARIO_WEIGHTS = {
    'always_on':  0.25,
    'always_off': 0.25,
    'turn_on':    0.20,
    'turn_off':   0.20,
    'interrupted': 0.10,
}

def main():
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
        'width': 60.0,
        'length': 60.0,
    }
    
    # Compute the physical footprint radius of the mic array from the config,
    # then add a small chassis margin. The source must never get closer than this.
    CHASSIS_MARGIN_M = 1.0  # extra clearance around the outermost mic position
    with open(MIC_CONFIG_PATH) as f:
        _mic_data = json.load(f)
    import numpy as np
    _mic_pts = np.array(list(_mic_data.values()))
    array_radius = float(np.max(np.hypot(_mic_pts[:, 0], _mic_pts[:, 1]))) + CHASSIS_MARGIN_M

    sim_config = {
        'num_simulations': NUM_SIMULATIONS,
        'fs_control': FS_CONTROL,
        'random_seed': RANDOM_SEED,
        'output_dir': OUTPUT_DIR,
        'room_config': room_config,
        'array_radius': array_radius,
        'speech_max_dist_m': SPEECH_MAX_DIST_M,
    }

    trajectory = AmbulanceTrajectory(
        room_config=room_config,
        cruise_speed_kmh=60.0,  # target speed on open stretches; runs vary in [30%, 100%] of this
        min_distance_m=array_radius,  # derived from the actual mic config + chassis margin
        acceleration_g=0.3,     # 0.3 g ≈ normal driving comfort
        num_waypoints=None,     # None = random 5-9 waypoints per run
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