import numpy as np
import random
from abc import ABC, abstractmethod

class Trajectory(ABC):
    @abstractmethod
    def generate(self, duration: float, dt: float) -> np.ndarray:
        pass

class SinWaveTrajectory(Trajectory):
    def __init__(self, room_config: dict, min_speed_kmh: float, max_speed_kmh: float, min_distance_m: float = 1.0):
        self.room_config = room_config
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh
        self.min_distance_m = min_distance_m

    def generate(self, duration: float, dt: float) -> np.ndarray:
        t_check = np.linspace(0, duration, 1000)
        num_steps = int(duration / dt)
        t_sim = np.linspace(0, duration, num_steps + 1)
        
        half_w = self.room_config['width'] / 2.0
        half_l = self.room_config['length'] / 2.0
        
        while True:
            target_speed_ms = random.uniform(self.min_speed_kmh, self.max_speed_kmh) / 3.6

            amp_x = random.uniform(0.1 * half_w, 0.9 * half_w)
            amp_y = random.uniform(0.1 * half_l, 0.9 * half_l)

            cx = random.uniform(-half_w + amp_x, half_w - amp_x)
            cy = random.uniform(-half_l + amp_y, half_l - amp_y)

            omega_x = np.clip((target_speed_ms / amp_x) * random.uniform(0.8, 1.2), 0.05, 0.8)
            omega_y = np.clip((target_speed_ms / amp_y) * random.uniform(0.8, 1.2), 0.05, 0.8)

            phi_x = random.uniform(0, 2 * np.pi)
            phi_y = random.uniform(0, 2 * np.pi)

            path_x = cx + amp_x * np.sin(omega_x * t_check + phi_x)
            path_y = cy + amp_y * np.sin(omega_y * t_check + phi_y)
            
            dists_2d = np.hypot(path_x, path_y)
            if np.min(dists_2d) >= self.min_distance_m:
                sim_x = cx + amp_x * np.sin(omega_x * t_sim + phi_x)
                sim_y = cy + amp_y * np.sin(omega_y * t_sim + phi_y)
                sim_z = np.full_like(sim_x, 1.5)
                return np.column_stack((sim_x, sim_y, sim_z))

class RandomWalkTrajectory(Trajectory):
    def __init__(self, room_config: dict, min_speed_kmh: float, max_speed_kmh: float,
                 person_height: float = 1.5, exclude_center_radius: float = 0.0):
        self.room_config = room_config
        self.min_speed_kmh = min_speed_kmh
        self.max_speed_kmh = max_speed_kmh
        self.person_height = person_height
        self.exclude_center_radius = exclude_center_radius

    def generate(self, duration: float, dt: float) -> np.ndarray:
        num_steps = int(duration / dt)
        speed_ms = random.uniform(self.min_speed_kmh, self.max_speed_kmh) / 3.6
        step_size = speed_ms * dt

        margin = 0.3
        half_w = self.room_config['width'] / 2.0
        half_l = self.room_config['length'] / 2.0
        excl_r = self.exclude_center_radius

        def new_waypoint():
            for _ in range(200):
                pt = np.array([
                    random.uniform(-half_w + margin, half_w - margin),
                    random.uniform(-half_l + margin, half_l - margin),
                ])
                if excl_r <= 0.0 or np.hypot(pt[0], pt[1]) >= excl_r:
                    return pt
            # fallback: place the waypoint on a random bearing just outside the radius
            angle = random.uniform(0, 2 * np.pi)
            r = excl_r + margin
            return np.array([r * np.cos(angle), r * np.sin(angle)])

        # Start outside the exclusion zone
        while True:
            x = random.uniform(-half_w * 0.6, half_w * 0.6)
            y = random.uniform(-half_l * 0.6, half_l * 0.6)
            if excl_r <= 0.0 or np.hypot(x, y) >= excl_r:
                break

        heading = random.uniform(0, 2 * np.pi)
        target = new_waypoint()

        max_turn = np.radians(12)
        positions = np.zeros((num_steps + 1, 3))
        positions[0] = [x, y, self.person_height]

        for i in range(1, num_steps + 1):
            dx = target[0] - x
            dy = target[1] - y

            if np.hypot(dx, dy) < 0.4:
                target = new_waypoint()
                dx = target[0] - x
                dy = target[1] - y

            desired_heading = np.arctan2(dy, dx)
            angle_diff = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi
            heading += np.clip(angle_diff, -max_turn, max_turn)

            nx = np.clip(x + step_size * np.cos(heading), -half_w, half_w)
            ny = np.clip(y + step_size * np.sin(heading), -half_l, half_l)

            # Push back outside the exclusion circle if entered
            if excl_r > 0.0:
                dist = np.hypot(nx, ny)
                if dist < excl_r:
                    if dist < 1e-9:
                        nx, ny = excl_r * np.cos(heading), excl_r * np.sin(heading)
                    else:
                        scale = excl_r / dist
                        nx, ny = nx * scale, ny * scale
                    # Pick new waypoint to steer away from center
                    target = new_waypoint()

            x, y = nx, ny
            positions[i] = [x, y, self.person_height]

        return positions