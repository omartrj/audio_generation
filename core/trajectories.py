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

class AmbulanceTrajectory(Trajectory):
    """Pseudo-realistic ambulance trajectory in an open-field acoustic scene.

    The vehicle follows a series of waypoints distributed around the listener
    using a kinematic controller that respects a speed-dependent turning radius
    and a smooth acceleration/braking model.

    Parameters
    ----------
    room_config : dict
        Must contain ``'width'`` and ``'length'`` (metres). The ambulance stays
        within these bounds; the maximum reachable distance from the listener is
        approximately ``min(width, length) / 2``.
    cruise_speed_kmh : float
        Target speed on open stretches between waypoints. The vehicle tries to
        maintain this speed and slows down automatically on bends. A slow,
        urban-like run → 20–30 km/h; a fast, open-road run → 60–90 km/h.
        Each ``generate()`` call randomises the actual cruise speed uniformly
        in ``[0.3 * cruise_speed_kmh, cruise_speed_kmh]``, so even with a
        fixed value you get varied runs.
    min_distance_m : float
        Hard exclusion radius around the microphone array origin — the vehicle
        never physically overlaps the array. Derive this from the actual mic
        config: ``max(hypot(mx, my) for each mic) + chassis_margin``.
        Default: 1.2 m (a reasonable fallback; always override with the real value).
    acceleration_g : float
        Maximum longitudinal acceleration/deceleration expressed as a fraction
        of g (9.81 m/s²). Controls how quickly the vehicle speeds up or brakes:
        0.2 g → smooth / comfortable, 0.5 g → hard braking. Default: 0.3.
    num_waypoints : int or None
        Number of waypoints the vehicle visits in a loop. More waypoints →
        more complex path. ``None`` (default) picks a random value in [5, 9].
    height : float
        Source height above ground in metres (centre of the vehicle).
        Default: 1.5 m.
    """

    def __init__(
        self,
        room_config: dict,
        cruise_speed_kmh: float = 50.0,
        min_distance_m: float = 1.2,
        acceleration_g: float = 0.3,
        num_waypoints: int = None,
        height: float = 1.5,
    ):
        self.room_config = room_config
        self.cruise_speed_kmh = cruise_speed_kmh
        self.min_distance_m = min_distance_m
        self.acceleration_g = acceleration_g
        self.num_waypoints = num_waypoints
        self.height = height

    def generate(self, duration: float, dt: float) -> np.ndarray:
        num_steps  = int(duration / dt)
        cruise_max = self.cruise_speed_kmh / 3.6
        # Minimum speed is automatically 30 % of cruise (sharp bends / slow runs)
        cruise_min = 0.30 * cruise_max

        # Each trajectory gets a cruise speed drawn from [30 %, 100 %] of the
        # configured value, producing naturally varied fast and slow sequences.
        cruise_ms = random.uniform(cruise_min, cruise_max)

        half_w = self.room_config['width'] / 2.0
        half_l = self.room_config['length'] / 2.0
        margin = 1.0  # keep vehicle inside room with this safety margin
        max_r   = min(half_w, half_l) - margin

        # ---- generate waypoints: alternate near and far from listener --------
        # This produces realistic approach/departure patterns instead of a
        # uniform ring, making the Doppler and level variations more diverse.
        num_wp = self.num_waypoints if self.num_waypoints is not None else random.randint(5, 9)
        sector = 2 * np.pi / num_wp
        angle_offset = random.uniform(0, sector)
        r_near = max(self.min_distance_m * 1.5, max_r * 0.20)
        r_far  = max_r * 0.95
        waypoints = []
        for k in range(num_wp):
            angle = angle_offset + k * sector
            # alternate between near-ish and far-ish radii
            if k % 2 == 0:
                r = random.uniform(r_near, r_near + (r_far - r_near) * 0.45)
            else:
                r = random.uniform(r_near + (r_far - r_near) * 0.55, r_far)
            pt = np.array([r * np.cos(angle), r * np.sin(angle)])
            pt[0] = np.clip(pt[0], -half_w + margin, half_w - margin)
            pt[1] = np.clip(pt[1], -half_l + margin, half_l - margin)
            waypoints.append(pt)

        # typical inter-waypoint distance, used to scale proximity braking
        avg_wp_dist = np.mean([
            np.hypot(waypoints[(k+1) % num_wp][0] - waypoints[k][0],
                     waypoints[(k+1) % num_wp][1] - waypoints[k][1])
            for k in range(num_wp)
        ])
        # start braking when within 40 % of the average leg length
        proximity_scale = avg_wp_dist * 0.4

        # ---- starting position -----------------------------------------------
        start_r = random.uniform(max_r * 0.4, max_r * 0.85)
        start_a = random.uniform(0, 2 * np.pi)
        x = np.clip(start_r * np.cos(start_a), -half_w + margin, half_w - margin)
        y = np.clip(start_r * np.sin(start_a), -half_l + margin, half_l - margin)

        wp_idx  = 0
        target  = waypoints[wp_idx]
        heading = np.arctan2(target[1] - y, target[0] - x)
        speed   = cruise_ms

        positions    = np.zeros((num_steps + 1, 3))
        positions[0] = [x, y, self.height]

        # waypoint capture radius: at least 2 m, or 0.4 s of travel at cruise speed
        wp_capture_r = max(2.0, cruise_ms * 0.4)

        # wall repulsion starts 3 margin-widths from each boundary
        wall_repulsion_d = margin * 3.0

        for i in range(1, num_steps + 1):
            dx = target[0] - x
            dy = target[1] - y
            dist_to_wp = np.hypot(dx, dy)

            # advance to next waypoint when close enough
            if dist_to_wp < wp_capture_r:
                wp_idx = (wp_idx + 1) % num_wp
                target = waypoints[wp_idx]
                dx = target[0] - x
                dy = target[1] - y
                dist_to_wp = np.hypot(dx, dy)

            desired_heading = np.arctan2(dy, dx)

            # --- smooth wall repulsion: nudge heading away from boundaries ---
            repulse_x = repulse_y = 0.0
            gap_x = half_w - abs(x) - margin
            gap_y = half_l - abs(y) - margin
            if gap_x < wall_repulsion_d:
                repulse_x = -np.sign(x) * (1.0 - gap_x / wall_repulsion_d)
            if gap_y < wall_repulsion_d:
                repulse_y = -np.sign(y) * (1.0 - gap_y / wall_repulsion_d)
            if repulse_x != 0.0 or repulse_y != 0.0:
                wall_heading = np.arctan2(repulse_y, repulse_x)
                w_wall = min(np.hypot(repulse_x, repulse_y), 0.8)
                wall_diff = (wall_heading - desired_heading + np.pi) % (2 * np.pi) - np.pi
                desired_heading += w_wall * wall_diff

            angle_diff = (desired_heading - heading + np.pi) % (2 * np.pi) - np.pi

            # speed-dependent minimum turning radius (real vehicle constraint)
            min_radius = max(2.0, 0.35 * speed)  # metres
            step_size  = speed * dt
            max_turn   = np.arcsin(min(step_size / min_radius, 0.999))
            heading   += np.clip(angle_diff, -max_turn, max_turn)

            # target speed: reduce on sharp bends and when approaching waypoint
            sharpness = 1.0 - 0.6 * (abs(angle_diff) / np.pi)
            proximity = np.clip(dist_to_wp / proximity_scale, 0.4, 1.0)
            tgt_speed = np.clip(cruise_ms * sharpness * proximity, cruise_min, cruise_ms)

            # smooth acceleration / braking — bounded by acceleration_g
            max_dv = self.acceleration_g * 9.81 * dt
            speed    = float(np.clip(tgt_speed, speed - max_dv, speed + max_dv))
            step_size = speed * dt

            nx = x + step_size * np.cos(heading)
            ny = y + step_size * np.sin(heading)

            # hard safety clamps — physical array footprint and room bounds
            nx = np.clip(nx, -half_w + margin, half_w - margin)
            ny = np.clip(ny, -half_l + margin, half_l - margin)
            d = np.hypot(nx, ny)
            if d < self.min_distance_m:
                scale  = self.min_distance_m / max(d, 1e-9)
                nx, ny = nx * scale, ny * scale

            x, y = nx, ny
            positions[i] = [x, y, self.height]

        return positions


class RandomWalkTrajectory(Trajectory):
    def __init__(self, room_config, min_speed_kmh, max_speed_kmh, person_height, exclude_center_radius):
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