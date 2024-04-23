"""
Highway Env Basic - (5) Toy Project
"""

import gymnasium as gym
import numpy as np


def main():
    env = gym.make("highway-v0", render_mode="rgb_array")
    # Env 설정
    env.unwrapped.configure(
        {
            # Simulation Time 설정
            "duration": 60,  # [s]
            "simulation_frequency": 50,  # [Hz]
            "policy_frequency": 50,  # [Hz]

            "real_time_rendering": True,

            # Control 설정
            "action": {
                "type": "DiscreteMetaAction",
            },

            # Observation 설정
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "normalize": False,
                "absolute": True,
                "see_behind": True,
                "order": "sorted",
            },
        }
    )
    obs, info = env.reset()

    # 제어 하는 차량의 Target Speed 를 설정 (min / max)
    env.unwrapped.controlled_vehicles[0].target_speeds = np.array([15, 30])

    sampling_time = 1 / env.unwrapped.config["policy_frequency"]
    simulation_steps = env.unwrapped.config["duration"] * env.unwrapped.config["policy_frequency"]
    for step in range(simulation_steps):
        # Autonomous Driving Logic
        action = plan_safe_discrete_action(obs)

        # Update Step
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        ego = obs[0]
        print(f"[{step * sampling_time:.2f} sec] position {ego[1], -ego[2]} / velocity {ego[3], -ego[4]}")


# 전방 일정 거리 내 차량이 있으면 속도를 줄이고, 없으면 속도를 높이는 로직
def plan_safe_discrete_action(obs):
    actions = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}

    ego_status = obs[0]
    for surrounding_status in obs[1:]:
        vehicle_width = 2
        if surrounding_status[0] == 1 and abs(surrounding_status[2] - ego_status[2]) < vehicle_width / 2:
            safety_distance = 30
            if 0 < surrounding_status[1] - ego_status[1] < safety_distance:
                return actions["SLOWER"]

    return actions["FASTER"]


if __name__ == "__main__":
    main()
