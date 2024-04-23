"""
Highway Env Basic - (4) Observation
"""

import gymnasium as gym


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
            "manual_control": True,

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

    sampling_time = 1 / env.unwrapped.config["policy_frequency"]
    simulation_steps = env.unwrapped.config["duration"] * env.unwrapped.config["policy_frequency"]
    for step in range(simulation_steps):
        # Update Step
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        env.render()

        ego = obs[0]
        print(f"[{step * sampling_time:.2f} sec] position {ego[1], -ego[2]} / velocity {ego[3], -ego[4]}")


if __name__ == "__main__":
    main()
