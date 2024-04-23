"""
Highway Env Basic - (1) start highway env
"""

import gymnasium as gym


def main():
    env_list = ["highway-v0", "intersection-v0", "parking-v0"]
    env = gym.make(env_list[0], render_mode="rgb_array")
    # Env 설정
    env.unwrapped.configure(
        {
            # Simulation Time 설정
            "duration": 60,  # [s]
            "simulation_frequency": 50,  # [Hz]
            "policy_frequency": 50,  # [Hz]

            "real_time_rendering": True,

            "vehicles_count": 50,

            # Control 설정
            "manual_control": True,
        }
    )
    obs, info = env.reset()

    sampling_time = 1 / env.unwrapped.config["policy_frequency"]
    simulation_steps = env.unwrapped.config["duration"] * env.unwrapped.config["policy_frequency"]
    for step in range(simulation_steps):
        # Update Step
        env.step(env.action_space.sample())
        env.render()


if __name__ == "__main__":
    main()