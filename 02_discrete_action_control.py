"""
Highway Env Basic - (2) Discrete Action Control
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
            "action": {
                "type": "DiscreteMetaAction",
            },
        }
    )
    obs, info = env.reset()

    sampling_time = 1 / env.unwrapped.config["policy_frequency"]
    simulation_steps = env.unwrapped.config["duration"] * env.unwrapped.config["policy_frequency"]
    for step in range(simulation_steps):
        # Autonomous Driving Logic
        action = plan_discrete_action()

        # Update Step
        obs, reward, done, truncated, info = env.step(action)
        env.render()


def plan_discrete_action():
    actions = {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}
    return actions["IDLE"]


if __name__ == "__main__":
    main()
