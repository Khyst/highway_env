from typing import Dict, Text

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class DecisionMakingHighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 10,
            "initial_lane_id": 1,
            "controlled_vehicles": 1,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "unsafe_scenario": False,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=20,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            # current lane 의 front vehicle
            target_vehicle1 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"])
            target_vehicle1.randomize_behavior()
            target_vehicle1.position[0] = vehicle.position[0] + 120
            target_vehicle1.position[1] = vehicle.position[1]
            target_vehicle1.speed = 10
            target_vehicle1.target_speed = 10
            target_vehicle1.enable_lane_change = False
            self.road.vehicles.append(target_vehicle1)

            target_vehicle2 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"])
            target_vehicle2.randomize_behavior()
            target_vehicle2.position[0] = vehicle.position[0] + 140
            target_vehicle2.position[1] = vehicle.position[1]
            target_vehicle2.speed = 15
            target_vehicle2.target_speed = 15
            target_vehicle2.enable_lane_change = False
            self.road.vehicles.append(target_vehicle2)

            target_vehicle3 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"]-1)
            target_vehicle3.randomize_behavior()
            target_vehicle3.position[0] = vehicle.position[0] + 35
            target_vehicle3.position[1] = vehicle.position[1] - 4
            target_vehicle3.speed = 30
            target_vehicle3.target_speed = 30
            target_vehicle3.enable_lane_change = False
            self.road.vehicles.append(target_vehicle3)

            target_vehicle4 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"]-1)
            target_vehicle4.randomize_behavior()
            target_vehicle4.position[0] = vehicle.position[0] - 10
            target_vehicle4.position[1] = vehicle.position[1] - 4
            target_vehicle4.speed = 30
            target_vehicle4.target_speed = 30
            target_vehicle4.enable_lane_change = False
            self.road.vehicles.append(target_vehicle4)

            # left lane 의 rear vehicle
            target_vehicle5 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"]-1)
            target_vehicle5.randomize_behavior()
            target_vehicle5.position[0] = vehicle.position[0] - 20
            target_vehicle5.position[1] = vehicle.position[1] - 4
            target_vehicle5.speed = 25 if self.config["unsafe_scenario"] else 20
            target_vehicle5.target_speed = 25 if self.config["unsafe_scenario"] else 20
            target_vehicle5.enable_lane_change = False
            self.road.vehicles.append(target_vehicle5)

            target_vehicle6 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"]+1)
            target_vehicle6.randomize_behavior()
            target_vehicle6.position[0] = vehicle.position[0] + 25
            target_vehicle6.position[1] = vehicle.position[1] + 4
            target_vehicle6.speed = 10
            target_vehicle6.target_speed = 10
            target_vehicle6.enable_lane_change = False
            self.road.vehicles.append(target_vehicle6)

            target_vehicle7 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"]+1)
            target_vehicle7.randomize_behavior()
            target_vehicle7.position[0] = vehicle.position[0] + 80
            target_vehicle7.position[1] = vehicle.position[1] + 4
            target_vehicle7.speed = 10
            target_vehicle7.target_speed = 10
            target_vehicle7.enable_lane_change = False
            self.road.vehicles.append(target_vehicle7)

            target_vehicle8 = other_vehicles_type.create_random(
                self.road, spacing=1 / self.config["vehicles_density"], lane_id=self.config["initial_lane_id"]+1)
            target_vehicle8.randomize_behavior()
            target_vehicle8.position[0] = vehicle.position[0] + 200
            target_vehicle8.position[1] = vehicle.position[1] + 4
            target_vehicle8.speed = 10
            target_vehicle8.target_speed = 10
            target_vehicle8.enable_lane_change = False
            self.road.vehicles.append(target_vehicle8)

            for _ in range(others-8):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        return 0.0

    def _rewards(self, action: Action) -> Dict[Text, float]:
        return {'reward_name': 0.0}

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
