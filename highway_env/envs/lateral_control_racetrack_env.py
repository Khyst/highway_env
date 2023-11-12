from typing import Tuple

import numpy as np

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, SineLane
from highway_env.road.road import Road, RoadNetwork


class LateralControlRacetrackEnv(AbstractEnv):
    """
    A continuous control environment.

    The agent needs to learn two skills:
    - follow the tracks
    - avoid collisions with other vehicles

    Credits and many thanks to @supperted825 for the idea and initial implementation.
    See https://github.com/eleurent/highway-env/issues/231
    """

    def __init__(self, config: dict = None, render_mode=None) -> None:
        super().__init__(config, render_mode)
        self.nets = []

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "start_lane_index": 0,
            "collision_reward": -1,
            "controlled_vehicles": 1,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "init_lateral_bias": 0.0,
        })
        return config

    def _reward(self, action: np.ndarray) -> float:
        return 0.0

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speed_limits = [12, 8, 12, 8, 8, 12, 8, 8]

        # 1 - Initialise First Lane (Straight Section)
        lane1_inner = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5,
                                   speed_limit=speed_limits[0])
        lane1_outer = StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                   speed_limit=speed_limits[0])
        net.add_lane("a", "b", lane1_inner)
        net.add_lane("a", "b", lane1_outer)

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        lane2_inner = CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                   clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                   speed_limit=speed_limits[1])
        lane2_outer = CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                   clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                   speed_limit=speed_limits[1])
        net.add_lane("b", "c", lane2_inner)
        net.add_lane("b", "c", lane2_outer)

        # 3 - Vertical Straight
        lane3_inner = StraightLane([120, -20], [120, -30],
                                   line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                   speed_limit=speed_limits[2])
        lane3_outer = StraightLane([125, -20], [125, -30],
                                   line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                   speed_limit=speed_limits[2])
        net.add_lane("c", "d", lane3_inner)
        net.add_lane("c", "d", lane3_outer)

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        lane4_inner = CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                   clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                   speed_limit=speed_limits[3])
        lane4_outer = CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                   clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                   speed_limit=speed_limits[3])
        net.add_lane("d", "e", lane4_inner)
        net.add_lane("d", "e", lane4_outer)

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        lane5_inner = CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5,
                                   clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                   speed_limit=speed_limits[4])
        lane5_outer = CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                   clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                   speed_limit=speed_limits[4])
        net.add_lane("e", "f", lane5_inner)
        net.add_lane("e", "f", lane5_outer)

        # 6 - Slant
        lane6_inner = StraightLane([55.7, -15.7], [35.7, -35.7],
                                   line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                   speed_limit=speed_limits[5])
        lane6_outer = StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                   line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                   speed_limit=speed_limits[5])
        net.add_lane("f", "g", lane6_inner)
        net.add_lane("f", "g", lane6_outer)

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        lane7_1_inner = CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                     clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                     speed_limit=speed_limits[6])
        lane7_1_outer = CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5,
                                     clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                     speed_limit=speed_limits[6])
        net.add_lane("g", "h", lane7_1_inner)
        net.add_lane("g", "h", lane7_1_outer)

        lane7_2_inner = CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                     clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                     speed_limit=speed_limits[6])
        lane7_2_outer = CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5,
                                     clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                     speed_limit=speed_limits[6])
        net.add_lane("h", "i", lane7_2_inner)
        net.add_lane("h", "i", lane7_2_outer)

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        lane8_inner = CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5,
                                   clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                   speed_limit=speed_limits[7])
        lane8_outer = CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                   clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                   speed_limit=speed_limits[7])
        net.add_lane("i", "a", lane8_inner)
        net.add_lane("i", "a", lane8_outer)

        self.nets = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """
        lane_index = ("a", "b", self.config["start_lane_index"])
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            self.road.network.get_lane(lane_index).position(0, self.config["init_lateral_bias"]),
            heading=self.road.network.get_lane(lane_index).heading_at(0),
            speed=12)
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def get_next_lane_index(self, lane_index):
        first = lane_index[1]
        second = self.nets[(self.nets.index(lane_index[1]) + 1) % len(self.nets)]
        return first, second, lane_index[2]
