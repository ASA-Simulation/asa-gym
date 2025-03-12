from importlib.resources import files
from typing import Dict, List, Tuple

import pygame

import asagym.proto.simulator_pb2 as pb

from .drawing import SCREEN_WIDTH, DrawingUtils, Pose, Vector2, deg2pix

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


class Plane:
    """
    Represents a plane.
    """

    def __init__(self, center: Vector2, color: str):
        """
        Creates a plane.
        """
        if color == "blue":
            img_path = files("asagym").joinpath("./assets/fighter_icon.png")
            self.img = pygame.image.load(img_path)
            self.key = "owner"
        elif color == "green":
            img_path = files("asagym").joinpath("./assets/wingman_icon.png")
            self.img = pygame.image.load(img_path)
            self.key = "wingman"
        elif color == "red":
            img_path = files("asagym").joinpath("./assets/fighter_enemy_icon.png")
            self.img = pygame.image.load(img_path)
            self.key = "foe"
        else:
            raise Exception(f"unsupported color: {color}")
        self.pose = None
        self.center = center
        self.trail: List[Tuple[float, float]] = []

    def reset(self, obs: pb.PlayerState):
        """
        Resets the plane position.

        :param pose: the pose of the robot after the reset.
        :type pose: Pose.
        """
        self.trail = []
        self.update(obs)

    def update(self, obs: pb.PlayerState):
        """
        Updates the robot, including its controller.
        """

        lat_deg = obs.latitude
        long_deg = obs.longitude
        heading_deg = obs.heading

        delta_lat_px = -deg2pix(lat_deg - self.center.y)
        delta_long_px = deg2pix(long_deg - self.center.x)
        rot = 90.0 - heading_deg

        self.pose = Pose(delta_long_px, delta_lat_px, rot)
        self.trail.append(self.pose.position.to_tuple())

    @staticmethod
    def transform(pose, img: pygame.Surface):
        """
        Transforms (translate and rotate) a polygon by a given pose.

        :param pose: translation and rotation of the transform.
        :type pose: Pose.
        :param polygon: the polygon which will be transformed.
        :type polygon: list of Vector2.
        :return: the polygon after each is transformed accordingly.
        :rtype: list of Vector2.
        """
        img = pygame.transform.scale_by(img, 1 / 20 * SCREEN_WIDTH / img.get_width())
        return pygame.transform.rotate(img, pose.rotation)

    def draw(self, window):
        """
        Draws the robot sprite on the screen.

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        :param pose: current pose of the robot.
        :type pose: Pose.
        """
        # Drawing the icon/image
        center = self.pose.position
        DrawingUtils.draw_img_on_screen(
            window, Plane.transform(self.pose, self.img), center.to_tuple()
        )
        # Drawing the trail of the plane
        if len(self.trail) >= 2:
            for i in range(len(self.trail) - 1):
                point_A = self.trail[i]
                point_B = self.trail[i + 1]
                DrawingUtils.draw_line_on_screen(window, point_A, point_B, RED, 4)


class Simulation:
    """
    Represents the simulation.
    """

    def __init__(self):
        """
        Creates the simulation.

        :param line_follower: the line follower robot.
        :type line_follower: LineFollower.
        :param track: the line track.
        :type track: Track.
        """
        self.center_map = Vector2(-48.23752011, -15.71460745)
        self.fighter_planes: Dict[int, Plane] = dict()
        self.enemy_planes: Dict[int, Plane] = dict()

    def reset(self, summary: pb.Summary):
        """
        Resets the simulation.

        :param is_learning: if the robot is learning in this episode.
        :type is_learning: bool.
        """
        for id, state in summary.own_team.items():
            if id not in self.fighter_planes:
                self.fighter_planes[id] = Plane(self.center_map, "blue")
            self.fighter_planes[id].reset(state)
        for id, state in summary.ene_team.items():
            if id not in self.enemy_planes:
                self.enemy_planes[id] = Plane(self.center_map, "red")
            self.enemy_planes[id].reset(state)

    def update(self, summary: pb.Summary):
        """
        Updates the simulation.
        """
        for id, state in summary.own_team.items():
            if id not in self.fighter_planes:
                self.fighter_planes[id] = Plane(self.center_map, "blue")
            self.fighter_planes[id].update(state)
        for id, state in summary.ene_team.items():
            if id not in self.enemy_planes:
                self.enemy_planes[id] = Plane(self.center_map, "red")
            self.enemy_planes[id].update(state)

    def draw(self, window):
        """
        Draws the simulation (planes and paths around the CAP point).

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        """
        for _, plane in self.fighter_planes.items():
            plane.draw(window)
        for _, plane in self.enemy_planes.items():
            plane.draw(window)
