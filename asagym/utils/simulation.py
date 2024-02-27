from importlib.resources import files

import pygame

import asagym.proto.simulator_pb2 as pb

from .drawing import (
    Pose,
    DrawingUtils,
    Vector2,
    deg2pix,
    SCREEN_WIDTH,
)

BLUE = (0, 0, 255)
RED = (255, 0, 0)


class Plane:
    """
    Represents a plane.
    """

    def __init__(self, center: Vector2, color: str):
        """
        Creates a plane.
        """
        if color == "blue":
            img_path =  files("asagym").joinpath("./assets/fighter_icon.png")
            self.img = pygame.image.load(img_path)
            self.key = "owner"
        elif color == "red":
            img_path =  files("asagym").joinpath("./assets/enemy_icon.png")
            self.img = pygame.image.load(img_path)
            self.key = "foe"
        else:
            raise Exception(f"unsupported color: {color}")
        self.pose = None
        self.center = center

    def reset(self, obs: pb.State):
        """
        Resets the plane position.

        :param pose: the pose of the robot after the reset.
        :type pose: Pose.
        """
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
        center = self.pose.position
        DrawingUtils.draw_img_on_screen(
            window, Plane.transform(self.pose, self.img), center.to_tuple()
        )


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
        # self.cap_mission = CAP((0, 0), 300)

        center_map = Vector2(-48.23752011, -15.71460745)
        self.fighter_plane = Plane(center_map, "blue")
        self.enemy_plane = Plane(center_map, "red")

        self.fighter_point_list = []  # To draw the plane's path
        self.enemy_point_list = []  # To draw the plane's path

    def reset(self, obs: pb.State):
        """
        Resets the simulation.

        :param is_learning: if the robot is learning in this episode.
        :type is_learning: bool.
        """
        self.fighter_plane.reset(obs.owner.player_state)

        if len(obs.foes) > 0:
            self.enemy_plane.reset(obs.foes[0].player_state)

        self.fighter_point_list = []
        self.enemy_point_list = []

    def update(self, obs: pb.State):
        """
        Updates the simulation.
        """
        self.fighter_plane.reset(obs.owner.player_state)

        if len(obs.foes) > 0:
            self.enemy_plane.reset(obs.foes[0].player_state)

        self.fighter_point_list.append(self.fighter_plane.pose.position.to_tuple())
        self.enemy_point_list.append(self.enemy_plane.pose.position.to_tuple())

    def draw(self, window):
        """
        Draws the simulation (planes and space around CAP point).

        :param window: pygame's window where the drawing will occur.
        :type window: pygame's window.
        """

        if len(self.fighter_point_list) >= 2:
            for i in range(len(self.fighter_point_list) - 1):
                point_A = self.fighter_point_list[i]
                point_B = self.fighter_point_list[i + 1]
                DrawingUtils.draw_line_on_screen(window, point_A, point_B, BLUE, 4)
            # pygame.draw.lines(window, BLUE, False, self.fighter_point_list, 4)
        if len(self.enemy_point_list) >= 2:
            for i in range(len(self.enemy_point_list) - 1):
                point_A = self.enemy_point_list[i]
                point_B = self.enemy_point_list[i + 1]
                DrawingUtils.draw_line_on_screen(window, point_A, point_B, RED, 4)
            # pygame.draw.lines(window, RED, False, self.enemy_point_list, 4)
        self.fighter_plane.draw(window)
        self.enemy_plane.draw(window)
