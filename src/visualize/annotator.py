import colorsys
from enum import Enum
from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon

from data.unity_data import Capture
from util import point_to_cam


class CoordType(Enum):
    IMAGE = 0
    CAM = 1
    WORLD = 2


class ColorGenerator:
    def __init__(self):
        self.colors = []
        self.index = 0
        self.primary_colors = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
            (0, 1, 1)  # Cyan
        ]

    def next(self):
        if self.index < len(self.primary_colors):
            color = self.primary_colors[self.index]
        else:
            hue = (self.index - len(self.primary_colors)) / 12.0
            color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)

        self.colors.append(color)
        self.index += 1

        color = tuple([int(c * 255) for c in color])
        color = tuple([max(0, min(255, c)) for c in color])

        return color


class Annotator:
    __DEFAULT_COLOR = (255, 0, 0)
    __projection_matrix = None
    __color_generator = ColorGenerator()
    __instance_colors = {}
    __num_infos = 0

    def __init__(self, capture: Capture):
        self.__capture = capture
        self.__img = cv2.imread(capture.file_path)
        self.__init_projection_matrix()

    def __init_projection_matrix(self):
        if self.__capture.keypoints is None or len(self.__capture.keypoints.values) == 0:
            return self

        camera_points = []
        image_points = []
        for kpv in self.__capture.keypoints.values:
            for kp in kpv.keypoints:
                camera_points.append(
                    np.array(
                        [kp.camera_cartesian_location[0], kp.camera_cartesian_location[1],
                         kp.camera_cartesian_location[2],
                         1]))
                image_points.append(np.array([int(kp.location[0]), int(kp.location[1])]))
        camera_points = np.array(camera_points)
        image_points = np.array(image_points)

        a = []
        b = []

        for i in range(len(camera_points)):
            x, y, z, _ = camera_points[i]
            u, v = image_points[i]
            a.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z])
            a.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z])
            b.append(u)
            b.append(v)

        a = np.array(a)
        b = np.array(b)

        m_projection = np.linalg.lstsq(a, b, rcond=None)[0]
        m_projection = np.append(m_projection, 1).reshape(3, 4)

        self.__projection_matrix = m_projection

        return self

    def __project_point(self, point):
        point = np.array(point)
        point = np.append(point, 1)

        projected = self.__projection_matrix @ point
        projected /= projected[2]  # Normalize by the third coordinate

        return projected[:2].astype(int)

    def __convert_point(self, point, point_type: CoordType):
        point = np.array(point)
        if point_type == CoordType.WORLD:
            point = point_to_cam(point, self.__capture.position, self.__capture.rotation)
            point_type = CoordType.CAM
        if point_type == CoordType.CAM:
            point = self.__project_point(point)
        return point

    def __get_instance_color(self, instance_id):
        if instance_id not in self.__instance_colors:
            self.__instance_colors[instance_id] = self.__color_generator.next()
        return self.__instance_colors[instance_id]

    def __ensure_color(self, instance_id, color):
        if color is None:
            if instance_id is None:
                color = self.__DEFAULT_COLOR
            else:
                color = self.__get_instance_color(instance_id)
        return color

    def capture_info(self):
        cv2.putText(self.__img, f"Sequence {self.__capture.sequence} - ID {self.__capture.id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return self

    def additional_info(self, info):
        self.__num_infos += 1
        cv2.putText(self.__img, info, (10, 30 + self.__num_infos * 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return self

    def bb2d(self, instance_id: int, color=None, thickness=2):
        bb2d = self.__capture.bounding_boxes_2d[instance_id] if self.__capture.bounding_boxes_2d is not None else None

        if bb2d is None:
            return self

        if bb2d is not None:
            x = int(bb2d.origin[0])
            y = int(bb2d.origin[1])
            w = int(bb2d.dimension[0])
            h = int(bb2d.dimension[1])

            cv2.rectangle(self.__img, (x, y), (x + w, y + h), self.__ensure_color(instance_id, color), thickness)
            cv2.putText(self.__img, f'{bb2d.label.id} {bb2d.label.name}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.__ensure_color(instance_id, color), thickness)

        return self

    def all_bb2d(self, color=None, thickness=2):
        for instance_id in self.__capture.bounding_boxes_2d.ids():
            self.bb2d(instance_id, color, thickness)
        return self

    def bb3d(self, instance_id: int, color=None, thickness=2):
        bb3d = self.__capture.bounding_boxes_3d[instance_id] if self.__capture.bounding_boxes_3d is not None else None

        if bb3d is None:
            return self

        center = np.array(bb3d.translation)
        width = bb3d.size[0]
        height = bb3d.size[1]
        depth = bb3d.size[2]

        half_width = width / 2
        half_height = height / 2
        half_depth = depth / 2

        # Define the 8 corners in local coordinates
        local_corners = np.array([
            [-half_width, -half_height, -half_depth],
            [half_width, -half_height, -half_depth],
            [half_width, half_height, -half_depth],
            [-half_width, half_height, -half_depth],
            [-half_width, -half_height, half_depth],
            [half_width, -half_height, half_depth],
            [half_width, half_height, half_depth],
            [-half_width, half_height, half_depth]
        ])

        # Create a rotation object from the quaternion
        rotation = R.from_quat(bb3d.rotation)

        # Rotate the corners
        rotated_corners = rotation.apply(local_corners)

        # Translate the corners to the global coordinate system
        global_corners = rotated_corners + center

        lines = []
        for i in range(4):
            j = (i + 1) % 4
            lines.append((global_corners[i], global_corners[j]))
            lines.append((global_corners[i + 4], global_corners[j + 4]))
            lines.append((global_corners[i], global_corners[i + 4]))

        for line in lines:
            start = self.__project_point(line[0])
            end = self.__project_point(line[1])
            cv2.line(self.__img, tuple(start), tuple(end), self.__ensure_color(instance_id, color), thickness)

        return self

    def all_bb3d(self, color=None, thickness=2):
        for instance_id in self.__capture.bounding_boxes_3d.ids():
            self.bb3d(instance_id, color, thickness)
        return self

    def keypoints(self, instance_id: int, color=None, radius=5, thickness=2):
        kps = self.__capture.keypoints[instance_id] if self.__capture.keypoints is not None else None

        if kps is None:
            return self

        for kp in kps.keypoints:
            cv2.circle(self.__img, (int(kp.location[0]), int(kp.location[1])), radius, color, -1)
            cv2.putText(self.__img, f"{kp.index} ({kp.state})", (
                int(kp.location[0]) + 10, int(kp.location[1])), cv2.FONT_HERSHEY_SIMPLEX, .75,
                        self.__ensure_color(instance_id, color), thickness)
        return self

    def all_keypoints(self, color=None, radius=5, thickness=2):
        for instance_id in self.__capture.keypoints.ids():
            self.keypoints(instance_id, color, radius, thickness)
        return self

    def arrow(self, start, end, point_type: CoordType = CoordType.CAM, instance_id=None, color=None, thickness=2):
        start = self.__convert_point(start, point_type)
        end = self.__convert_point(end, point_type)

        cv2.arrowedLine(self.__img, tuple(start), tuple(end), self.__ensure_color(instance_id, color), thickness)

        return self

    def line(self, start, end, point_type: CoordType = CoordType.CAM, instance_id=None, color=None, thickness=2):
        start = self.__convert_point(start, point_type)
        end = self.__convert_point(end, point_type)

        cv2.line(self.__img, tuple(start), tuple(end), self.__ensure_color(instance_id, color), thickness)

        return self

    def point(self, point, point_type: CoordType = CoordType.CAM, instance_id=None, color=None, radius=5):
        point = self.__convert_point(point, point_type)
        cv2.circle(self.__img, tuple(point), radius, self.__ensure_color(instance_id, color), -1)

        return self

    def points(self, points, point_type: CoordType = CoordType.CAM, connect=False, instance_id=None, color=None,
               radius=5,
               thickness=2):
        for point in points:
            self.point(point, point_type, instance_id, color, radius)

        if connect:
            for i in range(len(points) - 1):
                start = self.__convert_point(points[i], point_type)
                end = self.__convert_point(points[i + 1], point_type)
                cv2.line(self.__img, tuple(start), tuple(end), self.__ensure_color(instance_id, color), thickness)

        return self

    def polygons(self, polygons: List[Polygon], point_type: CoordType = CoordType.CAM, instance_id=None, color=None,
                 thickness=2):
        for polygon in polygons:
            points = np.array(polygon.boundary.coords.xy).T.astype(int)

            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                cv2.line(self.__img, tuple(start), tuple(end), self.__ensure_color(instance_id, color), thickness)

        return self

    def save(self, path):
        cv2.imwrite(path, self.__img)

        return self

    def get_img(self):
        return self.__img

    def show(self):
        cv2.imshow('image', self.__img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return self
