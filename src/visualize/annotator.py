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


class Annotator:
    def __init__(self, capture: Capture):
        self.__capture = capture
        self.__img = cv2.imread(capture.file_path)
        self.__projection_matrix = None
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

    def capture_info(self):
        cv2.putText(self.__img, f"Sequence {self.__capture.sequence} - ID {self.__capture.id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return self

    def additional_info(self, info):
        cv2.putText(self.__img, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        return self

    def bb2d(self, instance_id: int):
        bb2d = self.__capture.bounding_boxes_2d[instance_id] if self.__capture.bounding_boxes_2d is not None else None

        if bb2d is None:
            return self

        if bb2d is not None:
            x = int(bb2d.origin[0])
            y = int(bb2d.origin[1])
            w = int(bb2d.dimension[0])
            h = int(bb2d.dimension[1])

            cv2.rectangle(self.__img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(self.__img, f'{bb2d.label.id} {bb2d.label.name}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return self

    def bb3d(self, instance_id: int, color=(0, 255, 0), thickness=2):
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
            cv2.line(self.__img, tuple(start), tuple(end), color, thickness)

        return self

    def keypoints(self, instance_id: int, color=(0, 255, 0), radius=5):
        kps = self.__capture.keypoints[instance_id] if self.__capture.keypoints is not None else None

        if kps is None:
            return self

        for kp in kps.keypoints:
            cv2.circle(self.__img, (int(kp.location[0]), int(kp.location[1])), radius, color, -1)
            cv2.putText(self.__img, str(kp.index), (int(kp.location[0]), int(kp.location[1]), cv2.FONT_HERSHEY_SIMPLEX,
                                                    1, color, 2))

        return self

    def arrow(self, start, end, point_type: CoordType = CoordType.CAM, color=(0, 0, 255), thickness=2):
        start = self.__convert_point(start, point_type)
        end = self.__convert_point(end, point_type)

        cv2.arrowedLine(self.__img, tuple(start), tuple(end), color, thickness)

        return self

    def line(self, start, end, point_type: CoordType = CoordType.CAM, color=(0, 0, 255), thickness=2):
        start = self.__convert_point(start, point_type)
        end = self.__convert_point(end, point_type)

        cv2.line(self.__img, tuple(start), tuple(end), color, thickness)

        return self

    def point(self, point, point_type: CoordType = CoordType.CAM, color=(0, 255, 0), radius=5):
        point = self.__convert_point(point, point_type)
        cv2.circle(self.__img, tuple(point), radius, color, -1)

        return self

    def points(self, points, point_type: CoordType = CoordType.CAM, connect=False, color=(0, 255, 0), radius=5,
               thickness=2):
        for point in points:
            self.point(point, point_type, color, radius)

        if connect:
            for i in range(len(points) - 1):
                start = self.__convert_point(points[i], point_type)
                end = self.__convert_point(points[i + 1], point_type)
                cv2.line(self.__img, tuple(start), tuple(end), color, thickness)

        return self

    def polygons(self, polygons: List[Polygon], point_type: CoordType = CoordType.CAM, color=(0, 255, 0), thickness=2):
        for polygon in polygons:
            bound_x, bound_y = polygon.boundary.coords.xy[0], polygon.boundary.coords.xy[1]
            points = [(x, y) for x, y in zip(bound_x, bound_y)]
            points = [self.__convert_point(point, point_type) for point in points]

            for i in range(len(points) - 1):
                start = points[i]
                end = points[i + 1]
                cv2.line(self.__img, tuple(start), tuple(end), color, thickness)

        return self

    def save(self, path):
        cv2.imwrite(path, self.__img)

        return self

    def get_img(self):
        return self.__img
