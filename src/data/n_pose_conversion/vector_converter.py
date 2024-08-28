import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from data.unity_data import Capture
from util import get_keypoint_usability, point_to_world, get_object_direction, normalize_vector
from util.polygons import detect_colored_polygons, simplify_polygon_group, get_polygon_boundary


class VectorConverter:
    def __init__(self):
        self._capture = None
        self._instance_id = None
        self._input_vector = []
        self._output_vector = []
        self._was_successful = True

    def _assert_requirements(self):
        assert self._capture is not None
        assert self._instance_id is not None

    def use(self, capture: Capture, instance_id: int):
        self._capture = capture
        self._instance_id = instance_id

    @property
    def input_vector(self):
        return self._input_vector

    @property
    def output_vector(self):
        return self._output_vector

    @property
    def was_successful(self):
        return self._was_successful

    def keypoints(self, v_size: int):
        self._assert_requirements()
        assert v_size % 2 == 0

        kp_value = self._capture.keypoints[self._instance_id]
        vector = np.zeros(v_size)

        for i, kp in enumerate(kp_value.keypoints):
            if i * 2 >= v_size:
                break

            vector[i * 2] = kp.location[0] / self._capture.dimension[0]
            vector[i * 2 + 1] = kp.location[1] / self._capture.dimension[1]

        self._input_vector.extend(vector.tolist())
        return vector

    def instance_segmentation(self, v_size: int):
        self._assert_requirements()
        assert v_size % 2 == 0
        target = v_size // 2

        seg_map = cv2.imread(self._capture.instance_segmentation.file_path)
        instance = self._capture.instance_segmentation[self._instance_id]
        _, polygons = detect_colored_polygons(seg_map, instance.color)

        try:
            simple_polygon = simplify_polygon_group(polygons, target=v_size // 2, tolerance=1e-10)
            x, y = get_polygon_boundary(simple_polygon)
        except NotImplementedError:
            self._was_successful = False
            return None

        if len(x) > target:
            self._was_successful = False
            return None

        vector = np.zeros(v_size)
        for i, x in enumerate(x):
            if i * 2 >= v_size:
                break

            vector[i * 2] = x
            vector[i * 2 + 1] = y[i]

        self._input_vector.extend(vector.tolist())
        return vector

    def position(self):
        pass

    def direction(self):
        pass


class KeypointBasedVectorConverter(VectorConverter):
    def __init__(self):
        super().__init__()

    def position(self):
        self._assert_requirements()
        kp_value = self._capture.keypoints[self._instance_id]

        obj_kps = kp_value.keypoints
        camera_pos = self._capture.position
        camera_rot = self._capture.rotation

        obj_kps.sort(key=lambda x: x.index)
        kp_usability = get_keypoint_usability(obj_kps)

        points = [kp.camera_cartesian_location for kp in obj_kps]
        m, *_ = np.array(points)
        m_usable, *_ = kp_usability

        if not m_usable:
            self._was_successful = False
            return None
        position = m

        position = point_to_world(position, camera_pos, camera_rot)

        self._output_vector.extend(position.tolist())
        return position

    def direction(self):
        self._assert_requirements()
        kp_value = self._capture.keypoints[self._instance_id]

        obj_kps = kp_value.keypoints
        camera_pos = self._capture.position
        camera_rot = self._capture.rotation

        obj_kps.sort(key=lambda x: x.index)
        kp_usability = get_keypoint_usability(obj_kps)

        points = [kp.camera_cartesian_location for kp in obj_kps]
        m, *_ = np.array(points)
        m_usable, *_ = kp_usability

        direction = get_object_direction(obj_kps)

        if not m_usable or direction is None:
            self._was_successful = False
            return None
        position = m
        direction_ep = position + direction

        position = point_to_world(position, camera_pos, camera_rot)
        direction_ep = point_to_world(direction_ep, camera_pos, camera_rot)

        world_direction = direction_ep - position
        world_direction = normalize_vector(world_direction, .05)

        self._output_vector.extend(world_direction.tolist())
        return world_direction


class TransformerBasedVectorConverter(VectorConverter):
    def __init__(self):
        super().__init__()

    def position(self):
        self._assert_requirements()
        t = self._capture.transforms[self._instance_id]
        pos = t.position

        self._output_vector.extend(pos)
        return pos

    def direction(self):
        self._assert_requirements()
        t = self._capture.transforms[self._instance_id]

        rot = R.from_quat(t.rotation)
        direction = rot.apply(np.array([-1, 0, 0]))

        direction = normalize_vector(direction, .05)

        self._output_vector.extend(direction.tolist())
        return direction
