from typing import TYPE_CHECKING, List

import numpy as np

from .vector import normalize_vector

if TYPE_CHECKING:
    from ..data.unity_data import Keypoint


def get_keypoint_usability(kps: List['Keypoint']):
    kp_usability = []
    for kp in kps:
        usable = False
        if kp.state != 0:
            for coord in kp.camera_cartesian_location:
                if coord != 0:
                    usable = True
                    break
        kp_usability.append(usable)
    return kp_usability


def get_object_direction(obj_kps: List['Keypoint']):
    obj_kps.sort(key=lambda x: x.index)
    kp_usability = get_keypoint_usability(obj_kps)

    points = [kp.camera_cartesian_location for kp in obj_kps]
    m, fl, fr, bl, br = np.array(points)
    m_usable, fl_usable, fr_usable, bl_usable, br_usable = kp_usability

    if len([x for x in kp_usability if x]) < 3:
        return None

    a, b, c = None, None, None
    inversion_needed = False

    if fl_usable and fr_usable:
        a = fl
        b = fr
        c = m if m_usable else bl if bl_usable else br if br_usable else None
        inversion_needed = False
    elif bl_usable and br_usable:
        a = bl
        b = br
        c = m if m_usable else fl if fl_usable else fr if fr_usable else None
        inversion_needed = True

    if a is None or b is None or c is None:
        return None

    direction = np.cross(b - a, np.cross(b - a, b - c))
    direction = normalize_vector(direction) / 10

    if inversion_needed:
        direction = -direction

    return direction
