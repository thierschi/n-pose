import cv2

from import_data.unity_data import UnityCapture, Unity2DBoundingBoxValue, UnityKeypointValue


def draw_capture_info(img: cv2.typing.MatLike, capture: UnityCapture):
    cv2.putText(img, f"Sequence {capture.sequence} - ID {capture.id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    return img


def draw_bounding_box(img: cv2.typing.MatLike, bounding_box: Unity2DBoundingBoxValue):
    x = int(bounding_box.origin[0])
    y = int(bounding_box.origin[1])
    w = int(bounding_box.dimension[0])
    h = int(bounding_box.dimension[1])

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, f'{bounding_box.label.unity_id} {bounding_box.label.name}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return img


def draw_keypoint_value(img: cv2.typing.MatLike, capture: UnityCapture, keypoint_value: UnityKeypointValue):
    keypoints = keypoint_value.keypoints
    for keypoint in keypoints:
        location = keypoint.location
        x = int(location[0])
        y = int(location[1])

        if x + y == 0:
            continue

        # y = int(capture.dimension[1] - y)

        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img, f"{keypoint.index} ({keypoint.state}) - ({x},{y})", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    (255, 255, 255), 2)

    return img


def draw_arrow_to_origin(img: cv2.typing.MatLike):
    cv2.arrowedLine(img, (50, 50), (0, 0), (0, 255, 0), 2)

    return img
