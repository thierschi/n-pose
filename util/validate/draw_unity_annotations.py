import cv2
from progressbar.bar import ProgressBar

from import_data.unity_data import UnityData, UnityKeypointValue

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (0, 64, 0),
    (0, 0, 64),
    (64, 64, 0),
    (64, 0, 64),
    (0, 64, 64),
]


def draw_keypoints(image, keypoints: UnityKeypointValue, color=(0, 255, 0), width=1920, height=1080, radius=5):
    for v in keypoints.keypoints:
        x = int(v.location[0])
        y = int(v.location[1])
        cv2.circle(image, (x, y), radius, color)

    return image


folder_name = 'solo_95'
data = UnityData(f'../../data/{folder_name}')

with ProgressBar(max_value=len(data.captures)) as bar:
    for capture in data.captures:
        bar.next()
        img = cv2.imread(capture.file_path)

        for i, kp in enumerate(capture.keypoints.values):
            color = colors[i % len(colors)]
            draw_keypoints(img, kp, color=color)

        cv2.putText(img, f'Sequence {capture.sequence}/{capture.id} - {folder_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(f'../../data/solo_annotated/{capture.sequence}_{capture.id}_{folder_name}.png', img)
