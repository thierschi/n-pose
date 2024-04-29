import numpy as np
import cv2
from shapely.geometry import Polygon


def mask_to_polygons(mask_path):
    '''
    Converts an image mask into polygons. Returns two lists:
    - List of unnormalised shapely polygons.
    - List of normalised shapely polygons (coordinates between 0 and 1).

    Args:
        img_path (str): Path to the original image file.
        mask_path (str): Path to the grayscale mask file.
    '''

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the contours
    mask = mask.astype(bool)
    # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert contours to Label Studio polygons
    polygons = []
    normalized_polygons = []
    for contour in contours:

        # I put it in a try because the polygon extraction that the opencv does from the mask
        # sometimes generates polygons of less than 4 vertices, which makes no sense because they are not closed,
        # causing it to fail when converting to shapely polygons.

        try:
            polygon = contour.reshape(-1, 2).tolist()

            # we normalise the coordinates between 0 and 1 because this is required by YOLOv8
            normalized_polygon = [[round(coord[0] / mask.shape[1], 4), round(coord[1] / mask.shape[0], 4)] for coord in
                                  polygon]

            # Convert to shapely polygon object (not normalised)
            polygon_shapely = Polygon(polygon)
            simplified_polygon = polygon_shapely.simplify(0.85, preserve_topology=True)
            polygons.append(simplified_polygon)

            # normalise
            normalized_polygons.append(Polygon(normalized_polygon))


        except Exception as e:
            pass

    return polygons, normalized_polygons

