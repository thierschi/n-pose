import cv2
import numpy as np
from shapely.geometry import Polygon

from util.color import RGBColor, RGBAColor


def get_polygons_from_mask(mask: cv2.typing.MatLike, precision=6):
    # Code from Víctor Chaparro Parra (https://github.com/vchaparro)
    # Copied from https://github.com/ultralytics/ultralytics/issues/3085

    # Calculate the contours
    mask = mask.astype(bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # convert contours to Label Studio polygons
    polygons = []
    normalized_polygons = []
    for contour in contours:
        try:
            polygon = contour.reshape(-1, 2).tolist()
            normalized_polygon = [
                [round(coord[0] / mask.shape[1], precision), round(coord[1] / mask.shape[0], precision)] for coord in
                polygon]

            polygon_shapely = Polygon(polygon)
            simplified_polygon = polygon_shapely.simplify(0.85, preserve_topology=True)
            polygons.append(simplified_polygon)

            normalized_polygons.append(Polygon(normalized_polygon))
        except Exception as e:
            pass

    return polygons, normalized_polygons


def get_colored_polygons_from_mask(mask: cv2.typing.MatLike, color: RGBColor | RGBAColor, bg_color=RGBColor(0, 0, 0),
                                   precision=6):
    rgb = color.to_array() if isinstance(color, RGBColor) else color.to_rgb(bg_color).to_array()
    bgr = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2BGR)[0][0]

    mask = cv2.inRange(mask, bgr, bgr)

    return get_polygons_from_mask(mask, precision)
