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


def get_polygon_boundary(polygon: Polygon):
    return polygon.boundary.coords.xy[0], polygon.boundary.coords.xy[1]


def simplify_polygon(poly: Polygon, target: int = 50, tolerance: float = 1e-10):
    x, _ = get_polygon_boundary(poly)
    if len(x) <= target:
        return poly

    low, high = 1e-9, 1.0
    best = poly

    while low <= high:
        mid = (low + high) / 2
        simp = poly.simplify(mid, preserve_topology=True)
        x, _ = get_polygon_boundary(simp)
        bound_len = len(x)

        if bound_len == target:
            return simp
        elif bound_len <= target:
            best = simp
            high = mid - tolerance
        else:
            low = mid + tolerance

    return best


def simplify_polygon_group(polygons: list[Polygon], target: int = 50, tolerance: float = 1e-10):
    collected_bound = []
    for p in polygons:
        x, y = get_polygon_boundary(p)
        collected_bound.extend([(x[i], y[i]) for i in range(len(x))])

    collected_poly = Polygon(collected_bound)

    return simplify_polygon(collected_poly, target=target, tolerance=tolerance)

# def simplify_polygon_group(polygons: list[Polygon], target: int = 50, tolerance: float = 1e-9, min_target: int = 5):
#     lengths = [len(get_polygon_boundary(p)[0]) for p in polygons]
#     total = sum(lengths)
#     ratios = [l / total for l in lengths]
#     targets = [round(r * target) for r in ratios]
#
#     for i in range(len(targets)):
#         try:
#             if targets[i] < min_target:
#                 tmp = targets[i]
#                 targets.remove(tmp)
#                 del polygons[i]
#                 smallest = min(targets)
#                 smallest_index = targets.index(smallest)
#                 targets[smallest_index] += tmp
#         except IndexError:
#             continue
#
#     return [simplify_polygon(polygons[i], target=targets[i], tolerance=tolerance) for i in range(len(polygons))]
