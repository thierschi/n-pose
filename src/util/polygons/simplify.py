from shapely.geometry import Polygon

from .util import get_polygon_boundary


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
