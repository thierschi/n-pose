from shapely.geometry import Polygon


def get_polygon_boundary(polygon: Polygon):
    return polygon.boundary.coords.xy[0], polygon.boundary.coords.xy[1]
