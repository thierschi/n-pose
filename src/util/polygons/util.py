from shapely.geometry import Polygon


def get_polygon_boundary(polygon: Polygon):
    """
    Helper function to get the boundary of a polygon
    :param polygon:
    :return: x, y coordinates of the boundary as a tupel (x_coords, y_coords)
    """
    return polygon.boundary.coords.xy[0], polygon.boundary.coords.xy[1]
