from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBColor:
    """
    RGB Color class
    """
    r: int
    g: int
    b: int

    def to_array(self):
        return [self.r, self.g, self.b]

    def to_np_array(self):
        return np.array(self.to_array())


@dataclass(frozen=True)
class RGBAColor:
    """
    RGBA Color class
    """
    r: int
    g: int
    b: int
    a: int

    def to_array(self):
        return [self.r, self.g, self.b, self.a]

    def to_np_array(self):
        return np.array(self.to_array())

    def to_rgb(self, bg: RGBColor = RGBColor(0, 0, 0)) -> RGBColor:
        """
        Convert the color to an RGB color
        :param bg: Background color
        :return: RGBColor
        """
        src = np.array([self.r, self.g, self.b, self.a])
        bg = np.array([bg.r, bg.g, bg.b])

        # Normalize
        src_norm = src / 255
        bg_norm = bg / 255

        rgb = np.array([(((1 - src_norm[3]) * bg_norm[i]) + (src_norm[3] * src_norm[i]))
                        for i in range(0, 3)])
        rgb *= 255

        return RGBColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
