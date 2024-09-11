from math import sqrt, degrees, radians, cos, sin, exp, atan2
import numpy as np

# Reference white (D65 illuminant)
WHITE_POINT = np.array([95.047, 100.0, 108.883])


# Helper function for the f(t) transformation in Lab conversion
def f(t):
    delta = 6 / 29
    if t > delta**3:
        return t ** (1 / 3)
    else:
        return (t / (3 * delta**2)) + (4 / 29)


# Helper function for the LAB to XYZ conversion
def f_inv(t: float) -> float:
    delta = 6 / 29
    if t > delta:
        return t**3
    else:
        return 3 * (delta**2) * (t - 4 / 29)


# Helper function to correct gamma in RGB conversion
def gamma_correct(value: float) -> float:
    if value <= 0.0031308:
        return 12.92 * value
    else:
        return 1.055 * (value ** (1 / 2.4)) - 0.055


def get_luminance(rgb: np.ndarray) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def hex_to_rgb(hex: str) -> np.ndarray:
    return np.array([int(hex[1:3], 16), int(hex[3:5], 16), int(hex[5:7], 16)])


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb / 255.0

    c_min = min(r, g, b)
    c_max = max(r, g, b)
    delta = c_max - c_min

    # Lightness
    l = (c_max + c_min) / 2

    # Hue
    if delta == 0:
        h = 0  # Undefined hue (gray color)
    elif c_max == r:
        h = 60 * (((g - b) / delta) % 6)
    elif c_max == g:
        h = 60 * (((b - r) / delta) + 2)
    elif c_max == b:
        h = 60 * (((r - g) / delta) + 4)

    # Saturation
    if delta == 0:
        s = 0  # If there is no chroma, saturation is 0
    else:
        s = delta / (1 - abs(2 * l - 1))

    return np.array([h, s, l])


def hsl_to_rgb(hsl: np.ndarray) -> np.ndarray:
    h, s, l = hsl
    c = (1 - abs(2 * l - 1)) * s  # Chroma
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0
    elif 120 <= h < 180:
        r1, g1, b1 = 0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0, c
    elif 300 <= h < 360:
        r1, g1, b1 = c, 0, x
    else:
        r1, g1, b1 = 0, 0, 0  # Undefined hue case

    r, g, b = [(channel + m) * 255 for channel in (r1, g1, b1)]
    return np.clip([r, g, b], 0, 255)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    rgb_to_xyz_matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )

    xyz = np.dot(rgb_to_xyz_matrix, rgb)
    xyz = xyz * 100.0
    return xyz


def xyz_to_rgb(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz / 100.0
    xyz_to_rgb_matrix = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    rgb = np.dot(xyz_to_rgb_matrix, xyz)
    rgb = np.clip([gamma_correct(c) for c in rgb], 0, 1)
    return (rgb * 255).astype(int)


def lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab

    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200

    x = WHITE_POINT[0] * f_inv(fx)
    y = WHITE_POINT[1] * f_inv(fy)
    z = WHITE_POINT[2] * f_inv(fz)

    return np.array([x, y, z])


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz_normalized = xyz / WHITE_POINT

    f_x = f(xyz_normalized[0])
    f_y = f(xyz_normalized[1])
    f_z = f(xyz_normalized[2])

    L = (116 * f_y) - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)

    return np.array([L, a, b])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    return rgb


def cie94_distance(lab1: np.ndarray, lab2: np.ndarray, kL=1, kC=1, kH=1) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    delta_L = L1 - L2
    C1 = sqrt(a1**2 + b1**2)
    C2 = sqrt(a2**2 + b2**2)
    delta_C = C1 - C2

    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_sq = delta_a**2 + delta_b**2 - delta_C**2
    delta_H = sqrt(max(0, delta_H_sq))

    S_L = 1
    S_C = 1 + 0.045 * C1
    S_H = 1 + 0.015 * C1

    delta_E94 = sqrt(
        (delta_L / (kL * S_L)) ** 2
        + (delta_C / (kC * S_C)) ** 2
        + (delta_H / (kH * S_H)) ** 2
    )

    return delta_E94


def ciede2000_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    C1 = sqrt(a1**2 + b1**2)
    C2 = sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    G = 0.5 * (1 - sqrt(C_bar**7 / (C_bar**7 + 25**7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = sqrt(a1_prime**2 + b1**2)
    C2_prime = sqrt(a2_prime**2 + b2**2)

    h1_prime = degrees(atan2(b1, a1_prime)) % 360
    h2_prime = degrees(atan2(b2, a2_prime)) % 360

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > 180:
        if h2_prime <= h1_prime:
            delta_h_prime += 360
        else:
            delta_h_prime -= 360

    delta_H_prime = 2 * sqrt(C1_prime * C2_prime) * sin(radians(delta_h_prime) / 2)

    L_bar_prime = (L1 + L2) / 2
    C_bar_prime = (C1_prime + C2_prime) / 2

    h_bar_prime = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > 180:
        if h1_prime + h2_prime < 360:
            h_bar_prime += 180
        else:
            h_bar_prime -= 180

    T = (
        1
        - 0.17 * cos(radians(h_bar_prime - 30))
        + 0.24 * cos(radians(2 * h_bar_prime))
        + 0.32 * cos(radians(3 * h_bar_prime + 6))
        - 0.20 * cos(radians(4 * h_bar_prime - 63))
    )

    delta_theta = 30 * exp(-(((h_bar_prime - 275) / 25) ** 2))

    R_C = 2 * sqrt(C_bar_prime**7 / (C_bar_prime**7 + 25**7))
    S_L = 1 + (0.015 * (L_bar_prime - 50) ** 2) / sqrt(20 + (L_bar_prime - 50) ** 2)
    S_C = 1 + 0.045 * C_bar_prime
    S_H = 1 + 0.015 * C_bar_prime * T

    R_T = -sin(2 * radians(delta_theta)) * R_C

    delta_E00 = sqrt(
        (delta_L_prime / S_L) ** 2
        + (delta_C_prime / S_C) ** 2
        + (delta_H_prime / S_H) ** 2
        + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
    )

    return delta_E00


def cie94_distance_rgb(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    print(rgb1, rgb2)
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    return cie94_distance(lab1, lab2)


def ciede2000_distance_rgb(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    return ciede2000_distance(lab1, lab2)


def darken_rgb(color: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(color * factor, 0, 255)


def lighten_hsl(hsl: np.ndarray, factor: float) -> np.ndarray:
    hsl[2] = min(hsl[2] + factor * (1 - hsl[2]), 1)
    return hsl


def darken_hsl(hsl: np.ndarray, factor: float) -> np.ndarray:
    hsl[2] = max(hsl[2] - factor * hsl[2], 0)
    return hsl


def darken_lab(lab: np.ndarray, factor: float) -> np.ndarray:
    lab[0] = max(lab[0] - factor * lab[0], 0)
    return lab


def lighten_lab(lab: np.ndarray, factor: float) -> np.ndarray:
    lab[0] = min(lab[0] + factor * (100 - lab[0]), 100)
    return lab


def to_rgb(rgba: np.ndarray) -> np.ndarray:
    return rgba[:3]
