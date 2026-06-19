from scipy.special import binom
import random
import numpy as np
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform
from typing import Sequence


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, Sequence):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


def center_crop(image, new_shape):
    center = tuple(ti // 2 for ti in image.shape)
    w, h = new_shape
    x = center[1] - w // 2
    y = center[0] - h // 2

    return image[int(y):int(y + h), int(x):int(x + w)]


class AddDustParticles(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):
        super(AddDustParticles, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return addDustParticles(img, mode='bernstein')


class AddStructuralDefects(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):
        super(AddStructuralDefects, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return addStructuralDefects(img)


class AddLaserPointer(ImageOnlyTransform):

    def __init__(self, always_apply=False, p=0.5):
        super(AddLaserPointer, self).__init__(always_apply, p)

    def apply(self, img, **params):
        # detect laser, if it already exists do not add augmentation
        is_laser = detect_laser(img)
        if is_laser:
            # cv2.putText(img,"Found laser!",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
            return img
        else:
            return addLaserPointer(img)


class AddOpticalFiber(ImageOnlyTransform):

    def __init__(
            self,
            fibers_density=(0.2, 0.35),
            dark_gradient_inside=True,
            draw_simple_cladding=False,
            gradient_fill_ratio=(0.9, 1.0),
            inverted_gradient_fill_ratio=(0.4, 0.8),
            gradient_overfill_ratio=1.1,
            quality=[8, 10, 12],
            cladding_colour=(16, 32),
            cladding_width=[1, 2, 4],
            invert_gradient_p=0.5,
            always_apply=False,
            p=0.5
    ):
        super(AddOpticalFiber, self).__init__(always_apply, p)
        self.dark_gradient_inside = dark_gradient_inside
        self.draw_simple_cladding = draw_simple_cladding
        self.fibers_density_limit = to_tuple(fibers_density)
        self.gradient_fill_ratio_limit = to_tuple(gradient_fill_ratio)
        self.inverted_gradient_fill_ratio_limit = to_tuple(inverted_gradient_fill_ratio)
        assert gradient_overfill_ratio >= 1.0, 'overfill ratio must be larger or equal to 1.0'
        self.gradient_overfill_ratio_limit = gradient_overfill_ratio
        self.cladding_colour_limit = to_tuple(cladding_colour)
        self.cladding_width_vals = cladding_width
        self.quality_vals = quality
        self.invert_gradient_p = invert_gradient_p

    def apply(self, img, fibers_density=0.15, gradient_fill_ratio=0.9, inverted_gradient_fill_ratio=0.4,
              gradient_overfill_ratio=1.0, cladding_colour=16, cladding_width=1, quality=1, **params):
        return addOpticalFiber(img, fibers_density, self.dark_gradient_inside, self.draw_simple_cladding,
                               cladding_colour, cladding_width, gradient_fill_ratio, inverted_gradient_fill_ratio,
                               gradient_overfill_ratio, quality, self.invert_gradient_p)

    def get_params(self):
        return {
            "fibers_density": random.uniform(self.fibers_density_limit[0], self.fibers_density_limit[1]),
            "gradient_fill_ratio": random.uniform(self.gradient_fill_ratio_limit[0], self.gradient_fill_ratio_limit[1]),
            "inverted_gradient_fill_ratio": random.uniform(self.inverted_gradient_fill_ratio_limit[0],
                                                           self.inverted_gradient_fill_ratio_limit[1]),
            "gradient_overfill_ratio": random.uniform(1.0, self.gradient_overfill_ratio_limit),
            "cladding_colour": random.randint(self.cladding_colour_limit[0], self.cladding_colour_limit[1]),
            "cladding_width": random.choice(self.cladding_width_vals),
            "quality": random.choice(self.quality_vals),
        }

    def get_transform_init_args_names(self):
        return ("fibers_density_limit", "dark_fill_ratio_limit", "quality_vals", "dark_gradient_inside")


# BEZIER SEGMENTS

def bernstein(n, k, p): return binom(n, k) * p ** k * (1. - p) ** (n - k)


def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve


class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2],
                      points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve


def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


def get_bezier_curve(a, rad=0.2, edgy=0):
    """
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest.
    """
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])

    def f(ang): return (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)

    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)


def create_bernstein_shape(image, center, intensity, scale, anchors_num=(4, 8), rad=0.25, edgy=0.5):
    a = get_random_points(n=np.random.randint(*anchors_num), scale=scale)
    x, y, _ = get_bezier_curve(a, rad=rad, edgy=edgy)
    points = np.stack((x, y), axis=-1)
    points = points + center
    points = points.astype(np.int32)
    cv2.fillPoly(image, pts=[points], color=intensity, lineType=cv2.LINE_AA)
    return image


# OVERLAY OF "DUST" PARTICLES

def motionblur(img, size=15, angle=45):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    kernel_motion_blur = cv2.warpAffine(kernel_motion_blur,
                                        cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0),
                                        (size, size))
    kernel_motion_blur = kernel_motion_blur / size
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output


def generatePoints(points_number, image_size, scale=0.9, circle=True, minimal_r=0.0):
    px_centers = []
    py_centers = []
    # circle area generator
    if circle:
        R = 1
        for _ in range(points_number):
            if minimal_r == 0:
                r = R * np.sqrt(random.random())
            else:
                r = R * np.sqrt(random.uniform(minimal_r, 1.0))
            theta = random.random() * 2 * np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            x = image_size / 2 + (image_size / 2.0 * x) * scale
            y = image_size / 2 + (image_size / 2.0 * y) * scale
            px_centers.append(int(x))
            py_centers.append(int(y))
    else:
        # square area generator
        for _ in range(points_number):
            x = int(random.uniform(0, image_size - 1))
            y = int(random.uniform(0, image_size - 1))
            px_centers.append(int(x))
            py_centers.append(int(y))

    return px_centers, py_centers


def addDustParticles(input_img, mode='ellipse', motionblur_p=0.5):
    # PARAMS

    FOV_ratio = 0.85

    # gaussian blur
    is_gaussian_blur = False
    blur_kernel_size = 5
    blur_x_sigma = random.uniform(1.0, 3.0)
    blur_y_sigma = random.uniform(1.0, 3.0)

    # particles size and shape
    intensity_range = (40, 80)
    particles_count = int(random.uniform(5, 20))
    mean_radius = 8
    size_variation = 7

    # motion blur strenght and direction
    height, width, _ = input_img.shape
    angle = int(random.uniform(-90, 90))
    blur_strenght = 0.07
    blur_value_floor = int(height * blur_strenght)
    blur_varation = 3
    blur_value = random.randint(blur_value_floor, blur_value_floor * blur_varation)
    blur_value = int(np.ceil(blur_value / 2.) * 2 + 1)
    if blur_value < 3:
        blur_value = 3

    img = np.zeros((height, width), dtype=np.uint8)

    # circle generator
    px_centers, py_centers = generatePoints(particles_count, height, FOV_ratio)

    if mode == 'ellipse':
        for i in range(particles_count):
            radius1 = int(random.uniform(mean_radius - size_variation, mean_radius + size_variation))
            radius2 = int(random.uniform(mean_radius - size_variation, mean_radius + size_variation))
            ellipse_angle = int(random.uniform(-90, 90))
            intensity = int(random.uniform(intensity_range[0], intensity_range[1]))
            cv2.ellipse(img, center=(px_centers[i], py_centers[i]), startAngle=0, endAngle=360, axes=(radius1, radius2),
                        color=intensity, thickness=-1, angle=ellipse_angle, lineType=cv2.LINE_AA)
    elif mode == 'bernstein':
        for i in range(particles_count):
            scale = int(random.uniform(5, 30))
            center = (px_centers[i], py_centers[i])
            intensity = int(random.uniform(intensity_range[0], intensity_range[1]))
            img = create_bernstein_shape(img, center, intensity, scale)

    if is_gaussian_blur:
        img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), blur_x_sigma, blur_y_sigma)

    if (random.random() < motionblur_p):
        img = motionblur(img, size=blur_value, angle=angle)
    else:
        img = cv2.GaussianBlur(img, (11, 11), 5, 5)

    output = cv2.add(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), input_img)
    return output


### ADD DIRT AND STRUCTURAL DEFECTS

def addStructuralDefects(input_img):
    # PARAMS
    FOV_ratio = 0.85

    # particles size and shape
    intensity_range = (150, 200)
    small_defects_count = int(random.uniform(3, 12))
    small_defects_blur_count = int(random.uniform(3, 12))
    large_defects_count = int(random.uniform(1, 4))

    height, width, _ = input_img.shape

    # circle generator
    px_centers, py_centers = generatePoints(small_defects_count, height, FOV_ratio)
    px_centers_blur, py_centers_blur = generatePoints(small_defects_blur_count, height, FOV_ratio)
    px_centers_large, py_centers_large = generatePoints(large_defects_count, height, FOV_ratio, minimal_r=0.95)

    # small_blur
    canvas = np.zeros((height, width), dtype=np.uint8)

    for i in range(small_defects_count):
        scale_small = int(random.uniform(2, 20))
        center = (px_centers[i], py_centers[i])
        intensity = int(random.uniform(intensity_range[0], intensity_range[1]))
        canvas = create_bernstein_shape(canvas, center, intensity, scale_small, anchors_num=(8, 12))
    canvas = cv2.GaussianBlur(canvas, (3, 3), 5, 5)
    input_img = cv2.subtract(input_img, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    # large blur
    canvas = np.zeros((height, width), dtype=np.uint8)
    for i in range(small_defects_blur_count):
        scale_small = int(random.uniform(4, 30))
        center = (px_centers_blur[i], py_centers_blur[i])
        intensity = int(random.uniform(intensity_range[0], intensity_range[1]))
        canvas = create_bernstein_shape(canvas, center, intensity, scale_small, anchors_num=(8, 12))
    canvas = cv2.GaussianBlur(canvas, (15, 15), 5, 5)
    input_img = cv2.subtract(input_img, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    # large defects
    canvas = np.zeros((height, width), dtype=np.uint8)
    for i in range(large_defects_count):
        scale_large = int(random.uniform(40, 60))
        center = (px_centers_large[i], py_centers_large[i])
        intensity = int(random.uniform(intensity_range[0], intensity_range[1]))
        canvas = create_bernstein_shape(canvas, center, 200, scale_large, anchors_num=(8, 12), rad=0.2, edgy=2.0)
    kernel = random.choice([5, 7, 9, 11])
    canvas = cv2.GaussianBlur(canvas, (kernel, kernel), 5, 5)
    input_img = cv2.subtract(input_img, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    return input_img


### LASER POINTER OVERLAY

def addLaserPointer(input_img, alpha=(0.2, 0.4), size=(0.1, 0.3), center_area=0.6, oe_p=0.3):
    if len(alpha) > 1:
        alpha = random.uniform(alpha[0], alpha[1])
    if len(size) > 1:
        size_factor = random.uniform(size[0], size[1])

    px, py = generatePoints(1, input_img.shape[0], center_area)
    img = np.zeros(input_img.shape[0:2], dtype=np.uint8)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    idx = random.randint(0, 2)
    laser_size = int(input_img.shape[0] * size_factor)

    laser_brightness = random.randint(50, 150)
    cv2.circle(img, (px[0], py[0]), laser_size, 255, -1, cv2.LINE_AA)
    img = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    img = cv2.normalize(img, img, 0, laser_brightness, cv2.NORM_MINMAX, cv2.CV_8U)
    img = cv2.blur(img, (11, 11))
    mask = cv2.threshold(cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)[1]
    input_img[:, :, idx] = cv2.add(img, input_img[:, :, idx], mask=mask)

    # overexposure
    if random.random() < oe_p:
        overexposure_brightness = random.randint(50, 200)
        over_exp_ratio = random.uniform(0.7, 0.9)
        img2 = np.zeros(input_img.shape[0:2], dtype=np.uint8)
        cv2.circle(img2, (px[0], py[0]), int(laser_size * over_exp_ratio), 255, -1, cv2.LINE_AA)
        img2 = cv2.distanceTransform(img2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        img2 = cv2.normalize(img2, img2, 0, overexposure_brightness, cv2.NORM_MINMAX, cv2.CV_8U)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        input_img = cv2.add(img2, input_img)

    return input_img


### LASER DETECTION

def get_std_map(img):
    img_std = np.std(img, axis=2).astype(np.uint8)
    out = cv2.medianBlur(img_std, 5)
    out = cv2.normalize(out, out, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return out


def get_cnt_circ(cnt):
    area_cnt = cv2.contourArea(cnt)
    arcLength = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area_cnt / (arcLength ** 2)
    return circularity


def detect_laser(img):
    is_laser = False

    # PARAMS
    initial_treshold = 150
    minimal_saturation = 10
    min_pointer_perc = 0.0015
    max_pointer_perc = 0.25
    min_circularity = 0.75
    min_laser_brightness = 100

    # mask circle view
    width, height = img.shape[0:2]
    mask = np.zeros(img.shape, np.uint8)
    center = (int(width / 2.0), int(width / 2.0))
    cv2.circle(mask, center, int(center[0] * 0.7), (1, 1, 1), -1, cv2.LINE_AA)
    img = img.copy() * mask

    # laser spot - std per pixel
    img = cv2.medianBlur(img, 5)
    laser_std = get_std_map(img)
    _, pretresholded = cv2.threshold(laser_std, initial_treshold, 255, cv2.THRESH_BINARY)
    tresholded = cv2.threshold(pretresholded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # hsv treshold - only colors
    img_copy = img * cv2.cvtColor((tresholded / 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img_copy_hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    hsv_container = np.zeros(img.shape[0:2], dtype=np.int32)
    green = cv2.inRange(img_copy_hsv, (30, minimal_saturation, 0), (80, 255, 255))
    blue = cv2.inRange(img_copy_hsv, (100, minimal_saturation, 0), (130, 255, 255))
    magenta = cv2.inRange(img_copy_hsv, (140, minimal_saturation, 0), (160, 255, 255))
    red1 = cv2.inRange(img_copy_hsv, (165, minimal_saturation, 0), (179, 255, 255))
    red2 = cv2.inRange(img_copy_hsv, (0, minimal_saturation, 0), (20, 255, 255))
    hsv_container = green + blue + magenta + red1 + red2
    hsv_container = cv2.normalize(hsv_container, hsv_container, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # apply hsv mask
    tresholded = (tresholded * (hsv_container / 255.0).astype(np.uint8)).astype(np.uint8)

    # get and filter contours
    cnt, _ = cv2.findContours(tresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour = int(width * height * min_pointer_perc)
    max_contour = int(width * height * max_pointer_perc)
    cnt = [c for c in cnt if (max_contour > cv2.contourArea(c) > min_contour) and get_cnt_circ(c) > min_circularity]

    if len(cnt) > 0:
        # draw ellipse-contour
        rot_rect = cv2.fitEllipse(cnt[0])
        center = (int(rot_rect[0][0]), int(rot_rect[0][1]))
        axes = (int(rot_rect[1][0] / 2.0), int(rot_rect[1][1] / 2.0))
        # analyse mean brightness of suspected area (inside ellipse)
        mask2 = np.zeros(img.shape, np.uint8)
        mask2 = cv2.ellipse(mask2, center, axes, rot_rect[2], 0, 360, (1, 1, 1), -1, cv2.LINE_AA)
        masked_area = img * mask2
        mean = masked_area[masked_area > 0].mean()
        if mean > min_laser_brightness:
            is_laser = True

    return is_laser


### ADD PATTERN OF OPTICAL FIBER HEXAGON
MAX_VAL = {
    np.dtype("uint8"): 2 ** 8 - 1,
    np.dtype("uint16"): 2 ** 16 - 1,
    np.dtype("uint32"): 2 ** 32 - 1,
    np.dtype("float32"): 1.0,
}


def adjust_brightness_simple(image, val=1.):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    img = np.array(img, dtype=np.float64)

    img[:, :, 1] *= val
    img[:, :, 1][img[:, :, 1] > 255] = 255

    img[:, :, 2] *= val
    img[:, :, 2][img[:, :, 1] > 255] = 255

    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return img


def adjust_brightness(img, alpha=1.0, beta=0.0, max_brightness=False):
    dtype = img.dtype
    max_value = MAX_VAL[dtype]

    lut = np.arange(0, max_value + 1).astype("float32")

    if alpha != 1:
        lut *= alpha

    # offset
    if beta != 0:
        if max_brightness:
            lut += beta * max_value
        else:
            lut += beta * np.mean(img)

    lut = np.clip(lut, 0, max_value).astype(dtype)
    img = cv2.LUT(img, lut)
    return img


# create circle filled with gradient - black inside, white outside
def get_gradient_map(radius, gradient_fill=0.5, margin=4, invert=True):
    map = np.zeros((2 * radius + margin, 2 * radius + margin), np.uint8)
    cv2.circle(map, (radius + margin // 2, radius + margin // 2), int(radius * gradient_fill), 255, -1, cv2.LINE_8)
    map = cv2.distanceTransform(map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    map = cv2.normalize(map, map, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    if invert:
        map = ~map
    return map


# mask gradient which is circle or square shape with hexagon shape mask
def mask_gradient_map(grad_mask, scale, gradient_overfill=1.0, hex_scale=1.04):
    grad_hex = np.ones(grad_mask.shape, dtype=np.uint8) * 255
    hexagon = get_polygon_vertices(6)
    hex = affine_transform(np.expand_dims(hexagon, axis=0),
                           scale=(scale * hex_scale, scale * hex_scale),
                           translate=(grad_mask.shape[1] // 2, grad_mask.shape[0] // 2))
    cv2.fillPoly(grad_hex, np.ceil(hex).astype(int), 0, cv2.LINE_8)
    if gradient_overfill > 1.0:
        grad_mask = cv2.resize(grad_mask, (0, 0), fx=gradient_overfill,
                               fy=gradient_overfill, interpolation=cv2.INTER_LINEAR)
        grad_mask = center_crop(grad_mask, grad_hex.shape)
    grad_mask = cv2.bitwise_or(grad_mask, grad_hex)
    return grad_mask


# multiply 3CH image with gradient mask within region of interest (ROI)
# ROI is calculated based on object center and radius
def apply_grad(img, grad_mask, center, radius):
    # image center roi
    min_x = center[0] - radius
    min_y = center[1] - radius
    max_x = center[0] + radius + 1
    max_y = center[1] + radius + 1
    # gradient mask roi
    mask_min_x = grad_mask.shape[0] // 2 - radius
    mask_min_y = grad_mask.shape[1] // 2 - radius
    mask_max_x = grad_mask.shape[0] // 2 + radius + 1
    mask_max_y = grad_mask.shape[1] // 2 + radius + 1

    # apply gradient if it fits the fragment
    # if max_x - min_x == mask_max_x - mask_min_x and max_y - min_y == mask_max_y - mask_min_y:
    if len(img.shape) == 3:
        mask = grad_mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x, :] / 255
        img_roi = img[min_y:max_y, min_x:max_x, :]
        if mask.shape == img_roi.shape:
            img[min_y:max_y, min_x:max_x, :] = (img_roi * mask).astype(np.uint8)
        else:
            return img
        # img[min_y:max_y, min_x:max_x,:] =  (img[min_y:max_y, min_x:max_x,:] * (grad_mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x,:]/255)).astype(np.uint8)
    else:
        img[min_y:max_y, min_x:max_x] = (img[min_y:max_y, min_x:max_x] * (
                grad_mask[mask_min_y:mask_max_y, mask_min_x:mask_max_x] / 255)).astype(np.uint8)
    return img


def make_grid(nx, ny, min_diam) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the coordinates of the hexagon centers
    """
    ratio = np.sqrt(3) / 2

    coord_x, coord_y = np.meshgrid(
        np.arange(nx), np.arange(ny), sparse=False, indexing='xy')
    coord_y = coord_y * ratio
    coord_x = coord_x.astype('float')
    coord_x[1::2, :] += 0.5
    coord_x = coord_x.reshape(-1, 1)
    coord_y = coord_y.reshape(-1, 1)

    coord_x *= min_diam  # Scale to requested size
    coord_y = coord_y.astype('float') * min_diam

    return coord_x, coord_y


def create_hex_grid(nx: int = 5,
                    ny: int = 5,
                    min_diam: float = 1.,
                    ) -> np.ndarray:
    """
    Creates and prints hexagonal lattices.
    :param nx: Number of horizontal hexagons in rectangular grid, [nx * ny]
    :param ny: Number of vertical hexagons in rectangular grid, [nx * ny]
    :param min_diam: Minimal diameter of each hexagon.
    """
    coord_x, coord_y = make_grid(nx, ny, min_diam)

    return np.hstack([coord_x, coord_y])


# assume 3CH RGB(0-255) image
def sample_colors_from_image_by_grid(img, x_coords, y_coords):
    abs_min = np.min([x_coords.T, y_coords.T])
    abs_max = np.max([x_coords.T, y_coords.T]) + 0.001
    minor_image_dim = min(img.shape[0], img.shape[1])
    p_x = np.floor((x_coords - abs_min) / (abs_max - abs_min) * minor_image_dim)
    p_y = np.floor((y_coords - abs_min) / (abs_max - abs_min) * minor_image_dim)
    colors = img[p_y.astype('int'), p_x.astype('int'), :]
    return colors


def affine_transform(points, scale=(1, 1), translate=(0, 0)):
    M = np.array([[scale[0], 0, translate[0]], [0, scale[1], translate[1]], [0, 0, 1]], dtype=np.float32)
    points = cv2.perspectiveTransform(points, M)
    return points


def get_polygon_vertices(numVertices=6):
    theta = ((2 * np.pi / numVertices) * np.arange(numVertices + 1) + np.pi / 2)
    vertices = np.column_stack((np.cos(theta), np.sin(theta)))
    return vertices


# requires square image - 1:1
def addOpticalFiber(img, fibers_density=0.2, dark_gradient_inside=False, draw_simple_cladding=False,
                    cladding_colour=16, cladding_width=1, gradient_fill_ratio=0.9, inverted_gradient_fill_ratio=0.8,
                    gradient_overfill_ratio=1.0, quality=1, invert_gradient_p=0.5, compensate_brightness_p=1.0):
    """
    fiber demsity - float - the higher density the more fibers in the image - the smaller fiber diameter
    quality param - int - accepted_values: 1 - LQ, 2- MQ, 4- HQ, 8- XHQ; the higher quality, the slower interpolation
    """
    # PARAMS
    hex_scale = 1.04
    invert_gradient = False
    cladding_colour = (cladding_colour,) * 3
    base = img.shape[0]
    width = int(base * quality)
    # normalize fiber density when quality (by oversampling) changed
    fibers_density = fibers_density / quality
    fibers = int(fibers_density * width)

    if random.random() < invert_gradient_p:
        invert_gradient = True

    canvas = np.zeros((width, width, 3), dtype=np.uint8)
    nx = fibers
    dx = width / nx
    ny = nx / (np.sqrt(3) / 2)
    scale = width / nx / 2 / (np.sqrt(3) / 2)

    # OBJECTS
    hexagon = get_polygon_vertices(6)
    hex_centers = create_hex_grid(nx, ny)
    x_hex_coords = hex_centers[:, 0] * dx
    y_hex_coords = hex_centers[:, 1] * dx
    colors = sample_colors_from_image_by_grid(img, x_hex_coords, y_hex_coords)
    if invert_gradient:
        grad_mask = get_gradient_map(int(scale), inverted_gradient_fill_ratio, invert=invert_gradient)
    else:
        grad_mask = get_gradient_map(int(scale), gradient_fill_ratio, invert=invert_gradient)
    grad_mask = mask_gradient_map(grad_mask, scale, gradient_overfill_ratio, hex_scale)
    grad_mask = cv2.cvtColor(grad_mask, cv2.COLOR_GRAY2RGB)

    # PAINTING
    for idx, (hex_x, hex_y) in enumerate(zip(x_hex_coords, y_hex_coords)):
        clr = tuple([int(x) for x in list(colors[idx])])
        # hexagones that are dark (outside field of camera view) are not painted
        if all(c > 3 for c in clr):
            translated_hex = affine_transform(np.expand_dims(hexagon, axis=0), scale=(scale, scale),
                                              translate=(hex_x, hex_y))
            translated_hex = np.ceil(translated_hex).astype(int)
            cv2.fillPoly(canvas, translated_hex, clr, cv2.LINE_8)
            if dark_gradient_inside:
                # do not apply gradient on border hexagons - they are not inside image
                if hex_x > scale and hex_x < width - scale and hex_y > scale and hex_y < width - scale:
                    canvas = apply_grad(canvas, grad_mask, (int(hex_x) + 1, int(hex_y) + 1), int(scale))
            # if draw_simple_cladding:
            #     cv2.polylines(canvas, translated_hex, True, cladding_colour, cladding_width, cv2.LINE_8)

    if quality != 1:
        canvas = cv2.resize(canvas, (base, base), cv2.INTER_AREA)

    # compensate darkening due to dark cladding gradient
    if invert_gradient == False:
        if random.random() < compensate_brightness_p:
            canvas = adjust_brightness(canvas, 2.0)
    else:
        if random.random() < compensate_brightness_p:
            canvas = adjust_brightness(canvas, 1.15)

    return canvas
