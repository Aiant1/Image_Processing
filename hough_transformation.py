import cv2
import math
import numpy as nu


def Grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def GaussianBlur(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def SobelFilter(image):
    image = Grayscale(GaussianBlur(image))
    convolved = nu.zeros(image.shape)
    G_x = nu.zeros(image.shape)
    G_y = nu.zeros(image.shape)
    size = image.shape
    kernel_x = nu.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    kernel_y = nu.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            G_x[i, j] = nu.sum(nu.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_x))
            G_y[i, j] = nu.sum(nu.multiply(image[i - 1: i + 2, j - 1: j + 2], kernel_y))

    convolved = nu.sqrt(nu.square(G_x) + nu.square(G_y))
    convolved = nu.multiply(convolved, 255.0 / convolved.max())

    angles = nu.rad2deg(nu.arctan2(G_y, G_x))
    angles[angles < 0] += 180
    convolved = convolved.astype('uint8')
    return convolved, angles


def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = nu.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                value_to_compare = max(image[i, j - 1], image[i, j + 1])
            elif (22.5 <= angles[i, j] < 67.5):
                value_to_compare = max(image[i - 1, j - 1], image[i + 1, j + 1])
            elif (67.5 <= angles[i, j] < 112.5):
                value_to_compare = max(image[i - 1, j], image[i + 1, j])
            else:
                value_to_compare = max(image[i + 1, j - 1], image[i - 1, j + 1])

            if image[i, j] >= value_to_compare:
                suppressed[i, j] = image[i, j]
    suppressed = nu.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed


def double_threshold_hysteresis(image, low, high):
    weak = 50
    strong = 255
    size = image.shape
    result = nu.zeros(size)
    weak_x, weak_y = nu.where((image > low) & (image <= high))
    strong_x, strong_y = nu.where(image >= high)
    result[strong_x, strong_y] = strong
    result[weak_x, weak_y] = weak
    dx = nu.array((-1, -1, 0, 1, 1, 1, 0, -1))
    dy = nu.array((0, 1, 1, 1, 0, -1, -1, -1))
    size = image.shape

    while len(strong_x):
        x = strong_x[0]
        y = strong_y[0]
        strong_x = nu.delete(strong_x, 0)
        strong_y = nu.delete(strong_y, 0)
        for direction in range(len(dx)):
            new_x = x + dx[direction]
            new_y = y + dy[direction]
            if ((new_x >= 0 & new_x < size[0] & new_y >= 0 & new_y < size[1]) and (result[new_x, new_y] == weak)):
                result[new_x, new_y] = strong
                nu.append(strong_x, new_x)
                nu.append(strong_y, new_y)
    result[result != strong] = 0
    return result


def Canny(image, low, high):
    image, angles = SobelFilter(image)
    image = non_maximum_suppression(image, angles)
    gradient = nu.copy(image)
    image = double_threshold_hysteresis(image, low, high)
    return image, gradient


def mark(x, y, radius):
    for i in range(360):
        edgeX = int(math.cos(i) * radius + x)
        edgeY = int(math.sin(i) * radius + y)
        if edgeX >= cols:
            edgeX = edgeX - (edgeX - cols) - 1
        if edgeY >= rows:
            edgeY = edgeY - (edgeY - rows) - 1

        accumulator[edgeY, edgeX] += 1


original = cv2.imread('C:\\Users\\ASUS\\Downloads\\blood.png')

rows, cols, _ = original.shape

after_nms, gradient = Canny(original, 75, 75)
MIN_R = 10
MAX_R = 16
for i in range(MIN_R, MAX_R):
    accumulator = nu.zeros((rows, cols, 1), dtype=nu.uint8)
    for r in range(rows):
        for c in range(cols):
            if after_nms[r, c] > 200:
                mark(c, r, i)
    for r in range(rows):
        for c in range(rows):
            if accumulator[r, c] >= 150:
                cv2.circle(original, (c, r), i, (0, 0, 0), 2)

#cv2.imshow('Original', original)
#cv2.imshow('Edged', after_nms)
cv2.imwrite("C:\\Users\\ASUS\\Desktop\\PYTHON\\antika_practice_3\\res\\object_edged_detected_blood.png",after_nms)

#cv2.imshow('Accumul', accumulator)
cv2.imwrite("C:\\Users\\ASUS\\Desktop\\PYTHON\\antika_practice_3\\res\\accumulator_image_blood.png",accumulator)


cv2.waitKey(0)