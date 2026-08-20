import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corners(img):
    # Convert to float
    gray = np.float32(img)

    # Harris Corner Detection
    dst = cv2.cornerHarris(
        gray,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # Dilate so corners are easier to see
    dst = cv2.dilate(dst, None)

    # Threshold
    corners = dst > 0.01 * dst.max()

    return corners

img1 = cv2.imread(r"S:\Computer Vision\Lab1\Part3\images\img1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"S:\Computer Vision\Lab1\Part3\images\img2.png", cv2.IMREAD_GRAYSCALE)

img3 = cv2.imread(r"S:\Computer Vision\Lab1\Part3\images\img3.png", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread(r"S:\Computer Vision\Lab1\Part3\images\img4.png", cv2.IMREAD_GRAYSCALE)



corners1 = harris_corners(img1)
corners2 = harris_corners(img2)

corners3 = harris_corners(img3)
corners4 = harris_corners(img4)



def show_corners(img, corners, title):

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    result[corners] = [255, 0, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(result)
    plt.title(title)
    plt.axis("off")
    plt.show()

show_corners(img1, corners1, "Pair 1 - Original Harris Corners")
show_corners(img2, corners2, "Pair 1 - Rotated Harris Corners")

show_corners(img3, corners3, "Pair 2 - Tilted Image 1 Harris Corners")
show_corners(img4, corners4, "Pair 2 - Tilted Image 2 Harris Corners")


h, w = img1.shape

M = cv2.getRotationMatrix2D(
    (w / 2, h / 2),
    180,
    1
)

aligned1 = cv2.warpAffine(
    img2,
    M,
    (w, h)
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img1, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap="gray")
plt.title("180° Rotated")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(aligned1, cmap="gray")
plt.title("Aligned")
plt.axis("off")

plt.show()



def select_points(img, title):

    points = []

    def click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            points.append([x, y])

            cv2.circle(
                display,
                (x, y),
                5,
                255,
                -1
            )

            cv2.imshow(title, display)

    display = img.copy()

    cv2.imshow(title, display)
    cv2.setMouseCallback(title, click)

    print("Click 4 corresponding points")

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == 13:   # Enter
            break

    cv2.destroyAllWindows()

    return np.float32(points)


points1 = select_points(img3, "Tilted Image 1")
points2 = select_points(img4, "Tilted Image 2")

H, mask = cv2.findHomography(
    points2,
    points1,
    cv2.RANSAC
)


h, w = img3.shape

aligned2 = cv2.warpPerspective(
    img4,
    H,
    (w, h)
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img3, cmap="gray")
plt.title("Tilted Image 1")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img4, cmap="gray")
plt.title("Tilted Image 2")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(aligned2, cmap="gray")
plt.title("Aligned Image")
plt.axis("off")

plt.show()