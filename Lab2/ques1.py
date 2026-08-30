import cv2
import numpy as np

img = cv2.imread("./images/img1.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # Resize the image to half its original size
# Step 1: Gaussian blur to reduce noise
img_gaussian = cv2.GaussianBlur(img, (5, 5), sigmaX=7, sigmaY=7)

# Step 2: Laplacian edge detection
img_gaussian_laplacian = cv2.Laplacian(img_gaussian, cv2.CV_64F)

# Step 3: Threshold the edge response to obtain a binary edge image
for x in range(img_gaussian_laplacian.shape[0]):
    for y in range(img_gaussian_laplacian.shape[1]):
        if img_gaussian_laplacian[x, y] < 15:
            img_gaussian_laplacian[x, y] = 0
        else:
            img_gaussian_laplacian[x, y] = 255

# Keep a uint8 version for Hough transform
edge_image = np.uint8(img_gaussian_laplacian)

# Step 4: Apply Hough Transform for line detection
lines = cv2.HoughLinesP(
    edge_image,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=20,
)

# Convert grayscale image to color for overlaying detected lines
overlay_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        if line.ndim == 1:
            x1, y1, x2, y2 = line
        else:
            x1, y1, x2, y2 = line[0]
        cv2.line(overlay_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display results
cv2.imshow("Original Image", img)
cv2.imshow("Gaussian Blurred Image", img_gaussian)
cv2.imshow("Gaussian Blurred Laplacian Image", img_gaussian_laplacian)
cv2.imshow("Thresholded Gaussian Blurred Laplacian Image", edge_image)
cv2.imshow("Detected Lines Overlaid on Original", overlay_img)
cv2.imshow("Lines", lines)
# cv2.imwrite("hough_lines_overlay.png", overlay_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
