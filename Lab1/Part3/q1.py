import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. LOAD IMAGES
# ============================================================

imgr1 = cv2.imread(
    r'S:\Computer Vision\Lab1\Part3\images\imgr1.png'
)

imgr2 = cv2.imread(
    r'S:\Computer Vision\Lab1\Part3\images\imgr2.png'
)

if imgr1 is None or imgr2 is None:
    raise ValueError("Could not load one or both images.")


# ============================================================
# 2. CONVERT TO GRAYSCALE
# ============================================================

gray1 = cv2.cvtColor(imgr1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(imgr2, cv2.COLOR_BGR2GRAY)


# ============================================================
# 3. CREATE ORB DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=1000
)


# ============================================================
# 4. DETECT ORB KEYPOINTS AND DESCRIPTORS
# ============================================================

keypoints1, descriptors1 = orb.detectAndCompute(
    gray1,
    None
)

keypoints2, descriptors2 = orb.detectAndCompute(
    gray2,
    None
)


print("Keypoints in Image 1:", len(keypoints1))
print("Keypoints in Image 2:", len(keypoints2))


# ============================================================
# 5. CREATE BF MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)


# ============================================================
# 6. MATCH ORB DESCRIPTORS
# ============================================================

matches = bf.match(
    descriptors1,
    descriptors2
)


# ============================================================
# 7. SORT MATCHES BY DISTANCE
# ============================================================

matches = sorted(
    matches,
    key=lambda x: x.distance
)


# ============================================================
# 8. KEEP BEST MATCHES
# ============================================================

good_matches = matches[:50]

print("Good matches:", len(good_matches))


# ============================================================
# 9. DRAW MATCHES
# ============================================================

matched_image = cv2.drawMatches(
    imgr1,
    keypoints1,
    imgr2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


# ============================================================
# 10. DISPLAY
# ============================================================

plt.figure(figsize=(18, 8))

plt.imshow(
    cv2.cvtColor(
        matched_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("ORB Feature Matching")
plt.axis("off")

plt.show()