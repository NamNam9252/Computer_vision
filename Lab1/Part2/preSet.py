import cv2
import numpy as np 



img1 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img1.png")
snp = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\saltandpeper.png")


img2 = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\img2.png")
gau = cv2.imread(r"S:\Computer Vision\Lab1\Part2\images\gaussian.png")


img1 = np.array(img1)
snp=np.array(snp)

img2 = np.array(img2)
gau=np.array(gau)

img3 = (img1 + (0.3*snp))
img4 = (img2 + (0.2*gau))

cv2.imwrite(r"S:\Computer Vision\Lab1\Part2\images\img3.png" , img3)
cv2.imwrite(r"S:\Computer Vision\Lab1\Part2\images\img4.png" , img4)



