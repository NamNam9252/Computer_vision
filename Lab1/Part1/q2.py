import cv2

img256 = cv2.imread("output/img1(256).png")
img128 = cv2.imread("output/img1(128).png")

# nearset 
# linear 
# cubic

imgs = [img128 , img256]

methods =[cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_CUBIC]
size = (512,512)

for y in range(0,2):
    for x in range(0,3):
        cv2.imwrite(f"output/img{y}_{x}.png",cv2.resize(imgs[y],size,interpolation=methods[x]))
    