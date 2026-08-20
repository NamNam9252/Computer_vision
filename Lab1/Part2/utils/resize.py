import numpy as np

def nearestNeighbour( img : np.array  , sizeh : int  ,sizew : int  ):
    h,w,c = img.shape
    result  = np.zeros((sizeh , sizew ,c) ,dtype = np.uint8)
    for i in range(sizeh):
        for j in range(sizew):
            oldi = int(i*h/sizeh)
            oldj = int (j*w/sizew)
            result[i][j] = img[oldi][oldj]
    return result

def bilinearInterpolation(img, sizeh, sizew):
    h, w, c = img.shape
    result = np.zeros((sizeh, sizew, c), dtype=np.uint8)
    for i in range(sizeh):
        for j in range(sizew):
            # Find corresponding position in original image
            x = i * (h - 1) / (sizeh - 1)
            y = j * (w - 1) / (sizew - 1)
            # Get surrounding pixel coordinates
            x1 = int(x)
            y1 = int(y)
            x2 = min(x1 + 1, h - 1)
            y2 = min(y1 + 1, w - 1)
            # Fractional part
            dx = x - x1
            dy = y - y1
            # Step 1: Interpolate horizontally
            top = (1 - dy) * img[x1, y1] + dy * img[x1, y2]
            bottom = (1 - dy) * img[x2, y1] + dy * img[x2, y2]
            # Step 2: Interpolate vertically
            value = (1 - dx) * top + dx * bottom
            result[i, j] = np.clip(value, 0, 255)
    return result
    
def bicubicInterpolation(img, sizeh, sizew):

    h, w, c = img.shape
    result = np.zeros((sizeh, sizew, c), dtype=np.uint8)

    # Cubic interpolation function
    def cubic(x):
        x = abs(x)

        if x <= 1:
            return 1 - 2*x*x + x*x*x

        elif x < 2:
            return 4 - 8*x + 5*x*x - x*x*x

        return 0

    for i in range(sizeh):
        for j in range(sizew):

            # 1. Find position in original image
            x = i * h / sizeh
            y = j * w / sizew

            x0 = int(x)
            y0 = int(y)

            # 2. Start final pixel with zero
            pixel = np.zeros(c, dtype=np.float32)

            # 3. Take 4 × 4 = 16 neighbouring pixels
            for m in range(-1, 3):
                for n in range(-1, 3):

                    # Get neighbour
                    x_neighbour = min(max(x0 + m, 0), h - 1)
                    y_neighbour = min(max(y0 + n, 0), w - 1)

                    # Calculate distance
                    dx = x - (x0 + m)
                    dy = y - (y0 + n)

                    # Calculate weight
                    weight_x = cubic(dx)
                    weight_y = cubic(dy)

                    weight = weight_x * weight_y

                    # Add contribution
                    pixel += img[x_neighbour, y_neighbour] * weight

            # 4. Store final pixel
            result[i, j] = np.clip(pixel, 0, 255)

    return result