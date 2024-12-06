import os
import sys

import cv2
import numpy as np
import skimage as sk

def get_roi(img, points):
    assert len(points) == 2
    y0 = min(int(points[0][1]),int(points[1][1]))
    y1 = max(int(points[0][1]),int(points[1][1]))
    x0 = min(int(points[0][0]),int(points[1][0]))
    x1 = max(int(points[0][0]),int(points[1][0]))
    return img[y0:y1, x0:x1]

def optimize_contrast_metric(rgb, points,
                        args=None,
                        gammas=[0.1, 0.25, 0.33, 0.5, 1, 1.25, 1.5, 2, 2.5, 3],
                        ):
    
    if args.mode != 'laplace_var':
        return -1 # invalid
    
    if len(points) != 2:
        return -1
    
    roi = get_roi(rgb, points)
    roi0 = roi.copy()
    vars = []
    for gm in gammas:
        roi1 = sk.exposure.adjust_gamma(roi0, gamma=gm)

        gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Laplacian(blur, cv2.CV_64F)    

        var = edge.var()
        vars.append(var)
    return max(vars), gammas[np.argmax(vars)]

# if __name__ == '__main__':
#     pass