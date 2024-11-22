import os 
import argparse

import numpy as np
import cv2

import matplotlib.pyplot as plt

from contrast import optimize_contrast_metric

argparser = argparse.ArgumentParser()
argparser.add_argument('--img_dir', type=str, default='.')
argparser.add_argument('--filename', type=str, default='image.jpg')
argparser.add_argument('--img_scale', type=float, default=1.0)
argparser.add_argument('--mode', type=str, default='intensity', 
                       help='intensity | edge | laplace_var')
argparser.add_argument('--optimize', action='store_true', 
                       help='Optimize contrast in a range of gamma values')
args = argparser.parse_args()

img_dir = args.img_dir
fn = args.filename

# get dir name by time stamp YYYY-MM-DD-HH-MM-SS from time
import time
save_dir = time.strftime("%Y%m%d_%H%M%S", time.localtime())
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_contrast_metric(edge, p1, p2,
                        save_graph=False,
                        save_graph_fn='contrast.png',
                        args=None,
                        rgb=None):
    x1, y1 = p1
    x2, y2 = p2
    # print('p1:', p1)
    # print('p2:', p2)
    # print('edge:', edge[y1, x1], edge[y2, x2])

    # get the line
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0:
        dx = 1
    m = dy / dx
    b = y1 - m * x1
    # print('m:', m, 'b:', b)

    # get the contrast metric
    cvalues = []
    for x in range(min(x1, x2), max(x1, x2)):
        y = m * x + b
        y = int(y)
        # print('x:', x, 'y:', y)
        cvalues.append(abs(edge[y, x]))

    if save_graph:
        print('cvalues:', cvalues)
        plt.figure()
        plt.plot(cvalues,'-')
        plt.savefig(
            os.path.join(save_dir,
                         save_graph_fn,))
        
    if args.mode == 'edge':
        print('max contrast:', max(cvalues))
        return max(cvalues)
    elif args.mode == 'intensity':
        cvalues = np.array(cvalues)
        print('cvalues shape:', cvalues.shape)
        print('max contrast:', np.max(cvalues, axis=0))
        return np.max(cvalues, axis=0)
    elif args.mode == 'laplace_var':
        if args.optimize:
            var, gamma = optimize_contrast_metric(
                rgb,
                points,
                args=args)
            return np.array([var]), np.array([gamma])
        else:
            y0 = min(int(points[0][1]),int(points[1][1]))
            y1 = max(int(points[0][1]),int(points[1][1]))
            x0 = min(int(points[0][0]),int(points[1][0]))
            x1 = max(int(points[0][0]),int(points[1][0]))
            cvalues = np.array([edge[y0:y1, x0:x1].var()])
            print('cvalues shape:', cvalues.shape)
            return cvalues

# mouse callback function
points = []
contrasts = []
gammas = []
cnt = -1
def draw_circle(event,x,y,flags,param):
    global points, edge_img, edge, contrasts, cnt, samples 
    global args
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        # print('clicked at', x, y)
        if len(points) < 2:
            points.append((x,y))

        cv2.circle(edge_img,(x,y),3,(255,0,0),-1)
        cv2.circle(samples,(x,y),3,(255,0,0),-1)

        if len(points) == 2:            
            cnt += 1

            if args.mode == 'edge' or args.mode == 'intensity':
                cv2.line(edge_img, points[0], points[1], (0, 255, 0), 2)
                cv2.line(samples, points[0], points[1], (0, 255, 0), 2)
            elif args.mode == 'laplace_var':
                cv2.rectangle(edge_img, points[0], points[1], (0, 255, 0), 2)
                cv2.rectangle(samples, points[0], points[1], (0, 255, 0), 2)

            cv2.putText(edge_img, 
                        str(cnt), 
                        points[0], 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2)
            cv2.putText(samples, 
                        str(cnt), 
                        points[0], 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        2)

            c = get_contrast_metric(edge, 
                                    points[0], 
                                    points[1],
                                    save_graph=True,
                                    save_graph_fn=f'contrast_{cnt}.png',
                                    args=args,
                                    rgb=img)
            if not args.optimize:
                contrasts.append(c)
                print('Contrasts:', contrasts)
            else:
                contrasts.append(c[0])
                gammas.append(c[1])
                print('Contrasts:', contrasts)
                print('Gammas:', gammas)

            # save to file
            with open(os.path.join(save_dir, 'contrast.txt'), 'w') as f:
                for c in contrasts:
                    f.write(str(c) + '\n')

            # save to file
            with open(os.path.join(save_dir, 'gamma.txt'), 'w') as f:
                for g in gammas:
                    f.write(str(g) + '\n')
            points = []

img = cv2.imread(os.path.join(img_dir, fn))
img = cv2.resize(img, (0, 0), fx=args.img_scale, fy=args.img_scale)
cv2.imwrite(os.path.join(save_dir, 'input.png'), 
            img)
samples = img.copy()

if args.mode == 'edge' or args.mode == 'laplace_var':
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Laplacian(blur, cv2.CV_64F)    

    edge_cp = edge.copy()
    edge_img = cv2.cvtColor(edge_cp.astype('uint8'), 
                            cv2.COLOR_GRAY2BGR)
elif args.mode == 'intensity':
    edge = img.copy()

    edge_cp = edge.copy()
    edge_img = edge_cp.copy()
save_dir = os.path.join(args.mode, save_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# draw pixel histogram of edge 
import matplotlib.pyplot as plt

if args.mode == 'intensity' or args.mode == 'laplace_var':
    n, bins, patches = plt.hist(edge.ravel(), 255, [0, 255])
elif args.mode == 'edge':
    n, bins, patches = plt.hist(edge.ravel(), 20, [0, 20])
# for i in range(len(patches)):
#     plt.text(bins[i], n[i], str(int(n[i])), fontsize=8, ha='center')
# plt.show()
plt.savefig(os.path.join(save_dir, 'edge_hist.png'))

cv2.imshow("image", img)

cv2.namedWindow('edge')
cv2.setMouseCallback('edge',draw_circle)
while(1):
    # cv2.imshow("edge", edge)
    cv2.imshow("edge", edge_img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.imwrite(os.path.join(save_dir, 'edge_img.png'), 
            edge_img)
cv2.imwrite(os.path.join(save_dir, 'samples.png'),
            samples)
cv2.destroyAllWindows()