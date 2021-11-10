from os import path
import cv2, numpy, math
from matplotlib import pyplot
from numpy.lib.type_check import imag


def create_rotation_matrix(x,y,z):
        Rz = numpy.array([
            [math.cos(z), -math.sin(z),0],
            [math.sin(z), math.cos(z), 0],
            [   0  ,       0     ,   1  ]
        ])
        Ry = numpy.array([
            [math.cos(y),  0, math.sin(y)],
            [   0  ,       1     ,    0  ],
            [-math.sin(y), 0, math.cos(y)]
        ])
        Rx = numpy.array([
            [   1    ,       0     ,  0   ],
            [0, math.cos(x),  -math.sin(x)],
            [0, math.sin(x),   math.cos(x)]
        ])
        # print(Rz)
        # print(Ry)
        # print(Rx)
        # R = numpy.matmul(numpy.matmul(Rx,Ry),Rz)
        R = numpy.matmul(numpy.matmul(Rz,Ry),Rx)
        R = R/R[2,2]
        return R


BASE_DIR = path.dirname(__file__)
image = cv2.imread(path.join(BASE_DIR,'inputs','images','vns.jpg'))

theta_z = -0.04054360573204363
theta_x = -0.10361639340234485

Rz = create_rotation_matrix(0,0,-theta_z)
Rx = create_rotation_matrix(-theta_x,0,0)
R = numpy.matmul(Rx,Rz)

f, Px, Py = 14067, 2487, 1410
K = numpy.array([
    [f, 0, Px],
    [0, f, Py],
    [0,0,1]
])
H = numpy.matmul(K, numpy.matmul(R, numpy.linalg.inv(K)))
print(H)

T = numpy.array([
    [1,0,100],
    [0,1,1700],
    [0,0,1],
])
H = numpy.matmul(T, H)
image = cv2.warpPerspective(image, H, (image.shape[1]+300, image.shape[0]+300))

cv2.imwrite(path.join(BASE_DIR, 'outputs','images','res04.jpg'), image)

