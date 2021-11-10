import cv2, numpy, math
from matplotlib import pyplot
from math import sin, cos


logo = cv2.imread('inputs/images/logo.png')

fx = fy = 500
s = 0
final_size = 1000
Px = Py = final_size
K = numpy.array([
    [fx, s , Px],
    [ 0, fy, Py],
    [ 0, 0 , 1 ]
])

Px = Py = 128
Kp = numpy.array([
    [fx, s , Px],
    [ 0, fy, Py],
    [ 0, 0 , 1 ]
])

horizontal_distance_of_camera_from_field_center = 40
# horizontal_distance_of_camera_from_field_center = math.sqrt(40**2 - 25**2)
C = numpy.array([0, horizontal_distance_of_camera_from_field_center, 0]).reshape((3,1))
n = numpy.array([0,0,1]).reshape((3,1))
d = -25


th = -1*math.atan(horizontal_distance_of_camera_from_field_center/25)

R = numpy.array([
    [1   ,  0  , 0],
    [0      ,  cos(th)  ,-1*sin(th)   ],
    [0,  sin(th)  , cos(th)]
])

t = (-1 * numpy.matmul(R,C))

temp = R +  -1 * numpy.matmul(t,n.T)/d

temp = numpy.matmul(Kp, temp)
H = numpy.matmul(temp, numpy.linalg.inv(K))
H_inv = numpy.linalg.inv(H)

res = cv2.warpPerspective(logo, H_inv, (2*final_size,2*final_size))

cv2.imwrite('outputs/images/res12.jpg', res)
# pyplot.imshow(res)

