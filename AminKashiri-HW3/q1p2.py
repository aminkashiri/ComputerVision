from os import path
import cv2, numpy, math
from matplotlib import pyplot


def get_camera_parameters(Vx, Vy, Vz):
    # Compute px, py
    a1, b1 = Vx[:2]
    a2, b2 = Vy[:2]
    a3, b3 = Vz[:2]

    print('Vx, Vy and Vz:')
    print(Vx,Vy,Vz) 
    A = numpy.array([
        [a1-a3, b1-b3],
        [a2-a3, b2-b3]
    ])

    b = numpy.array([
        a2*(a1-a3) + b2*(b1-b3),
        a1*(a2-a3) + b1*(b2-b3),
    ])

    Px, Py = list(map(int, list(numpy.linalg.solve(A, b)) ))
    print('Px = ', Px)
    print('Py = ', Py)
    f = int(math.sqrt(-Px**2-Py**2+(a1+a2)*Px+(b1+b2)*Py-(a1*a2+b1*b2)))
    print('f = ', f)

    return f, Px, Py


def draw_principal_point(image, f, Px, Py):
    image_copy = image.copy()
    cv2.circle(image_copy, (Px, Py), 25, (255,0,0), -1)
    pyplot.imshow(image_copy)
    pyplot.title('f = ' + str(f) )
    pyplot.savefig(path.join(BASE_DIR, 'outputs', 'images', 'res03.jpg'))


Vx = numpy.array([
    9364,
    2596,
    1
])
Vy = numpy.array([
    -26748,
    4061,
    1
])
Vz = numpy.array([
    -2996,
    -133753,
    1
])

BASE_DIR = path.dirname(__file__)
image = cv2.imread(path.join(BASE_DIR,'inputs','images','vns.jpg'))

f, Px, Py = get_camera_parameters(Vx, Vy, Vz)

draw_principal_point(image, f, Px, Py)

K = numpy.array([
    [f, 0, Px],
    [0, f, Py],
    [0,0,1]
])

Vz = numpy.matmul(numpy.linalg.inv(K), Vz)
Vy = numpy.matmul(numpy.linalg.inv(K), Vy)
Vx = numpy.matmul(numpy.linalg.inv(K), Vx)

Cam_Z = numpy.array([0,0,1])
Cam_Y = numpy.array([0,1,0])
Cam_X = numpy.array([1,0,0])


Vz_on_xy_plane = Vz.copy()
Vz_on_xy_plane[2] = 0


cos_phi = numpy.dot(Vz_on_xy_plane,Cam_X)/math.sqrt(numpy.dot(Vz_on_xy_plane,Vz_on_xy_plane))
theta_z = numpy.pi/2 - math.acos(cos_phi)
print('theta_z in degrees: ', theta_z/math.pi * 180)
# print(phi)
cos_theta = numpy.dot(Cam_Z,Vz)/math.sqrt(numpy.dot(Vz,Vz))
theta_x = math.acos(cos_theta) - numpy.pi/2
print('theta_x in degrees: ', theta_x/math.pi * 180)
# print(theta)
