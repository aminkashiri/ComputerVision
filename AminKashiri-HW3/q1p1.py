from os import path
import cv2, numpy, math
from matplotlib import pyplot

def draw_histogram(lines):
    thetas = lines[:,0,1]
    pyplot.hist(thetas, bins = 300)
    pyplot.show()


def draw_lines_and_vanishing_points(im, lines, Vx, Vy, Vz):
    def get_min_max(Vx,Vy,Vz):
        if not Vy is None and not Vx is None and not Vz is None:
            min_x = Vy[0]
            max_x = Vx[0]

            min_y = Vz[1]
            max_y = max(Vy[1], Vx[1])
        elif not Vy is None and not Vx is None:
            min_x = Vy[0]
            max_x = Vx[0]

            min_y = 0
            max_y = max(Vy[1], Vx[1])
        elif not Vy is None and not Vz is None:
            min_x = Vy[0]
            max_x = im.shape[1]

            min_y = Vz[1]
            max_y = Vy[1]
        elif not Vx is None and not Vz is None:
            min_x = 0
            max_x = Vx[0]

            min_y = Vz[1]
            max_y = Vx[1]
        elif not Vx is None:
            min_x = 0
            max_x = Vx[0]

            min_y = 0
            max_y = Vx[1]
        elif not Vy is None:
            min_x = Vy[0]
            max_x = im.shape[1]

            min_y = 0
            max_y = Vy[1]
        elif not Vz is None:
            min_x = 0
            max_x = im.shape[1]

            min_y = Vz[1]
            max_y = im.shape[0]
        else:
            min_x = 0
            max_x = im.shape[1]

            min_y = 0
            max_y = im.shape[0]

        return min_x, max_x, min_y, max_y

    def resize_everyting(scale):
        im_resized = cv2.resize(im_copy, None, fx=1/scale, fy=1/scale)
        return *list(map(int, [min_x/scale, max_x/scale, min_y/scale, max_y/scale, heigth/scale, width/scale])), im_resized

    print('Drawing VPs')
    length = 40000
    scale = 10
    im_copy = im.copy()


    min_x, max_x, min_y, max_y = get_min_max(Vx,Vy,Vz)
    width, heigth = max_x-min_x, max_y-min_y

    min_x, max_x, min_y, max_y, heigth, width, im_copy = resize_everyting(scale)

    T = numpy.array([
        [1,0,-min_x],
        [0,1,-min_y],
        [0,0,1]
    ], dtype=numpy.float32)

    im_copy = cv2.warpPerspective(im_copy, T, (width,heigth))

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            cos = math.cos(theta)
            sin = math.sin(theta)
            x0 = cos * rho /scale
            y0 = sin * rho /scale
            pt1 = (int(x0 + length*sin)-min_x, int(y0 - length*cos)-min_y)
            pt2 = (int(x0 - length*sin)-min_x, int(y0 + length*cos)-min_y)

            cv2.line(im_copy, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    if not Vx is None:
        V_int = Vx.astype(numpy.int32)
        cv2.circle(im_copy, (V_int[0]-min_x,V_int[1]-min_y), 25, (255,0,0),25)
    if not Vy is None:
        V_int = Vy.astype(numpy.int32)
        cv2.circle(im_copy, (V_int[0]-min_x,V_int[1]-min_y), 25, (255,255,0),25)
    if not Vz is None:
        V_int = Vz.astype(numpy.int32)
        cv2.circle(im_copy, (V_int[0]-min_x,V_int[1]-min_y), 25, (255,255,0),25)

    pyplot.imshow(im_copy)
    pyplot.show()


def draw_lines(im, lines):
    length = 32000
    im_copy = im.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            cos = math.cos(theta)
            sin = math.sin(theta)
            x0 = cos * rho
            y0 = sin * rho
            pt1 = (int(x0 + length*sin), int(y0 - length*cos))
            pt2 = (int(x0 - length*sin), int(y0 + length*cos))

            cv2.line(im_copy, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    pyplot.imshow(im_copy)
    pyplot.show()


def transform_to_cartesian(lines):
    lines_cart = numpy.zeros( (lines.shape[0],2), dtype=lines.dtype)
    for i in range(len(lines)):
        r = lines[i,0,0]
        theta = lines[i,0,1]

        sin = math.sin(theta)
        cos = math.cos(theta)
        # print(r, theta)

        y0 = r/sin
        m = - cos/sin

        lines_cart[i,0] = m
        lines_cart[i,1] = y0
    
    return lines_cart


def get_vanishing_point(lines):
    A = numpy.ones((len(lines),3), dtype=numpy.float32)
    A[:,0] = -1 * lines[:,0]
    A[:,2] = -1 * lines[:,1]

    U, S, Vt = numpy.linalg.svd(A)
    X = Vt[-1,:]
    X = X / X[2]
    X = X.astype(numpy.int32)
    # print(X)

    return X


def draw_horizon(image, Vx, Vy):
    image_copy = image.copy()

    T = numpy.array([
        [1,0,3000],
        [0,1,3000],
        [0,0,1]
    ], dtype=numpy.float32)


    image_copy = cv2.warpPerspective(image_copy, T, (6000+image_copy.shape[1],6000+image_copy.shape[0]))
    Vx_int = Vx.astype(numpy.int32)
    Vy_int = Vy.astype(numpy.int32)
    cv2.line(image_copy, tuple([Vx_int[0]+3000,Vx_int[1]+3000]), tuple([Vy_int[0]+3000,Vy_int[1]+3000]), (255,0,0), 5)

    cv2.imwrite(path.join(BASE_DIR,'outputs','images', 'res01.jpg'), image_copy)
    # pyplot.imshow(im_copy)
    # pyplot.show()


def print_horizon(Vx,Vy):
    line = numpy.cross(Vx,Vy)
    temp = math.sqrt(line[0]**2 + line[1]**2)
    line = line/temp
    print('a, b and c')
    # print(line[0]**2 + line[1]**2)
    print(line)

    pt1 = (0, -line[2]/line[1])
    pt2 = (-line[2]/line[0], 0)
    pt1 = numpy.array(pt1)
    pt2 = numpy.array(pt2)


def draw_structure(im, Vx, Vy, Vz):
    def get_min_max(Vx,Vy,Vz):
        if not Vy is None and not Vx is None and not Vz is None:
            min_x = Vy[0]
            max_x = Vx[0]

            min_y = Vz[1]
            max_y = max(Vy[1], Vx[1])
        elif not Vy is None and not Vx is None:
            min_x = Vy[0]
            max_x = Vx[0]

            min_y = 0
            max_y = max(Vy[1], Vx[1])
        elif not Vy is None and not Vz is None:
            min_x = Vy[0]
            max_x = im.shape[1]

            min_y = Vz[1]
            max_y = Vy[1]
        elif not Vx is None and not Vz is None:
            min_x = 0
            max_x = Vx[0]

            min_y = Vz[1]
            max_y = Vx[1]
        elif not Vx is None:
            min_x = 0
            max_x = Vx[0]

            min_y = 0
            max_y = Vx[1]
        elif not Vy is None:
            min_x = Vy[0]
            max_x = im.shape[1]

            min_y = 0
            max_y = Vy[1]
        elif not Vz is None:
            min_x = 0
            max_x = im.shape[1]

            min_y = Vz[1]
            max_y = im.shape[0]
        else:
            min_x = 0
            max_x = im.shape[1]

            min_y = 0
            max_y = im.shape[0]

        return min_x, max_x, min_y, max_y

    def resize_everyting(scale):
        im_resized = cv2.resize(im_copy, None, fx=1/scale, fy=1/scale)
        return *list(map(int, [min_x/scale, max_x/scale, min_y/scale, max_y/scale, heigth/scale, width/scale])), im_resized, *[x.astype(numpy.int32) for x in [Vx/scale, Vy/scale, Vz/scale] ]

    scale = 10
    padding = 500

    im_copy = im.copy()


    min_x, max_x, min_y, max_y = get_min_max(Vx,Vy,Vz)
    width, heigth = max_x-min_x, max_y-min_y

    min_x, max_x, min_y, max_y, heigth, width, im_copy, Vx, Vy, Vz = resize_everyting(scale)
    im_copy = numpy.ones(im_copy.shape, im.dtype) * 255


    T = numpy.array([
        [1,0,-min_x + padding],
        [0,1,-min_y + padding],
        [0,0,1]
    ], dtype=numpy.float32)


    im_copy = cv2.warpPerspective(im_copy, T, (width+2*padding,heigth+2*padding))


    cv2.line(im_copy, tuple([Vx[0]-min_x+padding,Vx[1]-min_y+padding]), tuple([Vy[0]-min_x+padding,Vy[1]-min_y+padding]), (255,0,0), 3)

    cv2.circle(im_copy, (Vx[0]-min_x+padding,Vx[1]-min_y+padding), 25, (255,0,255),25)
    cv2.circle(im_copy, (Vy[0]-min_x+padding,Vy[1]-min_y+padding), 25, (255,255,0),25)
    cv2.circle(im_copy, (Vz[0]-min_x+padding,Vz[1]-min_y+padding), 25, (0,255,255),25)

    cv2.imwrite(path.join(BASE_DIR,'outputs','images','res02.jpg'), im_copy)
    # pyplot.imshow(im_copy)
    # pyplot.show()



BASE_DIR = path.dirname(__file__)
image = cv2.imread(path.join(BASE_DIR,'inputs','images','vns.jpg'))

borders = cv2.Canny(image, 190, 210, None, 3)

x_lines = cv2.HoughLines(borders, 1, numpy.pi/180, 545, None, 0, 0, min_theta = numpy.pi/2, max_theta = 3*numpy.pi/4) #X
# draw_lines(im, x_lines)
# draw_histogram(x_lines)
y_lines = cv2.HoughLines(borders, 1, numpy.pi/180, 570, None, 0, 0, min_theta = numpy.pi/4, max_theta = numpy.pi/2)#Y
# draw_lines(im, y_lines)
# draw_histogram(y_lines)
z_lines = cv2.HoughLines(borders, 1, numpy.pi/180, 700, None, 0, 0, min_theta = numpy.pi - numpy.pi/10)#Z
# draw_lines(im, z_lines)
# draw_histogram(x_lines)


x_lines_cart = transform_to_cartesian(x_lines)
Vx = get_vanishing_point(x_lines_cart)
y_lines_cart = transform_to_cartesian(y_lines)
Vy = get_vanishing_point(y_lines_cart)
z_lines_cart = transform_to_cartesian(z_lines)
Vz = get_vanishing_point(z_lines_cart)

print('Vx, Vy and Vz')
print(Vx,Vy,Vz)

# lines = numpy.concatenate((x_lines, y_lines, z_lines), axis=0)
# draw_lines_and_vanishing_points(image, lines, Vx, Vy, Vz)

print_horizon(Vx, Vy)

draw_horizon(image, Vx, Vy)

draw_structure(image, Vx, Vy, Vz)



