import cv2, numpy, os, numpy
from datetime import datetime, timedelta
from matplotlib import pyplot

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.0005)

indexes = [
    (1,11),
    (6,16),
    (11,21),
    (1,21),
    ]

real_coord = numpy.mgrid[0:9,0:6].astype(numpy.float32)
zeros = numpy.zeros((9,6), numpy.float32)
real_coord = numpy.vstack((real_coord, zeros[None,:,:]))
real_coord = real_coord.T.reshape(-1,3) * 22

height, width = cv2.imread(f'inputs/images/im01.jpg').shape[:2]
print('img_shape/2 is: ', height/2, width/2)

for index in indexes:
    img_coords = []
    real_coords = []
    for i in range(*index):
        real_coords.append(real_coord)
    for i in range(*index):
        # print(f'image{i}')
        img = cv2.imread(f'inputs/images/im{i:02}.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found_pattern, corners = cv2.findChessboardCorners(img, (9,6), None)
        if not found_pattern:
            real_coords.pop()
            continue

        corners = cv2.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        img_coords.append(corners)

        # img = cv2.drawChessboardCorners(img, (9,6), corners, found_pattern)
        # pyplot.imshow(img)
        # pyplot.show()

    _, camera_matrix, _, _, _ = cv2.calibrateCamera(real_coords, img_coords,(width, height), None, None)
    # camera_matrix = numpy.array([
    #     [1,0,width/2],
    #     [0,1,height/2],
    #     [0,0,1]
    # ])
    # _, camera_matrix, _, _, _ = cv2.calibrateCamera(real_coords, img_coords,(width, height), camera_matrix, None, flags=cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_PRINCIPAL_POINT)

    print(f'For {index}:')
    print(camera_matrix)