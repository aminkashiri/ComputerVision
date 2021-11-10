import cv2, numpy, random
from matplotlib import pyplot
from os import path
from numpy.random import rand
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def get_matched_keypoints_and_coords(matches, keypoints1, keypoints2):
    matched_keypoints1 =[]
    matched_keypoints2 =[]
    img1_points = numpy.zeros((len(matches), 2), dtype=numpy.float32)
    img2_points = numpy.zeros((len(matches), 2), dtype=numpy.float32)
    for i in range(0,len(matches)):
        img1_points[i] = keypoints1[matches[i].queryIdx].pt
        matched_keypoints1.append(keypoints1[matches[i].queryIdx])
        img2_points[i] = keypoints2[matches[i].trainIdx].pt
        matched_keypoints2.append(keypoints2[matches[i].trainIdx])

    matched_keypoints1 = numpy.array(matched_keypoints1).reshape((-1,1))
    matched_keypoints2 = numpy.array(matched_keypoints2).reshape((-1,1))
    return img1_points, img2_points, matched_keypoints1, matched_keypoints2


def get_matches(descriptor1, descriptor2):
    matches_temp = cv2.BFMatcher(normType=cv2.NORM_L2).knnMatch(descriptor1, descriptor2, k=2)
    matches = []
    for m,n in matches_temp:
        if 1.20 * m.distance < n.distance:
            matches.append(m)
    return matches


def draw_epilines(image1_points, image2_points):
    image1_points = image1_points[:10]
    image1_points = numpy.hstack((image1_points, numpy.ones((10,1))))
    lines = numpy.matmul(F, image1_points.T).T

    image2_with_epipoles = image2.copy()
    height, width = image2.shape[:2]
    for line, point in zip(lines, image2_points):
        point = point.astype(numpy.int32)
        a,b,c = line
        color = tuple((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        x0,y0 = 0, int(-c/b)
        x1,y1 = width, int(-(c+a*width)/b)
        cv2.line(image2_with_epipoles, (x0,y0), (x1,y1), color,3)
        cv2.circle(image2_with_epipoles, (point[0],point[1]), 11, color=(0,0,255), thickness=-1)

    image2_points = image2_points[:10]
    image2_points = numpy.hstack((image2_points, numpy.ones((10,1))))
    lines = numpy.matmul(F.T, image2_points.T).T


    image1_with_epipoles = image1.copy()
    height, width = image1.shape[:2]
    for line, point in zip(lines, image1_points):
        point = point.astype(numpy.int32)
        a,b,c = line
        color = tuple((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        x0,y0 = 0, int(-c/b)
        x1,y1 = width, int(-(c+a*width)/b)
        cv2.line(image1_with_epipoles, (x0,y0), (x1,y1), color,3)
        cv2.circle(image1_with_epipoles, (point[0],point[1]), 11, color=(0,0,255), thickness=-1)

    combined_img = numpy.hstack((image1_with_epipoles, image2_with_epipoles))
    cv2.imwrite(path.join(BASE_DIR,'outputs','images','res08.jpg'), combined_img)
    # pyplot.subplot(121).imshow(image1_with_epipoles)
    # pyplot.subplot(122).imshow(image2_with_epipoles)
    # pyplot.show()


BASE_DIR = path.dirname(__file__)
INPUT_DIR = path.join(BASE_DIR, 'inputs', 'images')
image1 = cv2.imread(path.join(INPUT_DIR,'01.JPG'))
image2 = cv2.imread(path.join(INPUT_DIR,'02.JPG'))

sift = cv2.SIFT_create()
keypoints1, descriptor1 = sift.detectAndCompute(image1, None)
sift = cv2.SIFT_create()
keypoints2, descriptor2 = sift.detectAndCompute(image2, None)

matches = get_matches(descriptor1, descriptor2)
image1_points, image2_points, matched_keypoints1, matched_keypoints2 = get_matched_keypoints_and_coords(matches, keypoints1, keypoints2)


F, mask = cv2.findFundamentalMat(image1_points,image2_points, cv2.FM_RANSAC, 3, 0.99, maxIters=2000)
print('F is: ')
print(F)


temp1 = numpy.zeros(image1.shape)
temp2 = numpy.zeros(image1.shape)

temp1 = cv2.drawKeypoints(image1,matched_keypoints1[mask==1],temp1, color=(0,255,0))
temp1 = cv2.drawKeypoints(temp1,matched_keypoints1[mask==0],temp1, color=(0,0,255))

temp2 = cv2.drawKeypoints(image2,matched_keypoints2[mask==1],temp2, color=(0,255,0))
temp2 = cv2.drawKeypoints(temp2,matched_keypoints2[mask==0],temp2, color=(0,0,255))
combined_img = numpy.hstack((temp1, temp2))
cv2.imwrite(path.join(BASE_DIR,'outputs','images','res05.jpg'), combined_img)


image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, matchesMask=mask, singlePointColor=(0,0,255), matchColor=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite(path.join(BASE_DIR,'outputs','images','res05_2.jpg'), image)


image_copy = image1.copy()
U, S, Vt = numpy.linalg.svd(F, full_matrices=True)
e = Vt[-1,:]
e = e / e[2]
e = e.astype(numpy.int32)
print('e is: ')
print(e[:2])

scale = 10
e = (e/scale).astype(numpy.int32)
x_translate = 5000
y_translate = 1000
translate_matrix = numpy.array([
    [1,0,x_translate],
    [0,1,y_translate],
    [0,0,1]
], dtype=numpy.float32)

image_copy = cv2.resize(image_copy, None, fx=1/scale, fy=1/scale)

image_copy = cv2.warpPerspective(image_copy, translate_matrix, (image_copy.shape[1]+x_translate, image_copy.shape[0]+y_translate))
cv2.circle(image_copy, (e[0]+x_translate, e[1]+y_translate), 15, color=(255,0,0), thickness=-1)
cv2.imwrite(path.join(BASE_DIR,'outputs','images','res06.jpg'), image_copy)


image_copy = image2.copy()
U, S, Vt = numpy.linalg.svd(F.T, full_matrices=True)
ep = Vt[-1,:]
ep = ep / ep[2]
ep = ep.astype(numpy.int32)
print('ep is: ')
print(ep[:2])


scale = 10
ep = (ep/scale).astype(numpy.int32)
x_translate = 0
y_translate = 500
translate_matrix = numpy.array([
    [1,0,x_translate],
    [0,1,y_translate],
    [0,0,1]
], dtype=numpy.float32)

image_copy = cv2.resize(image_copy, None, fx=1/scale, fy=1/scale)

image_copy = cv2.warpPerspective(image_copy, translate_matrix, (image_copy.shape[1]+x_translate+2000, image_copy.shape[0]+y_translate))
cv2.circle(image_copy, (ep[0]+x_translate, ep[1]+y_translate), 15, color=(255,0,0), thickness=-1)
cv2.imwrite(path.join(BASE_DIR,'outputs','images','res07.jpg'), image_copy)

image1_points = image1_points[mask.ravel()==1]
image2_points = image2_points[mask.ravel()==1]
draw_epilines(image1_points, image2_points)

