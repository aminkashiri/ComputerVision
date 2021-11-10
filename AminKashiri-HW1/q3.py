import numpy, cv2
from matplotlib import pyplot

img1 = cv2.imread('inputs/images/im03.jpg')
img2 = cv2.imread('inputs/images/im04.jpg')


sift = cv2.SIFT_create()
keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
sift = cv2.SIFT_create()
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

def draw_keypoints():
    temp1 = numpy.zeros(img1.shape, dtype=numpy.uint8)
    temp2 = numpy.zeros( (img1.shape[0],img2.shape[1],3), dtype=numpy.uint8)

    temp1 = cv2.drawKeypoints(img1,keypoints1,temp1, color=(0,255,0))
    temp2 = cv2.drawKeypoints(img2,keypoints2,temp2, color=(0,255,0))

    temp = numpy.zeros( (img1.shape[0],img2.shape[1],3), dtype=numpy.uint8)
    temp[0:temp2.shape[0],:] = temp2

    combined_img = numpy.hstack((temp1, temp))
    cv2.imwrite('outputs/images/res13_corners.jpg', combined_img)


def get_matches():
    matches_temp = cv2.BFMatcher(normType=cv2.NORM_L2).knnMatch(descriptor1, descriptor2, k=2)
    matches = []
    for m,n in matches_temp:
        if 1.20 * m.distance < n.distance:
            matches.append(m)
    return matches


def get_matches2():
    matches = cv2.BFMatcher_create(normType=cv2.NORM_L2, crossCheck=True).match(descriptor1, descriptor2)
    return matches


def get_matched_keypoints_and_coords():
    img1_points = numpy.zeros((len(matches), 2), dtype=numpy.float32)
    img2_points = numpy.zeros((len(matches), 2), dtype=numpy.float32)
    matched_keypoints1 = []
    matched_keypoints2 = []
    for i in range(0,len(matches)):
        img1_points[i] = keypoints1[matches[i].queryIdx].pt
        matched_keypoints1.append(keypoints1[matches[i].queryIdx])
        img2_points[i] = keypoints2[matches[i].trainIdx].pt
        matched_keypoints2.append(keypoints2[matches[i].trainIdx])

    return img1_points, matched_keypoints1, img2_points, matched_keypoints2


def draw_correspondence():
    temp1 = numpy.zeros(img1.shape)
    temp2 = numpy.zeros( (img1.shape[0],img2.shape[1],3) )


    temp1 = cv2.drawKeypoints(img1,keypoints1,temp1, color=(0,255,0))
    temp1 = cv2.drawKeypoints(temp1,matched_keypoints1,temp1, color=(255,0,0))
    temp2 = cv2.drawKeypoints(img2,keypoints2,temp2, color=(0,255,0))
    temp2 = cv2.drawKeypoints(temp2,matched_keypoints2,temp2, color=(255,0,0))
    temp = numpy.zeros( (img1.shape[0],img2.shape[1],3) )
    temp[0:temp2.shape[0],:] = temp2

    combined_img = numpy.hstack((temp1, temp))
    cv2.imwrite('outputs/images/res14_correspondences.jpg', combined_img)


def draw_matched_keypoints():
    img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, singlePointColor=(0,255,0), matchColor=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imwrite('outputs/images/res15_matches.jpg', img)


def draw_matched_keypoints_and_inliers(mask):
    img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, singlePointColor=(0,255,0), matchColor=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    temp1 = img[:,:img1.shape[1]]
    temp2 = img[:,img1.shape[1]:]

    img = cv2.drawMatches(temp1, keypoints1, temp2, keypoints2, matches, None, matchesMask=mask, singlePointColor=(0,255,0), matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imwrite('outputs/images/res17.jpg', img)

    img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, matchesMask=mask, singlePointColor=(0,255,0), matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imwrite('outputs/images/res_INLIERS.jpg', img)


def draw_matched_keypoints_sample():
    matches_sample = sorted(matches, key = lambda x:x.distance)[:20]
    img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches_sample, None, matchColor=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('outputs/images/res16_bests.jpg', img)

    img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, matchColor=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('outputs/images/res16.jpg', img)


def draw_inliers():
    for i in range(len(matches)):
        warped_point = 4


draw_keypoints()

matches = get_matches()
print('Number of matches: ',len(matches))
img1_points, matched_keypoints1, img2_points, matched_keypoints2 = get_matched_keypoints_and_coords()

draw_correspondence()

draw_matched_keypoints()

draw_matched_keypoints_sample()

homography, mask = cv2.findHomography(img2_points, img1_points, cv2.RANSAC, maxIters=800, ransacReprojThreshold=2)

# print(mask)
# print(mask.shape)
print('Homography is:')
print(homography)
print('Number of inliers: ',len(numpy.argwhere(mask==1)))


draw_matched_keypoints_and_inliers(mask)

draw_inliers()


ans = cv2.warpPerspective(img2, homography, img1.shape[:2][::-1])
cv2.imwrite('outputs/images/res19.jpg',ans)

translate_matrix= numpy.array([
    [1, 0, 2900],
    [0, 1, 1500],
    [0, 0,  1 ]
], dtype=numpy.float64)

ans = cv2.warpPerspective(img2, numpy.matmul(translate_matrix,homography), (8100,4000))
cv2.imwrite('outputs/images/res19_2.jpg',ans)
# pyplot.imshow(ans)
# pyplot.show()