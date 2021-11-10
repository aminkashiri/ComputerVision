import numpy, cv2, random, math
from matplotlib import pyplot

img1 = cv2.imread('inputs/images/im03.jpg')
img2 = cv2.imread('inputs/images/im04.jpg')


sift = cv2.SIFT_create()
keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
sift = cv2.SIFT_create()
keypoints2, descriptor2 = sift.detectAndCompute(img2, None)


def get_matches():
    matches_temp = cv2.BFMatcher(normType=cv2.NORM_L2).knnMatch(descriptor1, descriptor2, k=2)
    matches = []
    for m,n in matches_temp:
        if 1.3 * m.distance < n.distance:
            matches.append(m)
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


def findHomography(img2_points, img1_points, maxIters=10000, threshold=3.0):
    def count_inliers(H, img2_points, img1_points):
        count = 0
        inliers = []

        ones = numpy.ones((number_of_pairs,1))
        img2_points_with_ones = numpy.hstack((img2_points,ones))

        estimated_dst = numpy.matmul(H, img2_points_with_ones.T)

        estimated_dst[0,:] = estimated_dst[0,:] / estimated_dst[2,:]
        estimated_dst[1,:] = estimated_dst[1,:] / estimated_dst[2,:]

        distances = (img1_points[:,0] - estimated_dst[0,:])**2 + (img1_points[:,1] - estimated_dst[1,:])**2

        inlier_indexes = numpy.argwhere(distances < threshold**2).reshape(-1)

        # print('Inliers: ', len(inliers_index), ' out of: ', number_of_pairs)
        return len(inlier_indexes), inlier_indexes

    def findHomographyUtil(indexes):
        rows = []
        for j in indexes:
            x, y = img2_points[j]
            xp, yp = img1_points[j]

            A = numpy.array([
                [ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp],
                [x, y, 1,  0,  0,  0, -x*xp, -y*xp, -xp],
            ])
            rows.append(A)

        A = rows[0]
        for j in range(1,4):
            A = numpy.vstack( (A,rows[j]) )

        U, S, Vt = numpy.linalg.svd(A)
        H = Vt[-1,:]
        H = H.reshape((3,3))

        return H

    def find_homography_with_inliers(inliers_count, inliers_indexes):
        rows = []
        for i in inliers_indexes:
            x, y = img2_points[i]
            xp, yp = img1_points[i]

            A = numpy.array([
                [ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp],
                [x, y, 1,  0,  0,  0, -x*xp, -y*xp, -xp],
            ])
            rows.append(A)

        A = rows[0]
        for i in range(1,inliers_count):
            A = numpy.vstack( (A,rows[i]) )

        U, S, Vt = numpy.linalg.svd(A)
        H = Vt[-1,:]
        H = H.reshape((3,3))
        return H

    number_of_pairs = len(img2_points)
    max_inliers = -1
    best_H = None
    
    w = 0
    p = 0.99
    N = maxIters
    i = 0
    while i < maxIters and i < N:
        if i%1000==0:
            print(i)
        indexes = random.sample(range(number_of_pairs), 4)
        H = findHomographyUtil(indexes)
        
        inliers_count, _ = count_inliers(H, img2_points, img1_points)
        i += 1
        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_H = H
            w = max_inliers/number_of_pairs
            N = math.log(1-p)/math.log(1 - math.pow(w,4))
            print('------------------------------------------> inliers count: ',inliers_count,' w: ',w,' maxIters: ', N)


    inliers_count, inliers_indexes = count_inliers(best_H, img2_points, img1_points)
    H = find_homography_with_inliers(inliers_count, inliers_indexes)

    return H, inliers_indexes

random.seed(0)

matches = get_matches()
print('number of matches: ',len(matches))
img1_points, matched_keypoints1, img2_points, matched_keypoints2 = get_matched_keypoints_and_coords()



homography, mask = findHomography(img2_points, img1_points, maxIters=1000000, threshold=2.0)

homography = homography / homography[2,2]

print(homography)
# print(mask)
# print(mask.shape)

ans = cv2.warpPerspective(img2, homography, img1.shape[:2][::-1])
cv2.imwrite('outputs/images/res20.jpg',ans)

translate_matrix= numpy.array([
    [1, 0, 2900],
    [0, 1, 1500],
    [0, 0,  1 ]
], dtype=numpy.float64)

ans = cv2.warpPerspective(img2, numpy.matmul(translate_matrix,homography), (9000,4000))
cv2.imwrite('outputs/images/res20_2.jpg',ans)