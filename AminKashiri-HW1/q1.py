import cv2, numpy
from matplotlib import pyplot

from skimage.feature import corner_peaks

img1 = cv2.imread('inputs/images/im01.jpg')
img2 = cv2.imread('inputs/images/im02.jpg')

n = 11


def print_info(img):
    print('-------------------------------------------------------')
    print(img.shape)
    print(img.dtype)
    print('Max: ',img.max(),' Min: ',img.min(),' Mean: ',img.mean())
    print('-------------------------------------------------------')

def convert_to_uint8(img):
    img = img - img.min()
    img = (img/img.max())*255
    img = img.astype(numpy.uint8)
    return img

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = numpy.uint8(179*labels/numpy.max(labels))
    blank_ch = 255*numpy.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    pyplot.imshow(labeled_img)
    pyplot.show()

def save_detection_result(img, points, index):
    copy = img.copy()

    #1
    # copy[points[:, 0], points[:, 1]] = (255,255,255)

    #2
    for point in points:
        cv2.circle(copy, tuple(point[::-1]), 3,(255,0,0), thickness=-1)
    cv2.imwrite(f'outputs/images/res0{index+7}_harris.jpg', copy)
    # pyplot.imshow(copy)
    # pyplot.show()

    #3
    # pyplot.imshow(copy)
    # pyplot.plot(points[:, 1], points[:, 0], 'r.', markersize=2)
    # pyplot.savefig(f'outputs/images/res0{index+7}_harris.jpg')
    # pyplot.show()

def non_maximum_supression(binary_mask):
    labels_count, labels = cv2.connectedComponents(binary_mask, connectivity=4)
    points = numpy.zeros( (labels_count,2), dtype=numpy.uint16)
    for i in range(labels_count):
        ith_component = (labels==i)*R
        points[i] = numpy.argwhere( ith_component==ith_component.max() )[0]

    return points

def remove_border_points(points, img_width, img_height, d):
    new_points = numpy.zeros(points.shape, dtype=numpy.uint16)
    index = 0 
    for point in points:
        start_x, finish_x, start_y, finish_y = point[0]-feature_radius,point[0]+feature_radius+1,point[1]-feature_radius,point[1]+feature_radius+1
        if start_x>0 and start_y>0 and finish_x<img_height and finish_y<img_width:
            new_points[index] = point
            index+=1
    
    new_points = new_points[:index,:]
    return new_points

detected_points = [None]*2
feature_vectors = [None]*2

for index, img in enumerate([img1,img2]):
    Ix = numpy.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    Iy = numpy.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy

    grad_magnitude = numpy.sqrt(Ix2+Iy2)

    cv2.imwrite(f'outputs/images/res0{index+1}_grad.jpg', convert_to_uint8(grad_magnitude))

    kernel_size = (11,11)
    Sx2 = cv2.GaussianBlur(Ix2, kernel_size, 0)
    Sy2 = cv2.GaussianBlur(Iy2, kernel_size, 0)
    Sxy = cv2.GaussianBlur(Ixy, kernel_size, 0)

    
    det = Sx2 * Sy2 - Sxy**2
    trace = Sx2 + Sy2

    k = 0.05
    R = det - k * trace**2
    R = numpy.sum(R, axis=2)

    cv2.imwrite(f'outputs/images/res0{index+3}_score.jpg', convert_to_uint8(R))

    threshold = 11333444

    mask = R < threshold

    binary_mask = numpy.where(mask, 0, 255).astype(numpy.uint8)

    cv2.imwrite(f'outputs/images/res0{index+5}_thresh.jpg', binary_mask)


    points = non_maximum_supression(binary_mask) 

    feature_radius = int(n/2)
    points = remove_border_points(points, *img.shape[:2], feature_radius)

    save_detection_result(img, points, index)

    detected_size = len(points)
    image_feature_vectors = numpy.zeros( (detected_size,3*n**2) )
    for i, point in enumerate(points):
        start_x, finish_x, start_y, finish_y = points[i,0]-feature_radius,points[i,0]+feature_radius+1,points[i,1]-feature_radius,points[i,1]+feature_radius+1
        # print(point)
        # print(start_x,start_y,finish_x,finish_y)
        image_feature_vectors[i] = img[start_x:finish_x,start_y:finish_y,:].reshape( (1,-1) ) 

    detected_points[index] = points
    feature_vectors[index] = image_feature_vectors

threshold = 1.8

matches = numpy.zeros( (len(detected_points[0]),len(detected_points[1]),2) )

for i in range(2):
    for q, point in enumerate(detected_points[i]):
        feature_vector = feature_vectors[i][q]
        distances = numpy.sum( (feature_vectors[(i+1)%2] - feature_vector)**2, axis=1)

        d1, d2 = numpy.sort(numpy.partition(distances,2)[:2])

        if d2 < threshold*d1:
            continue 

        p1_index = numpy.argwhere(distances==d1)[0]

        a, b = (q, p1_index) if i == 0 else (p1_index, q)
        
        matches[a, b, i] = 1


matches = matches[:,:,0] * matches[:,:,1]

matches_indexes = numpy.argwhere(matches==1)
print('number of matches: ', len(matches_indexes))

copy1 = img1.copy()
copy2 = img2.copy()
for v1, v2 in matches_indexes:
    cv2.circle(copy1, tuple(detected_points[0][v1][::-1]), 5,(255,0,0), thickness=-1)
    cv2.circle(copy2, tuple(detected_points[1][v2][::-1]), 5,(255,0,0), thickness=-1)

cv2.imwrite('outputs/images/res09_corres.jpg', copy1)
cv2.imwrite('outputs/images/res10_corres.jpg', copy2)


combined_img = numpy.hstack( (img1,img2) )

for i, match in enumerate(matches_indexes):
    v1, v2 = match
    color = ((i*20)%255,(150+i*25)%255,(200+i*55)%255)
    p1 = tuple(detected_points[0][v1][::-1])
    p2 = tuple(detected_points[1][v2][::-1]+(img1.shape[1],0))
    cv2.circle(combined_img, p1, 5,color, thickness=-1)
    cv2.circle(combined_img, p2, 5,color, thickness=-1)
    cv2.line(combined_img, p1, p2, color, thickness= 3)


cv2.imwrite('outputs/images/res11.jpg', combined_img)






