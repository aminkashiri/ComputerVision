import cv2, numpy, os, math
from scipy.spatial.transform import Rotation
from datetime import datetime, timedelta
from matplotlib import pyplot

def perspectiveTransform(points, H):
    temp = cv2.perspectiveTransform(points.astype(numpy.float32)[None,:,:], H).astype(numpy.int32)
    return temp[0]


def compute_background_resolution_and_translate_matrix():
    homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)

    all_corner_coords = numpy.zeros((4,2,900), dtype=numpy.int32)
    for i in range(0,900):
        hmg = homographies[i+1].reshape(3,3)

        corner_coords_transformed = perspectiveTransform(CORNER_COORDS, hmg)
        all_corner_coords[:,:,i] = corner_coords_transformed

    x_min = all_corner_coords[:,0,:].min()
    x_max = all_corner_coords[:,0,:].max()

    y_min = all_corner_coords[:,1,:].min()
    y_max = all_corner_coords[:,1,:].max()

    width = x_max - x_min + 1
    heigth = y_max - y_min + 1

    #NOTE: Resolution must be even*even
    width = width + width%2
    heigth = heigth + heigth%2

    background_resolution = (heigth, width, 3)

    translate_matrix= numpy.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0,   1  ]
    ], dtype=numpy.float64)

    return background_resolution, translate_matrix


def print_info(name, img):
    print(name," info: ")
    print("     dtype: ",img.dtype)
    print("     shape: ",img.shape)
    print("     max: ",img.max())
    print("     min: ",img.min())


def get_homography(f2, f1):
    def get_matched_keypoints_and_coords(matches, keypoints1, keypoints2):
        img1_points = numpy.zeros((len(matches), 2), dtype=numpy.float32)
        img2_points = numpy.zeros((len(matches), 2), dtype=numpy.float32)
        for i in range(0,len(matches)):
            img1_points[i] = keypoints1[matches[i].queryIdx].pt
            img2_points[i] = keypoints2[matches[i].trainIdx].pt

        return img1_points, img2_points

    def get_matches(descriptor1, descriptor2):
        matches_temp = cv2.BFMatcher(normType=cv2.NORM_L2).knnMatch(descriptor1, descriptor2, k=2)
        matches = []
        for m,n in matches_temp:
            if 1.20 * m.distance < n.distance:
                matches.append(m)
        return matches
    
    sift = cv2.SIFT_create()
    keypointsf2, descriptorf2 = sift.detectAndCompute(f2, None)
    sift = cv2.SIFT_create()
    keypointsf1, descriptorf1 = sift.detectAndCompute(f1, None)

    matches = get_matches(descriptorf1, descriptorf2)
    f1_points, f2_points = get_matched_keypoints_and_coords(matches, keypointsf1, keypointsf2)

    Hf2tof1, ـ = cv2.findHomography(f2_points, f1_points, cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000)

    return Hf2tof1


def compute_ref_homographies():
    print('Computing reference homographies')
    f90 = cv2.imread(f'inputs/videos/frames/frame90{QUALITY}.png')
    f270 = cv2.imread(f'inputs/videos/frames/frame270{QUALITY}.png')
    f450 = cv2.imread(f'inputs/videos/frames/frame450{QUALITY}.png')
    f630 = cv2.imread(f'inputs/videos/frames/frame630{QUALITY}.png')
    f810 = cv2.imread(f'inputs/videos/frames/frame810{QUALITY}.png')

    ref_homographies['450'] = numpy.eye(3,3)

    f270_hmg = get_homography(f270, f450)
    ref_homographies['270'] = f270_hmg

    f90_hmg = get_homography(f90, f270)
    f90_hmg = numpy.matmul(f270_hmg, f90_hmg)
    ref_homographies['90'] = f90_hmg

    f630_hmg = get_homography(f630, f450)
    ref_homographies['630'] = f630_hmg

    f810_hmg = get_homography(f810, f630)
    f810_hmg = numpy.matmul(f630_hmg, f810_hmg)
    ref_homographies['810'] = f810_hmg

    print('Computing reference homographies finished')


def extract_frames():
    if not os.path.isfile(f'inputs/videos/video{QUALITY}.mp4'):
        print('Decreasing video resolution')
        a = int(1080 * float(QUALITY[1:]))
        os.system(f'ffmpeg -i inputs/videos/video.mp4 -filter:v scale=-1:{a} -c:a copy inputs/videos/video{QUALITY}.mp4')

    print('Extracting video frames')
    os.system(f'ffmpeg -t 00:00:30 -i inputs/videos/video{QUALITY}.mp4 -vf fps=30 inputs/videos/frames/frame%d{QUALITY}.png')
    print(f'Frames of video with quality={QUALITY} extracted')


def convert_to_uint8(img):
    img = img - img.min()
    img = (img/img.max())*255
    img = img.astype(numpy.uint8)
    return img


def get_mask(H):
    back_panel = numpy.zeros( BACKGROUND_RESOLUTION[:2], dtype=numpy.uint8)
    corner_coords_transformed = perspectiveTransform(CORNER_COORDS, H)
    
    mask = cv2.fillPoly(back_panel, [corner_coords_transformed], 1)
    return mask


def compute_homographies():
    print('Computing all homographies')

    compute_ref_homographies()

    f90 = cv2.imread(f'inputs/videos/frames/frame90{QUALITY}.png')
    f270 = cv2.imread(f'inputs/videos/frames/frame270{QUALITY}.png')
    f450 = cv2.imread(f'inputs/videos/frames/frame450{QUALITY}.png')
    f630 = cv2.imread(f'inputs/videos/frames/frame630{QUALITY}.png')
    f810 = cv2.imread(f'inputs/videos/frames/frame810{QUALITY}.png')

    homographies = numpy.zeros((901,9), dtype=numpy.float32)
    # homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)

    ref_homographie = ref_homographies['90']
    for i in range(1,90):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = get_homography(frame, f90)
        hmg = numpy.matmul(ref_homographie, hmg)
        homographies[i] = hmg.reshape(1,9)
    homographies[90] = ref_homographie.reshape(1,9)

    ref_homographie = ref_homographies['270']
    for i in range(91,270):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = get_homography(frame, f270)
        hmg = numpy.matmul(ref_homographie, hmg)
        homographies[i] = hmg.reshape(1,9)
    homographies[270] = ref_homographie.reshape(1,9)

    ref_homographie = ref_homographies['450']
    for i in range(271,450):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = get_homography(frame, f450)
        hmg = numpy.matmul(ref_homographie, hmg)
        homographies[i] = hmg.reshape(1,9)
    homographies[450] = ref_homographie.reshape(1,9)
    for i in range(451,630):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = get_homography(frame, f450)
        hmg = numpy.matmul(ref_homographie, hmg)
        homographies[i] = hmg.reshape(1,9)

    ref_homographie = ref_homographies['630']
    for i in range(631,810):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = get_homography(frame, f630)
        hmg = numpy.matmul(ref_homographie, hmg)
        homographies[i] = hmg.reshape(1,9)
    homographies[630] = ref_homographie.reshape(1,9)

    ref_homographie = ref_homographies['810']
    for i in range(811,901):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = get_homography(frame, f810)
        hmg = numpy.matmul(ref_homographie, hmg)
        homographies[i] = hmg.reshape(1,9)
    homographies[810] = ref_homographie.reshape(1,9)

    # print_info('homographies',homographies)

    cv2.imwrite(f'outputs/temps/homographies{QUALITY}.tiff',homographies)
    print('Computing and saving all homographies Finished')


# Works only with full resolution frames
def part1():
    print('Starting part1')

    rectangle_coords = numpy.array([
        [300, 400],
        [1000, 400],
        [1000, 1000],
        [300, 1000]
    ], dtype=numpy.int32)

    f450 = cv2.imread('inputs/videos/frames/frame450.png')
    f270 = cv2.imread('inputs/videos/frames/frame270.png')

    H =get_homography(f270, f450)
    H_inv = numpy.linalg.inv(H)

    res01_450_rect = cv2.polylines(f450.copy(), [rectangle_coords], True, (0,0,255), 3)

    rectangle_coords_transformed = perspectiveTransform(rectangle_coords, H_inv)

    res02_270_rect = cv2.polylines(f270.copy(), [rectangle_coords_transformed], True, (0,0,255), 3)

    cv2.imwrite('outputs/images/res01-450-rect.jpg',res01_450_rect)
    cv2.imwrite('outputs/images/res02-270-rect.jpg',res02_270_rect)

    back_panel = cv2.warpPerspective(f270, numpy.matmul(TRANSLATE_MATRIX, H), (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) )

    start_x, start_y = int(TRANSLATE_MATRIX[1,2]), int(TRANSLATE_MATRIX[0,2])
    back_panel[start_x:start_x + f450.shape[0], start_y:start_y + f450.shape[1],:] = f450

    cv2.imwrite(f'outputs/images/res03-270-450-panorama.jpg', back_panel)


def part2():
    def non_maximum_supression(binary_mask):
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_DILATE, numpy.ones((9,9)))
        # pyplot.subplot(311).imshow(binary_mask)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_ERODE, numpy.ones((7,7)))
        # pyplot.subplot(312).imshow(binary_mask)
        labels_count, labels = cv2.connectedComponents(binary_mask, connectivity=8)
        # pyplot.subplot(313).imshow(labels)
        # pyplot.show()

        if labels_count > 3:
            print(labels_count)
            print(numpy.unique(labels))
            pyplot.imshow(labels)
            pyplot.show()
            raise Exception('Problem here')

        binary_mask = cv2.GaussianBlur(binary_mask.astype(numpy.float32), (11,11), 0)
        points = numpy.zeros( (labels_count-1,2), dtype=numpy.uint16)
        for i in range(1, labels_count):
            ith_component = (labels==i)*binary_mask
            points[i-1] = numpy.argwhere( ith_component==ith_component.max() )[-1]

        return points


    def logical_or(mask1, mask2):
        res = numpy.logical_or(mask1, mask2)
        res = numpy.where(res==True, 1, 0).astype(numpy.uint8)
        return res


    def find_min_cut_mask(f1, f2, f1_mask, f2_mask, point1, point2, combined):
        intersection_mask = f1_mask * f2_mask
        start_point, end_point = (point1,point2) if point1[0] < point2[0] else (point2, point1)
        
        y_min, y_max = start_point[0], end_point[0]

        y, x = numpy.nonzero(intersection_mask)
        x_min, x_max = x.min(), x.max()

        f1_rect = f1[y_min:y_max+1, x_min:x_max+1, :]
        f2_rect = f2[y_min:y_max+1, x_min:x_max+1, :]
        mask_rect = intersection_mask[y_min:y_max+1, x_min:x_max+1]

        matrix = ((f1_rect.astype(numpy.int64) - f2_rect.astype(numpy.int64))**2).sum(axis=2)

        values = numpy.where(intersection_mask == 1,0, INT64_MAX_VALUE)[y_min:y_max+1, x_min:x_max+1]
        parents = numpy.zeros( matrix.shape, dtype=numpy.int8)

        values[0,:] = INT64_MAX_VALUE
        values[0, start_point[1]-x_min-1] = 0

        for i in range(1,matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j == 0:
                    parent = values[i-1,j:j+3].argmin()
                elif j == 1:
                    parent = values[i-1,j-1:j+3].argmin()
                elif j == values.shape[1]-1:
                    parent = values[i-1,j-2:j+3].argmin() - 2
                elif j == values.shape[1]:
                    parent = values[i-1,j-2:j+2].argmin() - 2
                else:
                    parent = values[i-1,j-2:j+3].argmin() - 2

                parents[i,j] = parent
                values[i,j] = values[i-1,j+parent] + matrix[i,j]
            

        mask = numpy.zeros( f1.shape )
        for i in range(y_max+1, f1.shape[0])[::-1]:
            mask[i,:end_point[1]] = 1


        min_cut_index = end_point[1] - x_min
        for i in range(y_min, y_max+1)[::-1]:
            y = i
            x = min_cut_index + x_min
            # cv2.circle(combined, tuple([x,y]), 0, (255,0,0),1)
            mask[i,:x] = 1
            min_cut_index = parents[i-y_min,min_cut_index] + min_cut_index

        for i in range(y_min)[::-1]:
            mask[i,:start_point[1]] = 1

        return mask


    print('Starting part2')

    compute_ref_homographies()

    f90 = cv2.imread(f'inputs/videos/frames/frame90{QUALITY}.png')
    f270 = cv2.imread(f'inputs/videos/frames/frame270{QUALITY}.png')
    f450 = cv2.imread(f'inputs/videos/frames/frame450{QUALITY}.png')
    f630 = cv2.imread(f'inputs/videos/frames/frame630{QUALITY}.png')
    f810 = cv2.imread(f'inputs/videos/frames/frame810{QUALITY}.png')

    main_img = cv2.warpPerspective(f450, TRANSLATE_MATRIX, (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) ).astype(numpy.float32)

    main_img_mask = get_mask(TRANSLATE_MATRIX)
    main_img_border = cv2.morphologyEx(main_img_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    f270_hmg = numpy.matmul(TRANSLATE_MATRIX,ref_homographies['270'])

    f270_warped = cv2.warpPerspective(f270, f270_hmg, (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) ).astype(numpy.float32)
    f270_mask = get_mask(f270_hmg)
    f270_border = cv2.morphologyEx(f270_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    border_overlap = f270_border * main_img_border
    border_intersections = non_maximum_supression(border_overlap)

    mask = find_min_cut_mask(f270_warped, main_img, f270_mask, main_img_mask, border_intersections[0], border_intersections[1], None)

    main_img = mask * f270_warped + (1-mask) * main_img
    main_img_mask = logical_or(main_img_mask, f270_mask)
    print('f270 warped')

    #! --------------------------------------------------------------------------------------------------------------------------------------------

    main_img_border = cv2.morphologyEx(main_img_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    f90_hmg = numpy.matmul(TRANSLATE_MATRIX,ref_homographies['90'])

    f90_warped = cv2.warpPerspective(f90, f90_hmg, (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) ).astype(numpy.float32)
    f90_mask = get_mask(f90_hmg)

    f90_border = cv2.morphologyEx(f90_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    border_overlap = f90_border * main_img_border
    border_intersections = non_maximum_supression(border_overlap)

    mask = find_min_cut_mask(f90_warped, main_img, f90_mask, main_img_mask, border_intersections[0], border_intersections[1], None)

    main_img = mask * f90_warped + (1-mask) * main_img
    main_img_mask = logical_or(main_img_mask, f270_mask)
    print('f90 warped')

    #! --------------------------------------------------------------------------------------------------------------------------------------------

    main_img_border = cv2.morphologyEx(main_img_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    f630_hmg = numpy.matmul(TRANSLATE_MATRIX,ref_homographies['630'])

    f630_warped = cv2.warpPerspective(f630, f630_hmg, (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) ).astype(numpy.float32)
    f630_mask = get_mask(f630_hmg)

    f630_border = cv2.morphologyEx(f630_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    border_overlap = f630_border * main_img_border
    border_intersections = non_maximum_supression(border_overlap)

    mask = find_min_cut_mask(f630_warped, main_img, f630_mask, main_img_mask, border_intersections[0], border_intersections[1], None)

    main_img = mask * main_img + (1-mask) * f630_warped
    main_img_mask = logical_or(main_img_mask, f630_mask)
    print('f630 warped')

    #! --------------------------------------------------------------------------------------------------------------------------------------------

    main_img_border = cv2.morphologyEx(main_img_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    f810_hmg = numpy.matmul(TRANSLATE_MATRIX,ref_homographies['810'])

    f810_warped = cv2.warpPerspective(f810, f810_hmg, (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) ).astype(numpy.float32)
    f810_mask = get_mask(f810_hmg)

    f810_border = cv2.morphologyEx(f810_mask, cv2.MORPH_GRADIENT, numpy.ones((3,3)))

    border_overlap = f810_border * main_img_border
    border_intersections = non_maximum_supression(border_overlap)

    mask = find_min_cut_mask(f810_warped, main_img, f810_mask, main_img_mask, border_intersections[0], border_intersections[1], None)

    main_img = mask * main_img + (1-mask) * f810_warped
    main_img_mask = logical_or(main_img_mask, f810_mask)
    print('f810 warped')


    cv2.imwrite(f'outputs/images/‫‪res04-key-frames-panorama{QUALITY}.jpg', main_img)


# Depends on homographies
def part3():
    print('Starting part3: making warped images video')

    homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)
    for i in range(1,901):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
        hmg = homographies[i].reshape(3,3)
        hmg = numpy.matmul(TRANSLATE_MATRIX, hmg)
        frame_warped = cv2.warpPerspective(frame, hmg, (BACKGROUND_RESOLUTION[1],BACKGROUND_RESOLUTION[0]) ).astype(numpy.float32)
        cv2.imwrite(f'outputs/temps/q1_3_frames/q1_3_frame{i}{QUALITY}.jpg',frame_warped)

    #NOTE: Resolution must be even*even
    os.system(f'ffmpeg -start_number 1 -framerate 30 -i outputs/temps/q1_3_frames/q1_3_frame%d{QUALITY}.jpg outputs/videos/res05-reference-plane{QUALITY}.mp4')


# Depends on part3 output
def part4():
    print('Starting part4: Creating background homography')

    homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)

    curr_width = PART4_START_FROM

    if curr_width == 0:
        res = numpy.zeros( BACKGROUND_RESOLUTION )
    else:
        res = cv2.imread(f'outputs/temps/q1_4_frames/res06-background-panorama{QUALITY}_width_{PART4_START_FROM-20}.jpg')

    patch_width = PATCH_WIDTH
    while curr_width < BACKGROUND_RESOLUTION[1]:
        if curr_width + patch_width > BACKGROUND_RESOLUTION[1]:
            patch_width = BACKGROUND_RESOLUTION[1] - curr_width

        patch_mask = numpy.zeros(BACKGROUND_RESOLUTION[:2], dtype=numpy.uint8)
        patch_mask[:,curr_width:curr_width+patch_width] = 1

        colors = numpy.zeros((BACKGROUND_RESOLUTION[0],patch_width,3,900))

        alternate_mask = numpy.ones((BACKGROUND_RESOLUTION[0],patch_width,3), dtype=numpy.int16) * -256

        print(f'curr_width: {curr_width}/{BACKGROUND_RESOLUTION[1]}')

        for i in range(1,901):
            print(f'curr_width: {curr_width}/{BACKGROUND_RESOLUTION[1]} frame{i}')

            frame = cv2.imread(f'outputs/temps/q1_3_frames/q1_3_frame{i}{QUALITY}.jpg')

            hmg = numpy.matmul(TRANSLATE_MATRIX, homographies[i].reshape(3,3))
            frame_mask = get_mask(hmg)

            colored_mask = (patch_mask * frame_mask)[:, curr_width:curr_width+patch_width]
            temp = numpy.where(colored_mask==0, -1, 1)
            alternate_mask = alternate_mask * temp[:,:,None]
            colors[:,:,:,i-1] = alternate_mask
            colors[:,:,:,i-1][colored_mask==1] = frame[:, curr_width:curr_width+patch_width,:][colored_mask==1]

        patch_res = numpy.median(colors, axis=3)
        colors = numpy.concatenate((colors, -1*alternate_mask[:,:,:,None]), axis=3)
        patch_res[alternate_mask==256] = numpy.median(colors, axis=3)[alternate_mask==256]


        res[:,curr_width:curr_width+patch_width] = patch_res
        cv2.imwrite(f'outputs/temps/q1_4_frames/res06-background-panorama{QUALITY}_width_{curr_width}.jpg', res)

        curr_width += PATCH_WIDTH


    cv2.imwrite(f'outputs/images/‫‪res06-background-panorama{QUALITY}.jpg', res)


# Depends on part4 output
def part5():
    background = cv2.imread(f'outputs/images/‫‪res06-background-panorama{QUALITY}.jpg')
    translate_inverse = numpy.linalg.inv(TRANSLATE_MATRIX)
    homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)
    for i in range(1,901):
        print(f'frame{i}')
        hmg = homographies[i].reshape(3,3)
        hmg_inv = numpy.linalg.inv(hmg)
        hmg_inv = numpy.matmul(hmg_inv, translate_inverse)

        frame_warped = cv2.warpPerspective(background, hmg_inv, (VIDEO_RESLUTION[1],VIDEO_RESLUTION[0]) ).astype(numpy.float32)

        cv2.imwrite(f'outputs/temps/q1_5_frames/q1_5_frame{i}{QUALITY}.jpg',frame_warped)

    os.system(f'ffmpeg -start_number 1 -framerate 30 -i outputs/temps/q1_5_frames/q1_5_frame%d{QUALITY}.jpg outputs/videos/res07-background-video{QUALITY}.mp4')


thresholds = {
    # '' : 20000, #1
    '' : 20000,
    '_0.5': 20000
}
# Depends on part5 output
def part6():
    threshold = thresholds[QUALITY]
    for i in range(1,901):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png').astype(numpy.int32) #! int32 is Important
        frame_background = cv2.imread(f'outputs/temps/q1_5_frames/q1_5_frame{i}{QUALITY}.jpg')
        diff = numpy.sum((frame - frame_background)**2, axis=2)
        foreground_mask = numpy.where(diff>threshold, 1, 0).astype(numpy.uint8) 
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, numpy.ones((3,3))) #5,
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, numpy.ones((3,3))) #5,

        frame[foreground_mask==1] = frame[foreground_mask==1] - 100
        frame[:,:,2][foreground_mask==1] = (frame[:,:,2][foreground_mask==1] + 200)
        frame[frame<0] = 0
        frame[frame>255] = 255

        cv2.imwrite(f'outputs/temps/q1_6_frames/q1_6_frame{i}{QUALITY}.jpg',convert_to_uint8(frame))

    os.system(f'ffmpeg -start_number 1 -framerate 30 -i outputs/temps/q1_6_frames/q1_6_frame%d{QUALITY}.jpg outputs/videos/res08-foreground-video{QUALITY}.mp4')


def part7():
    background = cv2.imread(f'outputs/images/‫‪res06-background-panorama{QUALITY}.jpg')
    translate_inverse = numpy.linalg.inv(TRANSLATE_MATRIX)
    homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)
    for i in range(1,901):
        print(f'frame{i}')
        hmg = homographies[i].reshape(3,3)
        hmg_inv = numpy.linalg.inv(hmg)
        hmg_inv = numpy.matmul(hmg_inv, translate_inverse)

        frame_warped = cv2.warpPerspective(background, hmg_inv, (int(VIDEO_RESLUTION[1]*1.8),VIDEO_RESLUTION[0]) ).astype(numpy.float32)

        cv2.imwrite(f'outputs/temps/q1_7_frames/q1_7_frame{i}{QUALITY}.jpg',frame_warped)

    os.system(f'ffmpeg -start_number 1 -framerate 30 -i outputs/temps/q1_7_frames/q1_7_frame%d{QUALITY}.jpg -frames:v 525 outputs/videos/res09-background-video-wider{QUALITY}.mp4') # 525 frames for quality = 0.5


def part8():
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
        R = numpy.matmul(numpy.matmul(Rz,Ry),Rx)
        R = R/R[2,2]
        return R

    def compute_f():
        width = VIDEO_RESLUTION[1]
        all_f = numpy.zeros(901)
        for i in range(1,901):
            if i == 450:
                continue

            frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')
            H = homographies[i].reshape(3,3)
            print(f'frame{i}')
            # print(H)
            H = H/H[2,2]

            z = math.atan(-1 * H[0,1]/H[1,1])
            # print('z is: ', z)


            a = (H[2,0]/H[0,1]) * math.sin(z)
            # print('a is: ', a)
            temp = math.pow( (2*math.cos(z) - width*a*H[1,1]) / (2*H[1,1]) , 2)
            if temp>1:
                print('Precision was not enough')
                continue
            
            f =  math.sqrt(1 - temp) / a
            f = abs(f)
            all_f[i] = f

        all_f = all_f[all_f!=0]
        f_mean = numpy.mean(all_f, axis=0)
        f_std = numpy.std(all_f, axis=0, ddof=1)
        all_f = all_f[all_f<f_mean + 2*f_std]
        all_f = all_f[all_f>f_mean - 2*f_std]
        f_mean = numpy.mean(all_f, axis=0)
        f_std = numpy.std(all_f, axis=0, ddof=1)

        return f_mean

    # degree is odd
    def get_smooth_rotations(rotations, degree):
        half = int(degree/2)
        new_rotations = numpy.pad(rotations, ((0,0),(half, half)), mode='edge')
        new_rotations_x = numpy.convolve(new_rotations[0], numpy.ones(degree), 'valid') / degree
        new_rotations_y = numpy.convolve(new_rotations[1], numpy.ones(degree), 'valid') / degree
        new_rotations_z = numpy.convolve(new_rotations[2], numpy.ones(degree), 'valid') / degree
        new_rotations = numpy.stack((new_rotations_x,new_rotations_y,new_rotations_z), axis=0)
        x_indexs = numpy.arange(NUMBER_OF_FRAMES)
        # pyplot.subplot(211).plot(x_indexs, rotations[0])
        # pyplot.subplot(212).plot(x_indexs, new_rotations[0])
        # pyplot.savefig('outputs/temps/x_rotations.jpg')
        # pyplot.show()
        # pyplot.subplot(211).plot(x_indexs, rotations[1])
        # pyplot.subplot(212).plot(x_indexs, new_rotations[1])
        # pyplot.savefig('outputs/temps/y_rotations.jpg')
        # pyplot.show()
        # pyplot.subplot(211).plot(x_indexs, rotations[2])
        # pyplot.subplot(212).plot(x_indexs, new_rotations[2])
        # pyplot.savefig('outputs/temps/z_rotations.jpg')
        # pyplot.show()

        return new_rotations

    def make_frame_bigger(frame):
        heigth, width = frame.shape[:2]
        T = cv2.getRotationMatrix2D((width/2,heigth/2), 0, 1.1)
        frame = cv2.warpAffine(frame, T, (width, heigth))
        return frame

    homographies = cv2.imread(f'outputs/temps/homographies{QUALITY}.tiff', cv2.IMREAD_UNCHANGED)
    heigth, width = VIDEO_RESLUTION[:2]

    f = compute_f()
    print('f is: ', f)

    K = numpy.array([
        [f, 0, width/2 ],
        [0, f, heigth/2],
        [0, 0,    1    ]
    ])
    K_1 = numpy.linalg.inv(K)

    rotations = numpy.zeros((3,NUMBER_OF_FRAMES))
    for i in range(1,NUMBER_OF_FRAMES+1):
        print(f'frame{i}')
        H = homographies[i].reshape(3,3)
        R = numpy.matmul(K_1,numpy.matmul(H,K))

        x, y, z = Rotation.from_matrix(R).as_euler('xyz')
        rotations[0,i-1] = x
        rotations[1,i-1] = y
        rotations[2,i-1] = z


    rotations = get_smooth_rotations(rotations, MOVING_AVG_WIN)

    for i in range(1,NUMBER_OF_FRAMES+1):
        print(f'frame{i}')
        frame = cv2.imread(f'inputs/videos/frames/frame{i}{QUALITY}.png')

        R = create_rotation_matrix(rotations[0,i-1],rotations[1,i-1],rotations[2,i-1])
        H = homographies[i].reshape(3,3)
        Hp = numpy.matmul(K,numpy.matmul(R,K_1))
        hmg = numpy.matmul(numpy.linalg.inv(Hp), H)

        frame_warped = cv2.warpPerspective(frame, hmg, (VIDEO_RESLUTION[1],VIDEO_RESLUTION[0]) )
        frame_warped = make_frame_bigger(frame_warped)
        cv2.imwrite(f'outputs/temps/q1_8_frames/q1_8_frame{i}{QUALITY}_{MOVING_AVG_WIN}.jpg',frame_warped)

    os.system(f'ffmpeg -start_number 1 -framerate 30 -i outputs/temps/q1_8_frames/q1_8_frame%d{QUALITY}_{MOVING_AVG_WIN}.jpg outputs/videos/res10-video-shakeless{QUALITY}.mp4') # 525 frames for quality = 0.5
    # os.system(f'ffmpeg -start_number 1 -framerate 30 -i outputs/temps/q1_8_frames/q1_8_frame%d{QUALITY}_{MOVING_AVG_WIN}.jpg outputs/videos/res10-video-shakeless{QUALITY}_{MOVING_AVG_WIN}.mp4') # 525 frames for quality = 0.5


QUALITY = '1'
# QUALITY = '0.5'

QUALITY = '' if QUALITY in ['1', '1.0', ''] else '_' + QUALITY


INT64_MAX_VALUE = numpy.iinfo(numpy.int64).max/2

temp = cv2.imread(f'inputs/videos/frames/frame450{QUALITY}.png')
VIDEO_RESLUTION = temp.shape

CORNER_COORDS = numpy.array([
    [0, 0],
    [VIDEO_RESLUTION[1], 0],
    [VIDEO_RESLUTION[1], VIDEO_RESLUTION[0]],
    [0, VIDEO_RESLUTION[0]]
], dtype=numpy.int32)

PART4_START_FROM = 4760

PATCH_WIDTH = 20 # good for quality = 1
# PATCH_WIDTH = 60 # was good for quality = 1/2

ref_homographies = {}

NUMBER_OF_FRAMES = 900
MOVING_AVG_WIN = 201


# extract_frames()

# compute_homographies()

BACKGROUND_RESOLUTION, TRANSLATE_MATRIX = compute_background_resolution_and_translate_matrix() # homographies should have been computed before

# part1()

# part2()

# part3()

# part4()

# part5()

# part6()

# part7()

part8()

