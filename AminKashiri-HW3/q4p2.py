import os
os.environ['OPENCV_IO_MAX_IMAGE_HEIGHT'] = '123456789123456789'
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '123456789123456789'
import cv2, numpy, os, glob
from matplotlib import pyplot
from os import path


def compute_feature_vectors():
    if not RECOMPUTE_FEATURE_VECTORS and os.path.exists(path.join(TEMPS_DIR,'interest_points_feature_vectors_train.tiff')):
        print('Using saved feature vectors')
        feature_vectors_train = cv2.imread(path.join(TEMPS_DIR,'interest_points_feature_vectors_train.tiff'), cv2.IMREAD_UNCHANGED)
        feature_vectors_test = cv2.imread(path.join(TEMPS_DIR,'interest_points_feature_vectors_test.tiff'), cv2.IMREAD_UNCHANGED)
    else:
        print('Discarding saved feature vectors, and recomputing')
        feature_vectors_train = numpy.zeros((0,128), dtype=numpy.float32)
        feature_vectors_test = numpy.zeros((0,128), dtype=numpy.float32)
        for class_name in CLASS_NAMES:
            print('--> Computing feature vector for class: ', class_name)

            image_paths = glob.glob(path.join(TRAIN_DIR, class_name,'*'))
            image_paths.sort()  
            for image_path in image_paths:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                sift = cv2.SIFT_create()
                image_keypoints, image_features = sift.detectAndCompute(img, None) # img_descriptors are (128,)

                if not image_features is None:
                    feature_vectors_train = numpy.concatenate((feature_vectors_train, image_features))


            image_paths = glob.glob(path.join(TEST_DIR, class_name,'*'))
            image_paths.sort()  
            for image_path in image_paths:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                sift = cv2.SIFT_create()
                image_keypoints, image_features = sift.detectAndCompute(img, None) # img_descriptors are (128,)

                if not image_features is None:
                    feature_vectors_test = numpy.concatenate((feature_vectors_test, image_features))

        cv2.imwrite(path.join(TEMPS_DIR,'interest_points_feature_vectors_train.tiff'),feature_vectors_train)
        cv2.imwrite(path.join(TEMPS_DIR,'interest_points_feature_vectors_test.tiff'),feature_vectors_test)


    return feature_vectors_train, feature_vectors_test


def compute_visual_words():
    if not RECOMPUTE_DICTIONARY and os.path.exists(path.join(TEMPS_DIR,f'dictionary_{DICTIONARY_SIZE}.tiff')):
        print('Using saved dictionary for visual words')
        centers = cv2.imread(path.join(TEMPS_DIR,f'dictionary_{DICTIONARY_SIZE}.tiff'), cv2.IMREAD_UNCHANGED)
    else:
        print('Discarding saved dictionary for visual words, and recomputing dictionary')

        print('--> TRAIN_FEATURE_VECTORS shape is: ', TRAIN_FEATURE_VECTORS.shape, ' so number of all feature vectors = ', TRAIN_FEATURE_VECTORS.shape[0])
        compactness, labels, centers = cv2.kmeans(TRAIN_FEATURE_VECTORS, DICTIONARY_SIZE, None, K_MEANS_FINISH_SITUATUON, 10, cv2.KMEANS_RANDOM_CENTERS)

        # print('compactness is: ')
        # print(compactness)
        # print('labels are: ')
        # print(labels)
        # print(labels.shape)
        # print('centers are: ')
        # print(centers)
        # print(centers.dtype)
        # print(centers.shape)

        cv2.imwrite(path.join(TEMPS_DIR,f'dictionary_{DICTIONARY_SIZE}.tiff'), centers)

    return centers


def compute_histograms_and_test():
    global BEST_ACCURACY, BEST_K, BEST_DICTIONARY_SIZE

    if not RECOMPUTE_HISTOGRAMS and os.path.exists(path.join(TEMPS_DIR,f'test_labels_{DICTIONARY_SIZE}.tiff')):
        print('Using saved histograms as descriptors for images')
        train_histograms = cv2.imread(path.join(TEMPS_DIR,f'train_histograms_{DICTIONARY_SIZE}.tiff'), cv2.IMREAD_UNCHANGED)
        train_labels = cv2.imread(path.join(TEMPS_DIR,f'train_labels_{DICTIONARY_SIZE}.tiff'), cv2.IMREAD_UNCHANGED).astype(numpy.int32).ravel()

        test_histograms = cv2.imread(path.join(TEMPS_DIR,f'test_histograms_{DICTIONARY_SIZE}.tiff'), cv2.IMREAD_UNCHANGED)
        test_labels = cv2.imread(path.join(TEMPS_DIR,f'test_labels_{DICTIONARY_SIZE}.tiff'), cv2.IMREAD_UNCHANGED).astype(numpy.int32).ravel()
    else:
        print('Discarding saved histograms as descriptors for images and recomputing')

        knn = cv2.ml.KNearest_create()
        knn.train(DICTIONARY.astype(numpy.float32), cv2.ml.ROW_SAMPLE, numpy.arange(len(DICTIONARY)))

        train_histograms = numpy.zeros((0,DICTIONARY_SIZE))
        test_histograms = numpy.zeros((0,DICTIONARY_SIZE))

        test_labels = numpy.zeros((0), dtype=numpy.int32)
        train_labels = numpy.zeros((0), dtype=numpy.int32)

        for i, class_name in enumerate(CLASS_NAMES):
            print('--> ', class_name)

            # Compute for train data
            image_paths = glob.glob(path.join(TRAIN_DIR, class_name,'*'))
            image_paths.sort()  
            counter = 0
            for image_path in image_paths:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                sift = cv2.SIFT_create()
                image_keypoints, image_features = sift.detectAndCompute(img, None) 

                if not image_features is None:
                    _, results, neighbours, distances = knn.findNearest(image_features.astype(numpy.float32), 1)

                    results = results.ravel().astype(numpy.uint32)
                    histogram = numpy.histogram(results, bins=DICTIONARY_SIZE, range=(0,DICTIONARY_SIZE))[0]
                    # histogram = histogram / histogram.sum()
                    train_histograms = numpy.vstack((train_histograms, histogram))
                    counter += 1

            class_label = numpy.zeros((counter), dtype=numpy.int32) + i
            train_labels = numpy.concatenate((train_labels, class_label))


            # Compute for test data
            image_paths = glob.glob(path.join(TEST_DIR, class_name,'*'))
            image_paths.sort()  
            counter = 0
            for image_path in image_paths:
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                sift = cv2.SIFT_create()
                image_keypoints, image_features = sift.detectAndCompute(img, None) 

                if not image_features is None:
                    _, results, neighbours, distances = knn.findNearest(image_features.astype(numpy.float32), 1)
                    results = results.ravel().astype(numpy.uint32)
                    histogram = numpy.histogram(results, bins=DICTIONARY_SIZE, range=(0,DICTIONARY_SIZE))[0]
                    # histogram = histogram / histogram.sum()
                    test_histograms = numpy.vstack((test_histograms, histogram))
                    counter += 1

            class_label = numpy.zeros((counter), dtype=numpy.int32) + i
            test_labels = numpy.concatenate((test_labels, class_label))

        cv2.imwrite(path.join(TEMPS_DIR,f'train_histograms_{DICTIONARY_SIZE}.tiff'), train_histograms)
        cv2.imwrite(path.join(TEMPS_DIR,f'train_labels_{DICTIONARY_SIZE}.tiff'), train_labels)
        cv2.imwrite(path.join(TEMPS_DIR,f'test_histograms_{DICTIONARY_SIZE}.tiff'), test_histograms)
        cv2.imwrite(path.join(TEMPS_DIR,f'test_labels_{DICTIONARY_SIZE}.tiff'), test_labels)


    knn = cv2.ml.KNearest_create()
    knn.train(train_histograms.astype(numpy.float32), cv2.ml.ROW_SAMPLE, train_labels)
    _, results, neighbours, distances = knn.findNearest(test_histograms.astype(numpy.float32), K)

    results = results.ravel()
    corrects = (test_labels == results).sum()

    accuracy = corrects/len(test_labels)
    if accuracy > BEST_ACCURACY:
        BEST_ACCURACY = accuracy
        BEST_K = K
        BEST_DICTIONARY_SIZE = DICTIONARY_SIZE

    print('--> Number of correct guesses: ',corrects)
    print('--> Accuracy = ', accuracy,'\n')


K = 20
DICTIONARY_SIZE = 110

BASE_DIR = path.dirname(__file__)
DATA_SET_DIR = path.join(BASE_DIR, 'inputs', 'Data')
TRAIN_DIR = path.join(DATA_SET_DIR,'Train')
TEST_DIR = path.join(DATA_SET_DIR,'Test')
TEMPS_DIR = path.join(BASE_DIR, 'outputs','temps')


RECOMPUTE_FEATURE_VECTORS = False
RECOMPUTE_DICTIONARY = False
RECOMPUTE_HISTOGRAMS = False
FIND_BEST_HYPER_PARAMETERS = False

CLASS_NAMES = [ folder_path.split(path.sep)[-1] for folder_path in glob.glob(path.join(TRAIN_DIR, '*'))]


TRAIN_FEATURE_VECTORS, TEST_FEATURE_VECTORS = compute_feature_vectors()

K_MEANS_FINISH_SITUATUON = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,10, 0.1)


BEST_K = -1
BEST_DICTIONARY_SIZE = -1
BEST_ACCURACY = -1
if FIND_BEST_HYPER_PARAMETERS:
    for DICTIONARY_SIZE in range(50,201,20):
        for K in range(1,21):
            print('Running tests with parameters: ')
            print('--> K : ', K)
            print('--> DICTIONARY_SIZE: ', DICTIONARY_SIZE)
            DICTIONARY = compute_visual_words()

            compute_histograms_and_test()

    print('Best result: ')
    print('--> K : ', BEST_K)
    print('--> DICTIONARY_SIZE : ', BEST_DICTIONARY_SIZE)
    print('--> Accuracy = ', BEST_ACCURACY,'\n')

else:
    DICTIONARY = compute_visual_words()
    print('Running tests with parameters (with KNN as query algorithm): ')
    print('--> K : ', K)
    print('--> DICTIONARY_SIZE: ', DICTIONARY_SIZE)
    compute_histograms_and_test()

