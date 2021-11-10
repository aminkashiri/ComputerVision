import cv2, numpy, os, glob
from matplotlib import pyplot
from os import path
from sklearn.neighbors import KNeighborsClassifier

def create_feature_vectors():
    def create_dir(*dir_paths):
        for dir_path in dir_paths:
            if not path.isdir(dir_path):
                os.makedirs(dir_path)

    flag = True
    for class_name in CLASS_NAMES:
        if not path.exists(path.join(CLASS_DESCRIPTORS_DIR, f'test_images_descriptors_{class_name}.jpg')):
            flag = False

    if flag and not RECOMPUTE_FEATURE_VECTORS:
        print(f'Using previously computed feature vectors for size: {SIZE}*{SIZE}',)
        return

    print(f'Discarding previously computed feature vectors and recomputing for size: {SIZE}*{SIZE}',)
    create_dir(CLASS_DESCRIPTORS_DIR)

    print(f'Creating data set with images of size {SIZE}*{SIZE}')
    for class_name in CLASS_NAMES:
        print('--> Class: ',class_name)
        image_paths = glob.glob(path.join(TRAIN_DIR, class_name,'*'))
        image_paths.sort()


        image_descriptors = numpy.zeros((len(image_paths),SIZE**2), dtype=numpy.uint8)
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (SIZE,SIZE), RESIZE_METHOD)
            image_descriptors[i,:] = image.ravel()
        cv2.imwrite(path.join(CLASS_DESCRIPTORS_DIR,f'train_images_descriptors_{class_name}.jpg'), image_descriptors)

        #? Beautiful as HELL!
        # pyplot.imshow(image_descriptors)
        # pyplot.show()

        image_paths = glob.glob(path.join(TEST_DIR, class_name,'*'))
        image_paths.sort()
        image_descriptors = numpy.zeros((len(image_paths),SIZE**2), dtype=numpy.uint8)
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (SIZE,SIZE), RESIZE_METHOD)
            image_descriptors[i,:] = image.ravel()
        cv2.imwrite(path.join(CLASS_DESCRIPTORS_DIR,f'test_images_descriptors_{class_name}.jpg'), image_descriptors)


def test_KNN():
    global BEST_ACCURACY, BEST_K, BEST_NORM, BEST_SIZE
    print('Running tests with parameters: ')
    print('--> K : ', K)
    print('--> p-norm: ', P_NORM)
    print('--> image size: ', SIZE)

    create_feature_vectors()

    train_descriptors = numpy.zeros((0,SIZE**2))
    train_labels = numpy.zeros((0), dtype=numpy.int32)
    for i, class_name in enumerate(CLASS_NAMES):
        class_descriptor = cv2.imread(path.join(CLASS_DESCRIPTORS_DIR,f'train_images_descriptors_{class_name}.jpg'), cv2.IMREAD_GRAYSCALE)
        class_label = numpy.zeros((len(class_descriptor)), dtype=numpy.int32) + i

        train_descriptors = numpy.concatenate((train_descriptors, class_descriptor))
        train_labels = numpy.concatenate((train_labels, class_label))


    test_descriptors = numpy.zeros((0,SIZE**2))
    test_labels = numpy.zeros((0), dtype=numpy.int32)
    for i, class_name in enumerate(CLASS_NAMES):
        class_descriptor = cv2.imread(path.join(CLASS_DESCRIPTORS_DIR,f'test_images_descriptors_{class_name}.jpg'), cv2.IMREAD_GRAYSCALE)
        class_label = numpy.zeros((len(class_descriptor))) + i

        test_descriptors = numpy.concatenate((test_descriptors, class_descriptor))
        test_labels = numpy.concatenate((test_labels, class_label))


    classifier = KNeighborsClassifier(n_neighbors=K, p=P_NORM)
    classifier.fit(train_descriptors, train_labels)

    results = classifier.predict(test_descriptors).ravel()

    corrects = (test_labels == results).sum()

    accuracy = corrects/len(test_labels)
    if accuracy > BEST_ACCURACY:
        BEST_ACCURACY = accuracy
        BEST_K = K
        BEST_NORM = P_NORM
        BEST_SIZE = SIZE

    print('--> Number of correct guesses: ',corrects)
    print('--> Accuracy = ', accuracy,'\n')



SIZE = 22
P_NORM = 1
K = 1

RESIZE_METHOD = cv2.INTER_LINEAR # cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA
RECOMPUTE_FEATURE_VECTORS = False
FIND_BEST_HYPER_PARAMETERS = False

BASE_DIR = path.dirname(__file__)
DATA_SET_DIR = path.join(BASE_DIR, 'inputs', 'Data')
TRAIN_DIR = path.join(DATA_SET_DIR,'Train')
TEST_DIR = path.join(DATA_SET_DIR,'Test')

CLASS_NAMES = [ folder_path.split(path.sep)[-1] for folder_path in glob.glob(path.join(TRAIN_DIR, '*'))]


BEST_K = -1
BEST_NORM = -1
BEST_SIZE = -1
BEST_ACCURACY = -1
if FIND_BEST_HYPER_PARAMETERS:
    for SIZE in range(16,50,2):
        CLASS_DESCRIPTORS_DIR = path.join(BASE_DIR,'outputs','temps',f'Data_{SIZE}','image_desciptors')
        for P_NORM in range(1,3):
            for K in range(1,21):
                test_KNN()


    print('Best result: ')
    print('--> K : ', BEST_K)
    print('--> p-norm: ', BEST_NORM)
    print('--> image size: ', BEST_SIZE)
    print('--> Accuracy = ', BEST_ACCURACY,'\n')

else:
    CLASS_DESCRIPTORS_DIR = path.join(BASE_DIR,'outputs','temps',f'Data_{SIZE}','image_desciptors')
    test_KNN()



