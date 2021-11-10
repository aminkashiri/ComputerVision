import cv2, numpy, shutil, os, random, math, heapq, pickle
from pprint import pprint
from glob import glob
from matplotlib import pyplot
from os import path
from skimage.feature import hog
from sklearn import metrics, svm
import logging

parameters_string = lambda: f'_feature_vectors_image-size={IMAGE_SIZE[0]}*{IMAGE_SIZE[1]}_block-size={BLOCK_SIZE[0]}*{BLOCK_SIZE[1]}_block-norm={BLOCK_NORM}_cell-size={CELL_SIZE[0]}*{CELL_SIZE[1]}_orientations={ORIENTATION}.tiff'
parameters_string_svm = lambda: f'_feature_vectors_image-size={IMAGE_SIZE[0]}*{IMAGE_SIZE[1]}_block-size={BLOCK_SIZE[0]}*{BLOCK_SIZE[1]}_block-norm={BLOCK_NORM}_cell-size={CELL_SIZE[0]}*{CELL_SIZE[1]}_orientations={ORIENTATION}_kernel={KERNEL}_decision={DECISION}'
parameters_string_verbose = lambda: f"""
                        IMAGE_SIZE: {IMAGE_SIZE},
                        ORIENTATION: {ORIENTATION},
                        CELL_SIZE: {CELL_SIZE},
                        BLOCK_SIZE: {BLOCK_SIZE},
                        BLOCK_NORM: {BLOCK_NORM},
                        KERNEL: {KERNEL},
                        DECISION: {DECISION},
                        FEATURE_VECTORS_NAME: {parameters_string()}"""


class Record:

    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.IMAGE_SIZE =  IMAGE_SIZE
        self.ORIENTATION = ORIENTATION
        self.CELL_SIZE = CELL_SIZE
        self.BLOCK_SIZE = BLOCK_SIZE
        self.BLOCK_NORM = BLOCK_NORM
        self.KERNEL = KERNEL
        self.DECISION = DECISION
        self.FEATURE_VECTORS_NAME = parameters_string()

    def __lt__(self, other):
        return self.accuracy < other.accuracy


    def simple_to_string(self):
        return str(self.accuracy) + ' | ' + str(self.IMAGE_SIZE) + ' | ' + str(self.ORIENTATION) + ' | ' + str(self.CELL_SIZE) + ' | ' + str(self.BLOCK_SIZE) + ' | ' + str(self.BLOCK_NORM) + ' | ' + str(self.KERNEL) + ' | ' + str(self.DECISION) + ' | ' + str(self.FEATURE_VECTORS_NAME)

    def __str__(self):
        return f"""accuracy: {self.accuracy}
        IMAGE_SIZE: {self.IMAGE_SIZE},
        ORIENTATIONS: {self.ORIENTATION},
        CELL_SIZE: {self.CELL_SIZE},
        BLOCK_SIZE: {self.BLOCK_SIZE},
        BLOCK_NORM: {self.BLOCK_NORM},
        KERNEL: {self.KERNEL},
        DECISION: {self.DECISION},
        FEATURE_VECTORS_NAME: {self.FEATURE_VECTORS_NAME}"""
    __repr__ = __str__


def get_SVM_model(probability=False):
    Print('Training model')

    p = '_p' if probability else ''

    model_path = 'model' + parameters_string_svm() + p + '.sav'
    model_path = path.join(MODELS_DIR,model_path)

    if path.isfile(model_path):
        classifier = pickle.load(open(model_path, 'rb'))
        Print('--> Model was already trained')
    else:
        train_face_feature_vectors = cv2.imread(path.join(FEATURE_VECTORS_DIR,'train_face'+parameters_string()),  cv2.IMREAD_UNCHANGED)
        train_negative_feature_vectors = cv2.imread(path.join(FEATURE_VECTORS_DIR,'train_negative'+parameters_string()),  cv2.IMREAD_UNCHANGED)
        train_feature_vectors = numpy.concatenate( (train_face_feature_vectors, train_negative_feature_vectors) )
        train_labels = numpy.concatenate( (numpy.ones(len(train_face_feature_vectors)), numpy.zeros(len(train_negative_feature_vectors)))  )

        classifier = svm.SVC(
            decision_function_shape=DECISION,
            kernel=KERNEL,
            probability=probability)
        classifier.fit(train_feature_vectors, train_labels)

        pickle.dump(classifier, open(model_path, 'wb'))
        Print('--> Training model finished')
    
    return classifier


def init_logger():
    create_dir(LOGS_DIR)

    a = [
    ORIENTATIONS,
    BLOCK_NORMS,
    IMAGE_SIZE,
    CELL_SIZES,
    BLOCK_SIZES ,
    KERNELS,
    DECISIONS,
    ]

    a = list(map(str, a))
    s = ' - '.join(a)
    s = s + '.log'
    s = path.join(LOGS_DIR,s)

    if path.isfile(s):
        print('Logger is not needed')
        s = path.join(LOGS_DIR,'junk.log')


    logger = logging.getLogger()  
    for handler in logger.handlers[:]: 
        logger.removeHandler(handler)

    logging.basicConfig(filename=s,format='%(message)s',filemode='w', encoding='utf-8', level=logging.INFO)


def Print(*args):
    print(*args) if VERBOSE else None


heap = []
def keep_records(accuracy):
    global heap

    r = Record(accuracy)

    logging.info(r.simple_to_string())

    if len(heap) == N_BEST_PARAMETERS and accuracy < heap[0].accuracy:
            return

    if len(heap) < N_BEST_PARAMETERS:
        heapq.heappush(heap, r)
    else:
        heapq.heappushpop(heap, r)


def get_feature_vectors_size():
    x = math.floor(IMAGE_SIZE[0]/CELL_SIZE[0])
    y = math.floor(IMAGE_SIZE[1]/CELL_SIZE[1])
    number_of_blocks = (x-(BLOCK_SIZE[0]-1)) * (y-(BLOCK_SIZE[1]-1))
    number_of_cells_in_a_block = BLOCK_SIZE[0] * BLOCK_SIZE[1]
    feature_vector_size = number_of_blocks * number_of_cells_in_a_block * ORIENTATION
    return feature_vector_size


def create_dir(*dir_paths):
    for dir_path in dir_paths:
        if not path.isdir(dir_path):
            os.makedirs(dir_path)


def unpack_datasets():
    def unpack_dataset(input_dataset_dir, output_dataset_dir, number_of_images):
        create_dir(output_dataset_dir)
        
        indexes = list(range(number_of_images))
        random.shuffle(indexes)

        if path.isfile(path.join(output_dataset_dir,f'img_{number_of_images-1}.jpg')):
            Print('     --> Dataset was already unpacked\n')
            return

        folders_paths = [ folder_path for folder_path in glob(path.join(input_dataset_dir, '*'))]
        counter = 0
        for folder_path in folders_paths:
            Print('     --> Unpacking image:',counter)
            for face_path in glob(path.join(folder_path,'*')):
                shutil.copyfile(face_path, path.join(output_dataset_dir,f'img_{indexes[counter]}.jpg'))
                counter += 1
        Print('')


    def double_negative_dataset():
        Print('     Augmenting negative dataset by flipping')
        if path.isfile(path.join(NEGATIVE_DATASET_DIR,f'img_{5913*2-1}.jpg')):
            Print('     --> Dataset was already augmented\n')
            return

        for i in range(5913):
            img_path = path.join(NEGATIVE_DATASET_DIR,f'img_{i}.jpg')
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            cv2.imwrite(path.join(NEGATIVE_DATASET_DIR,f'img_{i+5913}.jpg'), img)

        Print('     --> Dataset augmented successfuly\n')

    Print('Unpacking datasets')

    Print('     Unpacking face dataset')
    input_face_dataset_dir = path.join(DATASETS_DIR,'lfw')
    unpack_dataset(input_face_dataset_dir,FACE_DATASET_DIR, 13233)

    Print('     Unpacking negative dataset')
    input_negative_dataset_dir = path.join(DATASETS_DIR,'archive','natural_images')

    if path.isdir(os.path.join(input_negative_dataset_dir,'person')):
        shutil.rmtree(os.path.join(input_negative_dataset_dir,'person'))

    unpack_dataset(input_negative_dataset_dir, NEGATIVE_DATASET_DIR, 5913)

    double_negative_dataset()

    Print('Datasets unpacked successfuly\n')


def compute_feature_vectors():
    Print('Computing feature vectors with: ',parameters_string())
    def compute_feature_vectors_util(indexes, dataset_dir, name):
        Print('     Computing ',name)
        if path.isfile(path.join(FEATURE_VECTORS_DIR,name+parameters_string())):
            Print('     --> feature vectors were already computed')
            return 

        feature_vectors = numpy.zeros((len(indexes),get_feature_vectors_size()))

        for i in indexes:
            image_path = path.join(dataset_dir,f'img_{i}.jpg')
            image = cv2.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE, None)

            feature_vector = hog(
                image, 
                orientations = ORIENTATION, 
                pixels_per_cell = CELL_SIZE,
                cells_per_block = BLOCK_SIZE, 
                block_norm=BLOCK_NORM
            )
            feature_vectors[i-indexes[0],:] = feature_vector.reshape(1,-1)


        cv2.imwrite(path.join(FEATURE_VECTORS_DIR,name+parameters_string()), feature_vectors)


    if get_feature_vectors_size() == 0:
        return False

    compute_feature_vectors_util(range(9800), FACE_DATASET_DIR,'train_face')
    compute_feature_vectors_util(range(9800,10800), FACE_DATASET_DIR,'validation_face')
    compute_feature_vectors_util(range(10800,11800), FACE_DATASET_DIR,'test_face')
    compute_feature_vectors_util(range(9800), NEGATIVE_DATASET_DIR,'train_negative')
    compute_feature_vectors_util(range(9800,10800), NEGATIVE_DATASET_DIR,'validation_negative')
    compute_feature_vectors_util(range(10800,11800), NEGATIVE_DATASET_DIR,'test_negative')

    return True


def run_test(validation=False):
    classifier = get_SVM_model()

    if validation:
        test_face_feature_vectors = cv2.imread(path.join(FEATURE_VECTORS_DIR,'validation_face'+parameters_string()),  cv2.IMREAD_UNCHANGED)
        test_negative_feature_vectors = cv2.imread(path.join(FEATURE_VECTORS_DIR,'validation_negative'+parameters_string()),  cv2.IMREAD_UNCHANGED)
        test_feature_vectors = numpy.concatenate( (test_face_feature_vectors, test_negative_feature_vectors) )
        test_labels = numpy.concatenate( (numpy.ones(len(test_face_feature_vectors)), numpy.zeros(len(test_negative_feature_vectors)))  )
    else:
        print('Final test result: ')
        test_face_feature_vectors = cv2.imread(path.join(FEATURE_VECTORS_DIR,'test_face'+parameters_string()),  cv2.IMREAD_UNCHANGED)
        test_negative_feature_vectors = cv2.imread(path.join(FEATURE_VECTORS_DIR,'test_negative'+parameters_string()),  cv2.IMREAD_UNCHANGED)
        test_feature_vectors = numpy.concatenate( (test_face_feature_vectors, test_negative_feature_vectors) )
        test_labels = numpy.concatenate( (numpy.ones(len(test_face_feature_vectors)), numpy.zeros(len(test_negative_feature_vectors)))  )


    results = classifier.predict(test_feature_vectors)
    corrects = (test_labels == results.ravel()).sum()

    accuracy = corrects/len(test_labels)
    print('--> Accuracy = ', accuracy,'\n')

    if validation:
        keep_records(accuracy)
    else:
        metrics.plot_roc_curve(classifier, test_feature_vectors, test_labels)  
        pyplot.savefig(path.join(OUTPUT_IMAGES_DIR,'res1.jpg'))   
        pyplot.title('ROC curve')

        metrics.plot_precision_recall_curve(classifier, test_feature_vectors, test_labels)
        pyplot.title('Precision-Recall curve')
        pyplot.tight_layout()
        pyplot.savefig(path.join(OUTPUT_IMAGES_DIR,'res2.jpg'))   

        decision_function = classifier.decision_function(test_feature_vectors)
        average_precision = metrics.average_precision_score(test_labels, decision_function)
        print('-- Average Prescision is :', average_precision)


def FaceDetector(image):
    classifier = get_SVM_model(True)

    steps = 7
    bounding_boxes = []
    scores = []
    scale = 1
    small_image = image.copy()
    detections = numpy.zeros((small_image.shape[0]-IMAGE_SIZE[0],small_image.shape[1]-IMAGE_SIZE[1]))
    while small_image.shape[0] > IMAGE_SIZE[0] and  small_image.shape[1] > IMAGE_SIZE[1]:
        print('Running with scale: ',scale)
        for i in range(0,small_image.shape[0]-IMAGE_SIZE[0],steps):
            print(i,'/',small_image.shape[0]-IMAGE_SIZE[0])
            for j in range(0,small_image.shape[1]-IMAGE_SIZE[1],steps):
                window = small_image[i:i+IMAGE_SIZE[0], j:j+IMAGE_SIZE[1]]
                feature_vector = hog(
                    window, 
                    orientations = ORIENTATION, 
                    pixels_per_cell = CELL_SIZE,
                    cells_per_block = BLOCK_SIZE, 
                    block_norm=BLOCK_NORM
                ).reshape(1,-1)
                result = classifier.predict_proba(feature_vector)

                if result[0][1] > THRESHOLD:
                    start_x = int(i / scale)
                    start_y = int(j / scale)
                    height = int(IMAGE_SIZE[0]/scale)
                    width = int(IMAGE_SIZE[1]/scale)
                    end_x = start_x+height
                    end_y = start_y+width

                    bounding_boxes.append([start_y,start_x,width,height])
                    scores.append(result[0][1])

        scale -= 0.05
        small_image = cv2.resize(image, None, None, fx=scale, fy=scale)


    image_copy = image.copy()
    detections = cv2.dnn.NMSBoxes(bounding_boxes, scores, THRESHOLD, NMS_THRESHOLD)
    for detection in detections:
        index = detection[0]
        bounding_box = bounding_boxes[index]
        start_y = bounding_box[0]
        start_x = bounding_box[1]
        width = bounding_box[2]
        height = bounding_box[3]
        end_x = start_x+height
        end_y = start_y+width

        cv2.rectangle(image_copy, (start_y,start_x),(end_y,end_x), (255,0,0))

    return image_copy



VERBOSE = True

FIND_BEST_PARAMETERS = True
RUN_ON_TEST = True
N_BEST_PARAMETERS = 5

NMS_THRESHOLD = 0.1
THRESHOLD = 0.999


BASE_DIR = path.dirname(__file__)
INPUTS_DIR = path.join(BASE_DIR, 'inputs')
INPUT_IMAGES_DIR = path.join(INPUTS_DIR, 'images')
DATASETS_DIR = path.join(INPUTS_DIR, 'datasets')
FACE_DATASET_DIR = path.join(DATASETS_DIR,'face_dataset')
NEGATIVE_DATASET_DIR = path.join(DATASETS_DIR,'negative_dataset')
OUTPUT_IMAGES_DIR = path.join(BASE_DIR, 'outputs', 'images')
FEATURE_VECTORS_DIR = path.join(BASE_DIR, 'temps', 'feature_vectors')
MODELS_DIR = path.join(BASE_DIR, 'temps', 'models')
LOGS_DIR = path.join(BASE_DIR, 'temps', 'logs')

# Parameters to check in finding best parameters
ORIENTATIONS = [8,16]
BLOCK_NORMS = ['L1','L1-sqrt','L2','L2-Hys']
IMAGE_SIZES = [(100,100),(80,80),(60,60)]
CELL_SIZES = [(8,8),(16,16)]
BLOCK_SIZES = [(1,1),(2,2)]
KERNELS = ['linear','rbf','poly']
DECISIONS = ['ovr','ovo']


# Best result
IMAGE_SIZE = (100,100)
ORIENTATION = 16
CELL_SIZE = (8,8)
BLOCK_SIZE = (2,2)
BLOCK_NORM = 'L2'
KERNEL = 'poly'
DECISION = 'ovo'


create_dir(FEATURE_VECTORS_DIR, MODELS_DIR)

unpack_datasets()

if FIND_BEST_PARAMETERS:
    for IMAGE_SIZE in IMAGE_SIZES:
        init_logger()
        for ORIENTATION in ORIENTATIONS:
            for CELL_SIZE in CELL_SIZES:
                for BLOCK_SIZE in BLOCK_SIZES:
                    for BLOCK_NORM in BLOCK_NORMS:
                        success = compute_feature_vectors()
                        if success:
                            for KERNEL in KERNELS:
                                for DECISION in DECISIONS:
                                    print('Running with parameters: ')
                                    print(parameters_string_verbose())
                                    run_test(validation=True)

    print('Best parameters: ')
    pprint(heap)

    r = heap[N_BEST_PARAMETERS-1]
    IMAGE_SIZE = r. IMAGE_SIZE
    ORIENTATION = r.ORIENTATION
    CELL_SIZE = r.CELL_SIZE
    BLOCK_SIZE = r.BLOCK_SIZE
    BLOCK_NORM = r.BLOCK_NORM
    KERNEL = r.KERNEL
    DECISION = r.DECISION
    FEATURE_VECTORS_NAME = r.FEATURE_VECTORS_NAME

    print('Running final test with parameters (best parameters): ')
    print(parameters_string_verbose())
    run_test()

elif RUN_ON_TEST:
    get_feature_vectors_size()
    print('Running with parameters: ')
    print(parameters_string_verbose())
    compute_feature_vectors()
    run_test()


image = cv2.imread(path.join(INPUTS_DIR,'images','Melli.jpg'))
result = FaceDetector(image)
cv2.imwrite(path.join(OUTPUT_IMAGES_DIR,'res4.jpg'),result)
image = cv2.imread(path.join(INPUTS_DIR,'images','Persepolis.jpg'))
result = FaceDetector(image)
cv2.imwrite(path.join(OUTPUT_IMAGES_DIR,'res5.jpg'),result)
image = cv2.imread(path.join(INPUTS_DIR,'images','Esteghlal.jpg'))
result = FaceDetector(image)
cv2.imwrite(path.join(OUTPUT_IMAGES_DIR,'res6.jpg'),result)


