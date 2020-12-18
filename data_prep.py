import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def data_preprocessing_A(labels_name, direct, image_direct):
    labels_name= 'labels.csv'
    labels_file=open(os.path.join(direct, labels_name),'r')
    image_paths = [os.path.join(image_direct, l) for l in os.listdir(image_direct)]
    lines = labels_file.readlines()
    #Get labels from csv file.
    labels_gender={line.split()[0]:line.split()[2] for line in lines[1:]}
    labels_smile={line.split()[0]:line.split()[3] for line in lines[1:]}
    return labels_gender, labels_smile, image_paths
#data_preprocessing('labels.csv', './Datasets/celeba', os.path.join('./Datasets/celeba', 'img'))

def extract_feat_A():
    #directory where the data is
    direct = './Datasets/celeba'
    #directory where labels are
    image_direct = os.path.join(direct, 'img')
    #get labels from csv file
    labels_gender, labels_smile, image_paths = data_preprocessing_A('labels.csv', direct, image_direct )
    target_size = None
    #find features for all images
    if os.path.isdir(image_direct):
        all_features = []
        all_labels_gender = []
        all_labels_smile = []
        for img_path in image_paths:
            file_name=img_path.split('/')[2].split('\\')[2]
            # load image
            img = image.img_to_array(
                image.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                #put all features of an image in a string
                print(file_name)
                all_features.append(features)
                all_labels_gender.append(labels_gender[file_name.split('.')[0]])
                all_labels_smile.append(labels_smile[file_name.split('.')[0]])
    #make previous strings into numpy arrays and return them
    landmark_features = np.array(all_features)
    gender_labels = (np.array(all_labels_gender))# + 1)/2 # simply converts the -1 into 0, so male=0 and female=1
    smile_labels= (np.array(all_labels_smile))
    gen=np.zeros(len(gender_labels))
    for i in range(len(gender_labels)):
        gen[i]=(int(gender_labels[i])+1)/2
    sm=np.zeros(len(smile_labels))
    for i in range(len(smile_labels)):
        sm[i]=(int(smile_labels[i])+1)/2
    return landmark_features, gen, sm

def data_preprocessing_B(labels_name, direct, image_direct):
    labels_name= 'labels.csv'
    labels_file=open(os.path.join(direct, labels_name),'r')
    image_paths = [os.path.join(image_direct, l) for l in os.listdir(image_direct)]
    lines = labels_file.readlines()
    #Get labels from csv file.
    labels_eye={line.split()[3]:line.split()[1] for line in lines[1:]}
    labels_face={line.split()[3]:line.split()[2] for line in lines[1:]}
    return labels_eye, labels_face, image_paths      
    
def extract_feat_B():
    #directory where the data is
    direct = './Datasets/cartoon_set'
    #directory where labels are
    image_direct = os.path.join(direct, 'img')
    #get labels from csv file
    labels_eye, labels_face, image_paths = data_preprocessing_B('labels.csv', direct, image_direct )
    target_size = None
    #find features for all images
    images=[]
    if os.path.isdir(image_direct):
        all_labels_eye = []
        all_labels_face = []
        for img_path in image_paths:
            file_name=img_path.split('/')[2].split('\\')[2]
            # load image
            print(file_name)
            img= cv2.imread(img_path)
            images.append(img)
            all_labels_eye.append(labels_eye[file_name])
            all_labels_face.append(labels_face[file_name])
    #make previous strings into numpy arrays and return them
    eye_labels = (np.array(all_labels_eye))
    face_labels= (np.array(all_labels_face))
    return images, eye_labels, face_labels


    
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image   