import cv2
import numpy as np
import tensorflow as tf
from shapely import geometry
import argparse
from measurements import Measurements
from object_detection.utils import label_map_util
from draw_prueba import Drawing
from outcomes import Outcomes
from calibrate import Calibration
import os

# Parser arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--model", required=True, help="path to object detection model")
parser.add_argument("-l", "--labels", required=True, help="path to labels file")
parser.add_argument("-i", "--input", default=0, type=str, help="path to optional input folder", required=True)
parser.add_argument("-o", "--output", type=str, default="results", help="path and name to optional output folder")
parser.add_argument("-t", "--threshold", type=float, default=0.8, help="minimum probability to filter weak detection")
parser.add_argument("-c", "--calibration", action="store_true", help="option for un-distort input image")
parser.add_argument("-r", "--resize", type=str, default="1,1", help="resize input image")
parser.add_argument("-H", "--camera_height", type=float, default=2.5, help="z-coordinate for camera positioning")
parser.add_argument("-p", "--people_height", type=float, default=1.7, help="z-coordinate for people high")
parser.add_argument("-a", "--angle", type=float, default=14, help="positioning angle in degrees")
args = parser.parse_args()
vargs = vars(args)

# Defining path and name of output picture
output_variable = vargs["output"]

# Storing paths
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,vargs["model"])
PATH_TO_LABELS = os.path.join(CWD_PATH,vargs["labels"])


PATH_TO_IMAGE = os.path.join(vargs["input"])

# Storing heights and angle
people_height = vargs["people_height"]
camera_height = vargs["camera_height"]
angle = float(np.pi*vargs["angle"]/180)   # radians


# Number of classes in the trained model
NUM_CLASSES = 2

# Loading the label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loading Tensorflow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Defining input and output tensors for detections
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

resized = (int(vargs["resize"].split(',')[0]),int(vargs["resize"].split(',')[1]))  ## y,x

active = 0
log_0 = []
log_error = []
for img in os.listdir(os.path.join(PATH_TO_IMAGE)):
    if img.endswith('png'):
        try:
            print(img)
            image = cv2.imread(os.path.join(PATH_TO_IMAGE,img))

            if resized[0]==1:
                resized = image.shape[:2]

            image = cv2.resize(image, (resized[1],resized[0]), interpolation = cv2.INTER_AREA)

            print('-------------------------------------------------------------------------')
            if vargs["calibration"] == True:
                calibration = Calibration([resized[1],resized[0]])
                calibration.Checkboard()
                balance = 1
                image_calib = image.copy()
                image = calibration.Undistort(image,balance)

            # Calling objects
            if active==0:
                outcomes = Outcomes()
                measures = Measurements(outcomes.Telling())
            else:
                outcomes = outcomes
                measures = measures

            # Auxiliary copies
            copy_image = image.copy()
            final_image = image.copy()

            # Expanding image dimensions to have shape: [1, None, None, 3]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_expanded = np.expand_dims(image_rgb, axis=0)

            # Object to draw detections
            draws = Drawing(outcomes.Telling(),image,copy_image,final_image,vargs["threshold"],angle)

            # Get polygon and centroid points

            pts = np.array(draws.Generate_Polygon('Image '+vargs["input"]), np.int32)

            poly = geometry.Polygon(pts)
            centroid = np.array(list(poly.centroid.coords)[0])

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Reshaping useful arrays about details of detections
            boxes = boxes.reshape(boxes.shape[1],4)
            scores = scores.reshape(scores.shape[1],1)
            classes = classes.reshape(classes.shape[1],1)

            draws.Prepare_data(scores,boxes,classes)

            # First detections
            detections_1 = draws.Draw_detections(1,camera_height,people_height)
            print('-------------------------------------------------------------------------')
            print('PASSENGERS DETECTED =',detections_1)

            # Drawing polygon and its centroid
            cv2.circle(copy_image,(int(centroid[0]),int(centroid[1])),6,(0,255,255),-1)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(copy_image,[pts],True,(0,255,255))
            cv2.polylines(image,[pts],True,(0,255,255))
            #cv2.imshow('Area selection',copy_image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            people = int(detections_1)
            output_variable_def = os.path.join(output_variable,img)

            print(output_variable)
            if people >= 1:
                polygon_area = draws.Voronoi_diagram(image,output_variable_def,outcomes.Telling(),people,1)#folder) ## in slf
            else:
                # log_0.append(img.split('.')[0]+'_'+str(folder)+'_'+str(people)+'.'+img.split('.')[1])
                log_0.append(img)


        #cv2.imshow('Image',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()




            active+=1
        except:
            log_error.append(img)

with open('results/sequence_log_0.txt', 'w') as f:
    for line in log_0:
        f.write(line)
        f.write('\n')

with open('results/sequence_log_error.txt', 'w') as f:
    for line in log_error:
        f.write(line)
        f.write('\n')


# 2nd calculations (refinement)
print('PASSENGERS DETECTED =',people)
print('------------------------------------------------------------------------')
print("PROGRAM FINISHED")
print('-------------------------------------------------------------------------')
