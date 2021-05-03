# Importing the Python packages
import cv2
import time 
import numpy as np 
np.random.seed(100)
import imutils

# Other packages
import FeatureExtraction as ftx
import ObjectDetection as obj
import ImageUtils


if __name__=="__main__":
    # --------- MASK RCNN --------- #
    # # MODEL PARAMETERS
    modelname = "mask-rcnn-coco"
    proto     = "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    classes   = "object_detection_classes_coco.txt"
    graph     = "frozen_inference_graph.pb"
    base_confidence     = 0.6
    classes_to_detect   = ["person", "car"]
    mask_threshold      = 0.3
    model = obj.TensorflowModel(modelname, proto, graph, classes,
                                base_confidence, classes_to_detect,
                                mask_threshold=mask_threshold)
    # Feature Extractor
    featureX = ftx.FeatureExtraction()

    # Loading Image
    image = cv2.imread("image.png")

    # Detections
    detections = model.detect(image, "Test Image")

    # Kepoints and descriptors
    print("\n")
    for itr in range(len(detections)):
        detect = detections[itr]
        print("Getting features for ", detect["label"])
        detections[itr]["kp"], detections[itr]["desc"] = featureX.get_keypoints_descriptors(image, detect["bbox"], detect["mask"])
        # Draw keypoints
        image = ImageUtils.draw_keypoints(image, detections[itr]["kp"])
        # Draw bbox and label on image
        image = ImageUtils.draw_detections(image, detect["label"], detect["confidence"], detect["bbox"], detect["mask"])
    
    # # Show outputs
    time.sleep(0.1)
    cv2.imshow("Test", imutils.resize(image, width=1280, inter=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
    