# Python Libraries
import cv2
import numpy as np 

class FeatureExtraction:
    def __init__(self, detector_type="SIFT"):
        # Type of feature detector
        if detector_type.lower() == "sift":
            self.__featureDetector = cv2.SIFT_create()
        elif detector_type.lower() == "surf":
            self.__featureDetector = cv2.xfeatures2d.SURF_create()
        print("\n[FeaT] Setting feature detector with {}".format(detector_type.upper()))
        
    def get_keypoints_descriptors(self, image, bbox, mask=None):
        """
        Function to calculate the kepypoints and descritors for the object in the image. 
        
        INPUT
            image(numpy.ndarray):   Numpy image. This is the complete image with object to be detected.
            bbbox(numpy.array)  :   Array of length 4. Contains start and end of object in the image. 
            mask(numpy.array)   :   2D array of same size as the bbox. This basically highlights where the mask on the object is.

        RETURN
            Return the keypoint and desc for the object whose bbox is given in the function.
        """
        print("[FeaT] Generating features from the image...", end=" ")
        # Box dimensions
        (startX, startY, endX, endY) = bbox
        # Create mask
        visMask = np.zeros(image.shape[:2], dtype="uint8")
        if mask is not None:
            visMask[startY:endY, startX:endX] = mask.astype("uint8")
        else:
            cv2.rectangle(visMask, (startX, startY), (endX, endY), 255, -1) 

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        kp, desc = self.__featureDetector.detectAndCompute(gray, mask=visMask)
        print("Done!")
        return kp, desc

