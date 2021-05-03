# FeatureExtraction
OpenCV module for extracting keypoints and descriptors for image objects.
The module is written with MASK RCNN model in mind.
The feature extractor can input mask from rhe MASK RCNN model and use to find features within the mask.

### Usage
Please check the **exmaple.py** on how to use. 
You will need other modules (link below) to run this, 
- [ObjectDetection](https://github.com/tanayrastogi/ObjectDetection)
- [ImageUtils](https://github.com/tanayrastogi/ImageUtils)


### REFERENCE
- [OpenCV SIFT Intro](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)
- [OpenCV Official Docs](https://docs.opencv.org/master/d4/d5d/group__features2d__draw.html)
- [The Python Code article on SIFT](https://www.thepythoncode.com/article/sift-feature-extraction-using-opencv-in-python)
- [Feature matching from OpenCV](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)
- [Feature Matching from Datahacker](http://datahacker.rs/feature-matching-methods-comparison-in-opencv/)
