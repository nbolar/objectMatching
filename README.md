# objectMatching
In the case of object matching in videos, an algorithm was used to detect a specific object in the test image/video based on finding points that connect it with the reference/training image. It is able to detect objects to a decent accuracy while there is a change in its orientation and/or scale. It becomes more accurate if there are less repeating patterns that allow for unique features to be matched and as a result detected and recognised.


MATLAB’s Computer Vision Toolbox is used extensively throughout this project. The feature detection, description and matching functions of this toolbox were used. MATLAB uses SURF key- points to generate the detectors using detectSURFFeatures(). Since this function only works with Greyscale images, the training images needed to converted to greyscale images using the rgb2gray() function. Once that was complete, the extractFeatures() function was used to get valid SURF Features. The ‘MetricThreshold’ parameter was used in-order to control the number of SURF points generated for each training image. It is a non-negative scalar which specifies a threshold for selecting the strongest features. Decreasing it returns more blobs.


Due to the fact that the folder is significantly larger than what GitHub allows, I have hosted the folder on Dropbox and the link is https://www.dropbox.com/sh/kirhw4adfs5l1po/AABxnWfK_aRWQ_QP7uDssc9ha?dl=0
