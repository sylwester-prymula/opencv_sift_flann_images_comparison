# opencv_sift_flann_images_comparison

Introduction
--------------------------
Comparing image files in two folders.

SIFT - Scale-Invariant Feature Transform\
https://en.wikipedia.org/wiki/Scale-invariant_feature_transform \
https://patents.google.com/patent/US6711293B1/

FLANN - Fast Library for Approximate Nearest Neighbors\
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

Alternatives:
- SURF (Speeded up robust features): \
https://en.wikipedia.org/wiki/Speeded_up_robust_features\
- ORB (Oriented FAST and Rotated BRIEF): \
http://www.willowgarage.com/sites/default/files/orb_final.pdf

My code was based on: \
https://pysource.com/category/tutorials/images-comparison/

Installation and requirements
--------------------------
1. https://www.python.org/ - Python - Install the `AMD64/EM64T/x64` version
2. `pip install opencv-contrib-python==3.4.2.17` \
it is the latest version of OpenCV with the SIFT algorithm (newer ones don't have it)

Example of use
--------------------------
1. We indicate what graphics we want to compare with other graphics, e.g .:

![Comparison - example 1](https://github.com/sylwester-prymula/opencv_sift_flann_images_comparison/blob/master/example/1.jpg)

2. In the folder where we have many different pictures are found:

![Comparison - example 2](https://github.com/sylwester-prymula/opencv_sift_flann_images_comparison/blob/master/example/2.jpg)

Feature matching:

![Feature matching](https://github.com/sylwester-prymula/opencv_sift_flann_images_comparison/blob/master/example/41.44144144144144_gp_92_t1_1.jpg_t2_2.jpg_uuid_b017597a-454e-43f8-9161-160ff48bda76.jpg)

The probability that these two images are the same: `41.4414414414414` \
Number of key common points: `92`

Launching
--------------------------
In the code, indicate the catalogs to compare: \
`directory_path_1 = r'C:\similar\dir_1\**\*.'` \
`directory_path_2 = r'C:\similar\dir_2\**\*.'`

`'\ ** \ *.'` (files also in subdirectories)
or `'\*.'` (without searching subdirectories, only root folder)

Catalog where the graphical representation of the comparison will be saved: \
`dir_feature_matching = r'C:\feature_matching'`

Currently, you can compare files with the extension: \
`file_extensions = ['jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'png']`

The results are saved in files: \
`['similar_images_1.csv', 'similar_images_2.csv', 'similar_images_3.csv']`

logs.txt - basic information about running the comparison process

FAQ
--------------------------
- If the file to compare is too large for the computer, the out of memory message will pop up: 

    `Traceback (most recent call last):` \
    `File "G:/py/similar_images.py", line 208, in <module>` \
    `kp_2, desc_2 = sift.detectAndCompute(second_image, None)` \
    `cv2.error: OpenCV(3.4.2) C:\projects\opencv-python\opencv\modules\core\src\matrix.cpp:367: error: (-215:Assertion failed) u != 0 in function 'cv::Mat::create'`

- Warnings that may appear but do not cause the program to stop (incorrect RGB profiles or broken (incomplete) files): 

    `Warning - '.png' - 'iCCP: known incorrect sRGB profile - libpng warning: iCCP: known incorrect sRGB profile'` \
    `Warning - 'Premature end of JPEG file'` \
    `Warning - 'Corrupt JPEG data: premature end of data segment'` \
    `Warning - 'Corrupt JPEG data: 21339 extraneous bytes before marker 0xd0'` 


