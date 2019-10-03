# Sylwester Prymula 2019/09

import uuid
import cv2
import glob
import csv
import os
import time
import datetime
import numpy

start = time.time()
datetime_object = datetime.datetime.now()
print(str(datetime_object))

''' Variables '''  # '\**\*.' (with subfolders) or '\*.' (without subfolders)
directory_path_1 = r'.\1\**\*.'
directory_path_2 = r'.\2\**\*.'
dir_feature_matching = r'C:\feature_matching'
file_extensions = ['jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'png']  # e.g.: ['jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'png']

# TODO - saving files in the "feature_matching" directory with UTF-8 characters

# TODO - Out of memory
#  Traceback (most recent call last):
#  File "G:/py/similar_images.py", line 208, in <module>
#    kp_2, desc_2 = sift.detectAndCompute(second_image, None)
#    cv2.error: OpenCV(3.4.2) C:\projects\opencv-python\opencv\modules\core\src\matrix.cpp:367: error: (-215:Assertion failed) u != 0 in function 'cv::Mat::create'

# TODO - Warning - '.png' - 'iCCP: known incorrect sRGB profile - libpng warning: iCCP: known incorrect sRGB profile'
# TODO - Warning - 'Premature end of JPEG file'
# TODO - Warning - 'Corrupt JPEG data: premature end of data segment'
# TODO - Warning - 'Corrupt JPEG data: 21339 extraneous bytes before marker 0xd0'

# TODO - reading a '.gif', '.pcd', '.psd', '.eps', '.fla' files

max_height_1 = 300
max_width_1 = 300
max_height_2 = 300
max_width_2 = 300

''' Delete a '.csv' file after checking whether it exists '''
file_to_delete = ['similar_images_1.csv', 'similar_images_2.csv', 'similar_images_3.csv']
for file_to_delete_1 in file_to_delete:
    if os.path.isfile(file_to_delete_1):
        os.remove(file_to_delete_1)
    else:  # If the file to be deleted does not exist
        print("Warning: %s file not found" % file_to_delete_1)

''' Delete files in folder: 'feature_matching' '''
files_in_folder = glob.glob(dir_feature_matching + '\\*')
for files_in_folder_1 in files_in_folder:
    os.remove(files_in_folder_1)

''' Prepare '''
all_images_to_compare_1 = []
titles_1 = []
all_images_to_compare_2 = []
titles_2 = []

list_of_files_1 = []
list_of_files_2 = []

print('===')
print(' directory_path_1: ' + directory_path_1)
print(' directory_path_2: ' + directory_path_2)
print(' file_extensions: ' + str(file_extensions))
print(' max_height_1: ' + str(max_height_1), 'max_width_1: ' + str(max_width_1))
print(' max_height_2: ' + str(max_height_2), 'max_width_2: ' + str(max_width_2))
print('===')

''' Writing to the file '''
f = open('logs.txt', 'w+', encoding='utf8')
f.write(str(datetime_object) + '\n' + '===' + '\n'
        + ' directory_path_1: ' + directory_path_1 + '\n'
        + ' directory_path_2: ' + directory_path_2 + '\n'
        + ' file_extensions: ' + str(file_extensions) + '\n'
        + ' max_height_1: ' + str(max_height_1) + '\n'
        + ' max_width_1: ' + str(max_width_1) + '\n'
        + ' max_height_2: ' + str(max_height_2) + '\n'
        + ' max_width_2: ' + str(max_width_2) + '\n'
        + '===')
f.close()

''' Load all the images - Folder 1 '''
for extension_1 in file_extensions:
    list_of_files_1.extend(glob.iglob(directory_path_1 + extension_1, recursive=True))

for f_1 in list_of_files_1:
    ''' Using cv2.imdecode because cv2.imread don't read UTF-8 characters '''
    stream_1 = open(f_1, 'rb')
    bytes_1 = bytearray(stream_1.read())
    numpy_array_1 = numpy.asarray(bytes_1, dtype=numpy.uint8)
    if numpy_array_1.size == 0:
        print('*** Warning: ' + f_1 + ' - doesn\'t have height or width or it has 0 bytes size! Please check it out! ***')
        f = open('logs.txt', 'a', encoding='utf8')
        f.write('\n' + '*** Warning: ' + f_1 + ' - doesn\'t have height or width or it has 0 bytes size! Please check it out! ***')
        f.close()
        continue
    image_1 = cv2.imdecode(numpy_array_1, cv2.IMREAD_UNCHANGED)
    if image_1 is None:
        print('*** Warning: ' + f_1 + ' - Please check it out! ***')
        f = open('logs.txt', 'a', encoding='utf8')
        f.write('\n' + '*** Warning: ' + f_1 + ' - Please check it out! ***')
        f.close()
        continue
    height_1, width_1 = image_1.shape[:2]
    ''' Only shrink if img is bigger than required '''
    if max_height_1 < height_1 or max_width_1 < width_1:
        ''' Get scaling factor '''
        scaling_factor_1 = max_height_1 / float(height_1)
        if max_width_1 / float(width_1) < scaling_factor_1:
            scaling_factor1 = max_width_1 / float(width_1)
        ''' Resize image '''
        image_1 = cv2.resize(image_1, None, fx=scaling_factor_1, fy=scaling_factor_1, interpolation=cv2.INTER_AREA)

    titles_1.append(f_1)
    all_images_to_compare_1.append(image_1)

''' Load all the images - Folder 2 '''
for extension_2 in file_extensions:
    list_of_files_2.extend(glob.iglob(directory_path_2 + extension_2, recursive=True))

for f_2 in list_of_files_2:
    ''' Using cv2.imdecode because cv2.imread don't read UTF-8 characters '''
    stream_2 = open(f_2, 'rb')
    bytes_2 = bytearray(stream_2.read())
    numpy_array_2 = numpy.asarray(bytes_2, dtype=numpy.uint8)
    if numpy_array_2.size == 0:
        print('*** Warning: ' + f_2 + ' - doesn\'t have height or width or it has 0 bytes size! Please check it out! ***')
        f = open('logs.txt', 'a', encoding='utf8')
        f.write('\n' + '*** Warning: ' + f_2 + ' - doesn\'t have height or width or it has 0 bytes size! Please check it out! ***')
        f.close()
        continue
    image_2 = cv2.imdecode(numpy_array_2, cv2.IMREAD_UNCHANGED)
    if image_2 is None:
        print('*** Warning: ' + f_2 + ' - Please check it out! ***')
        f = open('logs.txt', 'a', encoding='utf8')
        f.write('\n' + '*** Warning: ' + f_2 + ' - Please check it out! ***')
        f.close()
        continue
    height_2, width_2 = image_2.shape[:2]
    ''' Only shrink if img is bigger than required '''
    if max_height_2 < height_2 or max_width_2 < width_2:
        ''' Get scaling factor '''
        scaling_factor_2 = max_height_2 / float(height_2)
        if max_width_2 / float(width_2) < scaling_factor_2:
            scaling_factor_2 = max_width_2 / float(width_2)
        ''' Resize image '''
        image_2 = cv2.resize(image_2, None, fx=scaling_factor_2, fy=scaling_factor_2, interpolation=cv2.INTER_AREA)

    titles_2.append(f_2)
    all_images_to_compare_2.append(image_2)

''' Unicode '''
print('Unicode character encoding:')
for my_file1 in list_of_files_1:
    try:
        my_file1.encode('cp1252', 'strict')
    except UnicodeEncodeError:
        print(my_file1)

for my_file2 in list_of_files_2:
    try:
        my_file2.encode('cp1252', 'strict')
    except UnicodeEncodeError:
        print(my_file2)

''' How many images will be compared '''
t_1 = len(titles_1)
t_2 = len(titles_2)
t_1_2 = t_1 * t_2
print('===')
print(' directory_path_1: ' + str(t_1))
print(' directory_path_2: ' + str(t_2))
print(' directory_path_1 * directory_path_2: ' + str(t_1_2))
print('===')

''' Writing to the file '''
f = open('logs.txt', 'a', encoding='utf8')
f.write('\n' + ' directory_path_1: ' + str(t_1) + '\n'
        + ' directory_path_2: ' + str(t_2) + '\n'
        + ' directory_path_1 * directory_path_2: ' + str(t_1_2) + '\n'
        + '===')
f.close()

''' SIFT and FLANN '''
sift = cv2.xfeatures2d.SIFT_create()
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

for first_image, title_1 in zip(all_images_to_compare_1, titles_1):
    kp_1, desc_1 = sift.detectAndCompute(first_image, None)

    for second_image, title_2 in zip(all_images_to_compare_2, titles_2):

        uuid_file = uuid.uuid4()

        # print("Title_1: " + title_1)
        # print("Title_2: " + title_2)

        kp_2, desc_2 = sift.detectAndCompute(second_image, None)

        ''' Check for similarities between the 2 images '''
        if desc_1 is not None and desc_2 is not None and len(kp_1) > 2 and len(kp_2) > 2:
            matches = flann.knnMatch(desc_1, desc_2, k=2)

            good_points = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            ''' Define how similar they are '''
            number_key_points = 0
            if len(kp_1) >= len(kp_2):
                number_key_points = len(kp_1)
            else:
                number_key_points = len(kp_2)

            percentage_similarity = (len(good_points) / number_key_points) * 100

            ''' Workaround - OpenCV usually strips the alpha layer when reading an image, 
            so cv2.drawMatches drawing black background in .png files
            - https://github.com/opencv/opencv/issues/13227 - 3.4.2.17 version has this bug '''
            if title_1.endswith('.png'):
                first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
            if title_2.endswith('.png'):
                second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2RGB)

            ''' Find file name '''
            title_1_split = title_1.split('\\')[-1]
            title_2_split = title_2.split('\\')[-1]

            ''' Write the feature matching to a .jpg file '''
            if percentage_similarity > 2 and len(good_points) > 5:
                result = cv2.drawMatches(first_image, kp_1, second_image, kp_2, good_points, None)
                # use always "+ '.jpg'" because without extension 'cv2.IMWRITE_JPEG_QUALITY' shows error
                cv2.imwrite(
                    os.path.join(dir_feature_matching, str(percentage_similarity) + '_gp_' + str(len(good_points)) + '_t1_' + title_1_split + '_t2_' + title_2_split + '_uuid_' + str(uuid_file) + '.jpg'),
                    result, [cv2.IMWRITE_JPEG_QUALITY, 90])

                ''' Write the titles to a .csv file '''
                ''' y - It was analized '''
                csv_data = [[title_1, title_2, dir_feature_matching + '\\' + str(percentage_similarity) + '_gp_' + str(len(good_points)) + '_t1_' + title_1_split + '_t2_' + title_2_split + '_uuid_' + str(uuid_file) + '.jpg', percentage_similarity, len(good_points), len(kp_1), len(kp_2), 'y']]
                with open('similar_images_1.csv', 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(csv_data)
                    csv_file.close()

                t_1_2 = t_1_2 - 1
                print(t_1_2)
            else:
                ''' Write the titles to a .csv file '''
                ''' y2 - It was analized '''
                csv_data = [[title_1, title_2, '', percentage_similarity, len(good_points), len(kp_1), len(kp_2), 'y2']]
                with open('similar_images_2.csv', 'a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(csv_data)
                    csv_file.close()

                t_1_2 = t_1_2 - 1
                print(t_1_2)
        else:
            ''' Write the titles to a .csv file '''
            ''' n - it was not analized, there were no common points '''
            csv_data = [[title_1, title_2, '', '0', '0', len(kp_1), len(kp_2), 'n']]
            with open('similar_images_3.csv', 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(csv_data)
                csv_file.close()

            t_1_2 = t_1_2 - 1
            print(t_1_2)

end = time.time()
datetime_object = datetime.datetime.now()
print(str(datetime_object))
print('Time taken for program: ', end - start)

f = open('logs.txt', 'a', encoding='utf8')
f.write('\n' + str(datetime_object) + '\n' + 'Time taken for program: ' + str(end - start) + '\n')
f.close()
