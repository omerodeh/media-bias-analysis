import os
import json
from deepface import DeepFace
from deepface.extendedmodels import Gender
import face_recognition
import cv2
from collections import Counter


# The following lines will select your first GPU. Requires CUDA and tensorflow-gpu correctly configured
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def find_faces_in_image(path_to_image):
    image = face_recognition.load_image_file(path_to_image)
    face_locations = face_recognition.face_locations(image, model="hog")  # use model cnn with gpu if set up
    # face_locations = face_recognition.face_locations(image)
    filename = get_filename_from_path(strip_extension(path_to_image))
    # have to reorder the face locations to left_x, top_y, right_x, bottom_y to match openCV standard
    x, y = path_to_image, [(loc[3], loc[0], loc[1], loc[2]) for loc in face_locations]
    # yield y
    return y
    # yield x, y


def get_filename_from_path(path):
    return os.path.basename(path)


def strip_extension(file_name):
    return os.path.splitext(file_name)[0]


def write_crop_images_from_bounding_boxes(path_to_image, bounding_boxes, output_crop_dir):
    try:
        os.mkdir(output_crop_dir)
    except:
        pass
    box_count = 0
    image = cv2.imread(path_to_image)
    for bounding_box in bounding_boxes:
        left_x, top_y, right_x, bottom_y = bounding_box
        crop = image[top_y:bottom_y, left_x:right_x]
        cropped_image_name = f"{strip_extension(get_filename_from_path(path_to_image))}_{box_count}.jpg"
        full_output_path = os.path.join(output_crop_dir, cropped_image_name)
        if crop.size == 0:
            print(f"NO IMAGE for {cropped_image_name} for bbox {bounding_box}")
            continue
        cv2.imwrite(full_output_path, crop)
        box_count += 1
        yield full_output_path


def categorize_crops_by_gender(crops):
    """ :param path_to_image must be the path to a crop of a single face"""
    genders = {"Man": [], "Woman": [], "other": []}
    for crop in crops:
        gender_in_image = get_gender(crop)
        if gender_in_image in genders:
            genders[gender_in_image].append(strip_extension(get_filename_from_path(crop)))
        else:
            print("Unclassified gender")
            genders["other"].append(strip_extension(get_filename_from_path(crop)))
    return genders


def get_files_in_directory(directory):
    dir = os.fsencode(directory)
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        # TODO check that file is an actual image '.jpg' or '.png'
        yield directory + filename


def mp4_to_frames(input_video_file, output_directory='/tmp/frames', percent_to_capture=100):
    try:
        os.mkdir(output_directory)
    except:
        pass
    if (percent_to_capture == 0):
        return
    divisor = 100 / percent_to_capture
    vidcap = cv2.VideoCapture(input_video_file)
    success, image = vidcap.read()
    count = 0
    while success:
        if (count % divisor == 0):
            write_location = os.path.join(output_directory, f'frame{count}.jpg')
            cv2.imwrite(write_location, image)  # save frame as JPEG file
            yield write_location
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1


def get_gender(path_to_image):
    demography = DeepFace.analyze(path_to_image, actions=['gender'], models=models, enforce_detection=False)
    gender = demography.get("gender")
    return gender


if __name__ == '__main__':
    models = {}
    models["gender"] = Gender.loadModel()
    #images = ['/home/kareem/Programming/pythonProject/media_analysis/kids.jpg', 'adults.jpg']
    import time
    start = time.process_time()
    images = list(mp4_to_frames('sample.mkv', '/tmp/frames'))
    print(f'Time to read frames: {time.process_time() - start}')

    list_of_boxes = (find_faces_in_image(image) for image in images)
    paths_to_crops = (write_crop_images_from_bounding_boxes(image, face_boxes, '/tmp/crops/') for image, face_boxes in zip(images, list_of_boxes))
    image_genders_split = ([get_gender(crop) for crop in crops_for_one_picture] for crops_for_one_picture in paths_to_crops)
    #list_of_boxes = map(find_faces_in_image, images)
    #paths_to_crops = map(lambda x, y: write_crop_images_from_bounding_boxes(x, y, 'crops/'), images, list_of_boxes)
    #image_genders_split = map(lambda crops_for_one_picture: [get_gender(crop) for crop in crops_for_one_picture],
    #                          paths_to_crops)

    #final_list = [{x: genders.count(x) for x in genders} for genders in image_genders_split]
    #final_list = [dict(Counter(genders).items()) for genders in image_genders_split]
    per_frame_gender_counts = [dict(Counter(genders).items()) for genders in image_genders_split]
    final_split = {gender:sum(split[gender] for split in per_frame_gender_counts if gender in split) for gender in ['Man', 'Woman', 'Other']}
    for gender, count in final_split.items():
        print(gender, count)
    print(time.process_time() - start)
    print(f'Percent of faces that are female: {gender["Woman"] / sum(gender.values()):.2f}')
