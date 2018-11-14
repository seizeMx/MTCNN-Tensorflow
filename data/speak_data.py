# Program To Read video
# and Extract Frames
import cv2
import time
import glob
import os
from random import shuffle
import numpy as np


def locate(filePath, frame_num, length_frame):
    vidObj = cv2.VideoCapture(filePath)

    if frame_num > length_frame:
        frame_num = length_frame
    if frame_num <= 0:
        return vidObj, 1

    success = 1
    current_frame_num = 0
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        current_frame_num += 1

        # Saves the frames with frame-count
        if success:
            if current_frame_num == frame_num:
                break

    return vidObj, current_frame_num


# Function to extract frames
def FrameCapture(filePath, targetPath, strides=1, synthesisNum=25, func=2, size=(32, 32)):
    filePathList = filePath.split("\\")
    fileName = filePathList[-1]
    basePathSource = filePath.replace("\\" + fileName, "")
    fileName = basePathSource.split("\\")[-1] + "_" + fileName

    file_name_list = {'0': [], '1': []}

    targetPath = os.path.join(targetPath, filePathList[-3], filePathList[-2])
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)

    if not os.path.exists(os.path.join(targetPath, "1")):
        os.makedirs(os.path.join(targetPath, "1"))
    if not os.path.exists(os.path.join(targetPath, "0")):
        os.makedirs(os.path.join(targetPath, "0"))

    # Path to video file
    vidObj = cv2.VideoCapture(filePath)
    h = int(vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(vidObj.get(cv2.CAP_PROP_FRAME_WIDTH))
    length_frame = vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vidObj.get(cv2.CAP_PROP_FPS)

    if func == 1:
        wait_ms_scalar = int(1 / fps * 1000 / 1)
        speed = 0.5
        old_speed = 0.5
        # wait_ms = 1000
        delta = 25 * 3
        # Used as counter variable
        frame_num = 0
        silence_frames = []
        speak_frames = []
        start_point = 0

        # checks whether frames were extracted
        success = 1
        startTime = time.ctime()

        while success:
            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()
            frame_num += 1

            # Saves the frames with frame-count
            if success:
                cv2.putText(image, "F " + str(frame_num) + "/" + str(length_frame) + " S " + str(
                    start_point) + " Speed " + str(
                    speed) + " old_speed " + str(old_speed), (80, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, "frames " + str(silence_frames), (80, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0),
                            1)
                getAnnotations(basePathSource, frame_num, func=1,
                               func_ref=lambda annotation: cv2.circle(image, annotation, 1, (0, 255, 0)))

                cv2.imshow('image', image)
                keycode = cv2.waitKey(int(wait_ms_scalar * speed)) & 0xFF

                # flow control
                if keycode == ord('f'):
                    frame_num = frame_num + int(delta / (old_speed if speed == 0 else speed))
                    vidObj, frame_num = locate(filePath, frame_num, length_frame)
                    continue
                if keycode == ord('b'):
                    frame_num = frame_num - int(delta / (old_speed if speed == 0 else speed))
                    vidObj, frame_num = locate(filePath, frame_num, length_frame)
                    continue
                if keycode == ord('R'):
                    vidObj = cv2.VideoCapture(filePath)
                    frame_num = 0
                    continue
                if keycode == ord('s'):
                    start_point = frame_num
                if keycode == ord('e'):
                    if start_point != 99999:
                        silence_frames.append((start_point, frame_num))
                        start_point = 99999
                if keycode == ord('C'):
                    length = len(silence_frames)
                    if length > 0:
                        s, e = silence_frames.pop(length - 1)
                        start_point = s
                if keycode == ord(' '):
                    if speed == 0:
                        speed = old_speed
                    else:
                        old_speed = speed
                        speed = 0
                if keycode == ord('-'):
                    if speed == 0:
                        old_speed = old_speed + 0.1
                    else:
                        speed = speed + 0.1
                if keycode == ord('+'):
                    if speed - 0.1 > 0.01:
                        speed = speed - 0.1
                if keycode == 27:
                    break
            else:
                keycode = cv2.waitKey(int(wait_ms_scalar * speed)) & 0xFF
                while keycode != ord('P'):
                    if keycode == ord('b'):
                        frame_num = frame_num - int(delta / (old_speed if speed == 0 else speed))
                        vidObj, frame_num = locate(filePath, frame_num, length_frame)
                        success = 1
                        break
                    if keycode == ord('e'):
                        if start_point != 99999:
                            silence_frames.append((start_point, frame_num))
                            start_point = 99999
                    if keycode == ord('R'):
                        vidObj = cv2.VideoCapture(filePath)
                        success = 1
                        frame_num = 0
                        break

                    keycode = cv2.waitKey(int(wait_ms_scalar * speed)) & 0xFF

        mark_to_file(basePathSource, frame_num, silence_frames)

        endTime = time.ctime()
        print("{} start {}, end {}".format(startTime, endTime, filePath))
    else:

        speak_list, silence_list = getLabled(basePathSource)

        # Used as counter variable
        frame_num = 0

        # checks whether frames were extracted
        success = 1

        while success:
            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()
            frame_num += 1

            # Saves the frames with frame-count
            if success:
                if str(frame_num) in speak_list or frame_num in speak_list:
                    save_path = "1"
                    file_name_list['1'].append(targetPath + "\\" + save_path + "\\" + ("%06d.jpg" % frame_num) + "\n")
                else:
                    save_path = "0"
                    file_name_list['0'].append(targetPath + "\\" + save_path + "\\" + ("%06d.jpg" % frame_num) + "\n")

                if False:
                    annotations = getAnnotations(basePathSource, frame_num)
                    image = frameCropResize(annotations, image, size)
                    cv2.imwrite(os.path.join(targetPath, save_path, "%06d.jpg" % frame_num), image)

                # if frame_num % synthesisNum == 0:
                #     groupPic.append(image)
                #     image = np.concatenate(groupPic)
                #     cv2.imwrite(os.path.join(targetPath, "frame_con_%06d.jpg" % frame_num), image)
                #     groupPic = []

        # if len(groupPic) > 0:
        #     image = np.concatenate(groupPic)
        #     cv2.imwrite(os.path.join(targetPath, "frame_con_%06d.jpg" % frame_num), image)

        print("{} speak_list {}, silence_list {}".format(filePathList[-2], len(speak_list), len(silence_list)))

        # num_bottleneck = min(len(file_name_list['0']), len(file_name_list['1']))
        # shuffle(file_name_list['0'])
        # shuffle(file_name_list['1'])
        # return file_name_list['0'][:num_bottleneck], file_name_list['1'][:num_bottleneck]

        return file_name_list['0'], file_name_list['1']


def mark_to_file(basePathSource, frame_num, silence_frames):
    speak_frames = []
    if len(silence_frames) > 0:
        _s, _e = zip(*silence_frames)
        s_l = sorted(_s + _e)

        flag_temp = True
        i = 0
        while flag_temp:
            if i == 0:
                if s_l[i] > 1:
                    speak_frames.append((1, s_l[i] - 1))
                i = i + 1
            else:
                if i + 1 == len(s_l):
                    if s_l[i] < frame_num:
                        speak_frames.append((s_l[i] + 1, frame_num))
                    break
                speak_frames.append((s_l[i] + 1, s_l[i + 1] - 1))
                i = i + 2

    else:
        speak_frames.append((1, frame_num))

    # frame_set_silence = set()
    # frame_set_all = {x for x in range(1, frame_num)}
    # for s, e in silence_frames:
    #     frame_set_silence.update({x for x in range(s, e)})
    # frame_set_speak = frame_set_all.difference(frame_set_silence)

    # list_frames = sorted(frame_set_speak)
    # if len(list_frames) > 0:
    #     last_frame = list_frames.pop(0)
    #     for frame in list_frames:
    #         if frame != last_frame + 1:
    #             speak_frames.append((last_frame, frame))
    #
    # f = open(basePathSource + "//classify_frames.txt", "w")
    # f.write("speak" + (str(sorted(frame_set_speak))) + "\n")
    # f.write("silence" + (str(sorted(frame_set_silence))))

    f = open(basePathSource + "//classify_frames.txt", "w")
    f.write("speak" + (str(speak_frames)) + "\n")
    f.write("silence" + (str(silence_frames)))
    f.close()


def frameCropResize(annotations, image, size):
    xmax, xmin, ymax, ymin = findZone(annotations)
    # cv2.circle(image, (11,700), 5, (255,0,255))
    # extend y to 2*y
    ymax = ymax + (ymax - ymin)
    image = image[ymin:ymax, xmin:xmax, :]
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    return image


def getAnnotations(basePathSource, frame, func=3, func_ref=lambda x: x):
    result = []
    with open(basePathSource + (r"\annot\%06d.pts" % frame)) as pts:
        annotations = pts.readlines()
    annotations = annotations[3:71]
    if func == 3:
        annotations = [annotations[30], annotations[48], annotations[54]]
    for annotation in annotations:
        temp = tuple(map(int, (map(float, annotation.split(" ")))))
        func_ref(temp)
        result.append(temp)

    return result


def expand_range2seq(pair_str_list, splitor):
    result = []
    for pair_str in pair_str_list:
        range_t = pair_str.split(splitor)
        for i in range(int(range_t[0]), int(range_t[1]) + 1):
            result.append(i)
    return result


def getLabled(basePathSource):
    speak_list = []
    silence_list = []
    with open(basePathSource + r"\classify_frames.txt") as labeledFile:
        labels = labeledFile.readlines()

    for label in labels:
        if "(" in label:
            if label.startswith("speak"):
                speak_list = expand_range2seq(
                    label.replace("speak[(", "").replace(")]", "").replace("\n", "").split("), ("), ", ")
            elif label.startswith("silence"):
                silence_list = expand_range2seq(
                    label.replace("silence[(", "").replace(")]", "").replace("\n", "").split("), ("), ", ")
        else:
            if label.startswith("speak"):
                if len(label.replace("speak[", "").replace("]", "")) == 0:
                    speak_list = []
                else:
                    speak_list = label.replace("speak[", "").replace("]", "").replace("\n", "").split(", ")
            elif label.startswith("silence"):
                if len(label.replace("silence[", "").replace("]", "")) == 0:
                    silence_list = []
                else:
                    silence_list = label.replace("silence[", "").replace("]", "").replace("\n", "").split(", ")

    return speak_list, silence_list


def findZone(annotations):
    xmin = 9999
    ymin = 9999
    xmax = 0
    ymax = 0
    for annotation in annotations:
        xmin = min(annotation[0], xmin)
        ymin = min(annotation[1], ymin)
        xmax = max(annotation[0], xmax)
        ymax = max(annotation[1], ymax)

    if ymin >= ymax:
        ymin, ymax = ymax, ymin
    if (ymax - ymin) / (xmax - xmin) < 1 / 4:
        ymax = ymax + (xmax - xmin) / 8
        ymin = ymin - (xmax - xmin) / 8
        # cv2.circle(image, tuple(map(int, (map(float, annotation.split(" "))))), 1, (0,255,0))
    return int(xmax), int(xmin), int(ymax), int(ymin)


# Driver Code
if __name__ == '__main__':
    # Calling the function
    # mark, crop, resize, synthesis
    targetPath = r"d:\datasetConvert"

    pathV = r"D:\dataset\300VW_Dataset_2015_12_14"
    nameV = "*.avi"
    flag = 1

    file_0 = []
    file_1 = []
    f_train_0 = open(os.path.join(targetPath, "300VW_Dataset_2015_12_14\\0_train.txt"), "w")
    f_train_1 = open(os.path.join(targetPath, "300VW_Dataset_2015_12_14\\1_train.txt"), "w")
    f_test_0 = open(os.path.join(targetPath, "300VW_Dataset_2015_12_14\\0_test.txt"), "w")
    f_test_1 = open(os.path.join(targetPath, "300VW_Dataset_2015_12_14\\1_test.txt"), "w")

    for imagename in glob.glob(os.path.join(pathV + r"\*", nameV)):
        if (1 == 2 and imagename == 'D:\\dataset\\300VW_Dataset_2015_12_14\\057\\vid.avi'):
            flag = 1

        if (1 == 1 and imagename == 'D:\\dataset\\300VW_Dataset_2015_12_14\\126\\vid.avi'):
            flag = 0
            break

        # try:
        if flag == 1:
            file_0_part, file_1_part = FrameCapture(imagename, targetPath,
                                                    func=2)  # (r"D:\dataset\300VW_Dataset_2015_12_14\001\vid.avi")
            file_0.extend(file_0_part)
            file_1.extend(file_1_part)

    num = min(len(file_0), len(file_1))

    f_train_0.writelines(file_0[:int(0.7 * num)])
    f_train_1.writelines(file_1[:int(0.7 * num)])
    f_test_0.writelines(file_0[int(0.7 * num):num])
    f_test_1.writelines(file_1[int(0.7 * num):num])

    print('summary_sp {}, summary_sl {}, ALL {}'.format(len(file_1), len(file_0), len(file_1) + len(file_0)))
    f_train_0.close()
    f_train_1.close()
    f_test_0.close()
    f_test_1.close()

    # except:
