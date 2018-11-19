# coding:utf-8
import sys

sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np

from keras.models import load_model

from data.speak_data import frameCropResize, getAnnotations
import os
import glob

# x1, y1, x2, y2, score, id, thresh_frame_counter, related_new_id
def re_id(identity_box, boxes_c):
    boxes_len = len(boxes_c)
    if boxes_len == 0:
        if len(identity_box) > 0:
            identity_box = identity_box[identity_box[:, -2] > 0]
            identity_box[:, -2] = identity_box[:, -2] - 1
        return identity_box

    temp = np.arange(boxes_len).reshape((-1, 1))
    temp_boxes = np.hstack((boxes_c, temp, temp, temp))
    if len(identity_box) == 0:
        temp_boxes[:, -2] = thresh_frame
        temp_boxes[:, -1] = temp_boxes[:, -3]
        identity_box = temp_boxes
        return identity_box
    else:
        id_start = max(identity_box[:, -3]) + 1

        identity_box = identity_box[identity_box[:, -2] > 0]
        # if can't re-id, then reduce effect counter. only thresh_frame_counter == thresh_frame to show pic
        identity_box[:, -2] = identity_box[:, -2] - 1
        if len(identity_box) == 0:
            return []
        id_has_relation = []

        for pos in range(len(temp_boxes)):
            x1 = temp_boxes[pos][0]
            y1 = temp_boxes[pos][1]
            x2 = temp_boxes[pos][2]
            y2 = temp_boxes[pos][3]

            vx1 = identity_box[:, 0]
            vy1 = identity_box[:, 1]
            vx2 = identity_box[:, 2]
            vy2 = identity_box[:, 3]

            area_v = (vx1 - vx2) * (vy1 - vy2)

            h = np.fmax(0, np.fmin(vx2, x2) - np.fmax(vx1, x1))
            w = np.fmin(vy2, y2) - np.fmax(vy1, y1)
            intersection = h * w
            iou = intersection / (area_v + (x1 - x2) * (y1 - y2) - intersection)
            max_i = np.argmax(iou)
            if iou[max_i] > thresh_iou:
                identity_box[max_i][0] = x1
                identity_box[max_i][1] = y1
                identity_box[max_i][2] = x2
                identity_box[max_i][3] = y2
                identity_box[max_i][-2] = thresh_frame
                identity_box[max_i][-1] = pos

                id_has_relation.append(pos)
            else:
                print()
        new_items = set(range(len(temp_boxes))).difference(set(id_has_relation))
        for i in new_items:
            temp_boxes[i]
            # new item id
            temp_boxes[i][-3] = id_start
            id_start += 1
            identity_box = np.vstack((identity_box, temp_boxes[i]))

        return identity_box


thresh_frame = 8
thresh_iou = 0.6


def main(saved_model, videopath, to_save=False):
    # speaker detect
    model_3dc = load_model(saved_model)

    test_mode = "onet"
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    # vis = True
    detectors = [None, None, None]
    prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet',
              '../data/MTCNN_model/ONet_landmark/ONet']
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    video_capture = cv2.VideoCapture(videopath)
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    length_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    codec = video_capture.get(cv2.CAP_PROP_FOURCC)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    source_file_name = videopath.split(os.sep)[-1]
    model_file_name = saved_model.split(os.sep)[-1]
    source_file_name_split = source_file_name.split('.')
    target_path = videopath.replace(source_file_name, source_file_name_split[0] + '-' + model_file_name + '.avi') # + source_file_name_split[1])
    print('target_path {}', target_path)
    if to_save:
        videoWriter = cv2.VideoWriter(target_path, fourcc, fps, (w, h))

    # video_capture.set(3, 340)
    # video_capture.set(4, 480)
    corpbbox = None
    frame_num = 0
    identity_box = []
    image_data_landmark = dict()
    speak_status_data = dict()

    while True:
        # fps = video_capture.get(cv2.CAP_PROP_FPS)
        t1 = cv2.getTickCount()
        ret, frame = video_capture.read()
        frame_num += 1
        # if frame_num < 466: #511 1459 466
        #     continue
        # if frame_num > 1700: # 470
        #     break

        print(frame_num)
        if ret:
            image = np.array(frame)
            boxes_c, landmarks = mtcnn_detector.detect(image)

            identity_box = re_id(identity_box, boxes_c)
            if len(identity_box) > 0:
                to_predict = identity_box[identity_box[:, -2] == thresh_frame]
                to_predict_ids = to_predict[:, -3]

                if len(landmarks) > 0:
                    # get prediction result
                    for i in range(len(to_predict)):
                        annotations = []
                        item = to_predict[i]
                        unify_id = int(item[-3])
                        new_id = int(item[-1])

                        if image_data_landmark.__contains__(unify_id):
                            img_array = image_data_landmark[unify_id]
                        else:
                            img_array = list()
                            image_data_landmark[unify_id] = img_array

                        landmarks_3_points = landmarks[new_id]

                        for j in range(2, len(landmarks_3_points) // 2):
                            annotations.append((int(landmarks_3_points[2 * j]), int(landmarks_3_points[2 * j + 1])))

                        img_arr = frameCropResize(annotations, frame, (32, 32))

                        # TODO just for test
                        # filePathList = videopath.split("\\")
                        # fileName = filePathList[-1]
                        # basePathSource = videopath.replace("\\" + fileName, "")
                        # annotations = getAnnotations(basePathSource, frame_num)
                        # img_arr = frameCropResize(annotations, frame, (32, 32))


                        # targetPath = os.path.join(r'D:\datasetConvert\test', str(int(unify_id)))
                        # if not os.path.exists(targetPath):
                        #     os.makedirs(targetPath)
                        # cv2.imwrite(os.path.join(targetPath, "%06d.jpg" % frame_num), img_arr)

                        r = (img_arr[:, :, 1] / 255.).astype(np.float32)
                        img_array.append(r)
                        # image_data_landmark[unify_id] = np.dstack(img_array, r)

                        if len(img_array) == 16:
                            data = np.dstack(img_array)
                            prediction = model_3dc.predict(np.expand_dims(data, axis=0))
                            if np.argmax(prediction[0], axis=0) == 1:
                                speak_status_data[unify_id] = 1
                            img_array.pop(0)

            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t

            #to_show = identity_box[identity_box[:, -2] == thresh_frame]

            for i in range(boxes_c.shape[0]):
                unify_id = -1
                if len(identity_box) > 0 and len(identity_box[identity_box[:, -1] == i]) > 0:
                    o_data = identity_box[identity_box[:, -1] == i]
                    unify_id = o_data[0, -3]
                    if unify_id in to_predict_ids and speak_status_data.get(unify_id) is not None and speak_status_data[unify_id] > 0:
                        speak_status_data[unify_id] = speak_status_data[unify_id] - 1
                        c = (0, 0, 255)
                    else:
                        c = (0, 255, 0)
                else:
                    c = (0, 255, 0)
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # if score > thresh:
                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), c, 2)
                cv2.putText(frame, ("id %04d" % int(unify_id)), (corpbbox[0], corpbbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0), 1)
                for j in range(len(landmarks[i]) // 2):
                    cv2.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 1, (0, 255, 0))
            cv2.putText(frame, '{}'.format(length_frame) + ":" + '{}'.format(frame_num), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            cv2.imshow("", frame)
            if to_save:
                videoWriter.write(frame)
            # for i in range(landmarks.shape[0]):
            #     for j in range(len(landmarks[i])//2):
            #         cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))
            # time end

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('device not find')
            break
    if to_save:
        videoWriter.release()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #saved_model_list = ['../data/3dc/v6-102-3d_in_c-images.001-0.433.hdf5']
    #saved_model_list = ['../data/3dc/v5-local-3d_in_c-images.002-0.094.hdf5']
    saved_model_list = ['..' + os.sep + 'data' + os.sep + '3dc' + os.sep + 'v6-102-3d_in_c-images.001-0.433.hdf5']
    #saved_model_list = sorted(glob.glob(os.path.join('..' + os.sep, 'data', '3dc', '*hdf5')))
    # ['../data/3dc/v4-3d_in_c-images.001-0.071.hdf5', '../data/3dc/v3_3d_in_c-images.003-0.030.hdf5',
    #                 '../data/3dc/v2_3d_in_c-images.002-0.040.hdf5', '../data/3dc/v1_3d_in_c-images.047-0.018.hdf5']
    #videopath = "./videoplayback.mp4"  # 466 1700
    videopath = 'D:' + os.sep + 'dataset' + os.sep + '300VW_Dataset_2015_12_14' + os.sep + '003' + os.sep + 'vid.avi'  # 466 1700
    # saved_model = '../data/3dc/v2_3d_in_c-images.002-0.040.hdf5'
    for saved_model in saved_model_list:
        main(saved_model, videopath, True)
