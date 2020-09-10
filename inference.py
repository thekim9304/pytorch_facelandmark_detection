import cv2
import time
import numpy as np

import torch

from face_detection_model.mobilenetv1 import MobileNetV1
from face_detection_model.ssd import SSD, Predictor
from face_detection_model.utils import box_utils
from tcdcn import TCDCN
from custom_mobilenet_v1 import MobileNetV1 as mv1

def infer_frame():
    model_landmark = TCDCN((40, 40, 3), 70).cuda().eval()
    state_landmark = torch.load('E:/models/f_landmark-tcdcn-52-0.0004.pth')
    model_landmark.load_state_dict(state_landmark['model_state_dict'])

    img = cv2.imread('./dataset/images/51_Dresses_wearingdress_51_377_face0_0.jpg')

    print(img.shape)
    x = cv2.resize(img.astype(np.float32), (40, 40)) / 255
    print(x.shape)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
    print(x.shape)
    landmark = model_landmark(x)
    print(landmark.shape)

    for ii in range(0, landmark.shape[1], 2):
        img = cv2.circle(img,
                          (int(landmark[0][ii] * img.shape[1]),
                           int(landmark[0][ii + 1] * img.shape[0])),
                          2, (0, 0, 255), -1)

    cv2.imshow('annotated', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def infer_cam():
    model = SSD(2, MobileNetV1(2), is_training=False)
    state = torch.load('C:/Users/th_k9/Desktop/git/pytorch_ssd/models/ssd-mobilev1-face-2134_0.0192.pth')
    model.load_state_dict(state['model_state_dict'])

    add_region = 10
    model_name = 'mobilev1'
    # model_name = 'tcdcn'
    land_img_size = (40, 40, 3)
    # model_landmark = TCDCN(land_img_size, 70).cuda().eval()
    model_landmark = mv1(land_img_size, 70).cuda().eval()
    state_landmark = torch.load(f'E:/models/f_landmark-{model_name}-{land_img_size}.pth')
    model_landmark.load_state_dict(state_landmark['model_state_dict'])

    predictor = Predictor(model, 300)

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('E:/DB_FaceLandmark/300VW/001/vid.avi')

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('C:/Users/th_k9/Desktop/t.avi', fourcc, 30.0, (940, 480))

    bbox_low_pass_filter = 5
    landmark_low_pass_filter = 4

    frm_cnt = 1
    fps = 0
    fps_res = 0
    face_detect = False
    land_detect = False
    cnt = 0
    while True:
        ret, img = cap.read()

        if ret:
            img = cv2.resize(img, (640, 480))
            prevTime = time.time()
            boxes, labels, probs = predictor.predict(img, 1, 0.5)
            curTime = time.time()
            sec1 = curTime - prevTime

            for i in range(boxes.size(0)):
                box = boxes[i, :]
                label = f"Face: {probs[i]:.2f}"

                x1, x2, y1, y2 = int(box[0].item() - add_region), int(box[2].item() + add_region), int(box[1].item()), int(box[3].item()+add_region)

                if face_detect:
                    if abs(prev_x1 - x1) < bbox_low_pass_filter:
                        x1 = prev_x1
                    else:
                        prev_x1 = x1
                    if abs(prev_x2 - x2) < bbox_low_pass_filter:
                        x2 = prev_x2
                    else:
                        prev_x2 = x2
                    if abs(prev_y1 - y1) < bbox_low_pass_filter:
                        y1 = prev_y1
                    else:
                        prev_y1 = y1
                    if abs(prev_y2 - y2) < bbox_low_pass_filter:
                        y2 = prev_y2
                    else:
                        prev_y2 = y2
                else:
                    face_detect = True
                    prev_x1 = x1
                    prev_x2 = x2
                    prev_y1 = y1
                    prev_y2 = y2
                # face = img[y1:y2, x1:x2].copy()
                # print(f'x1: {x1}, x2: {x2}, y1: {y1}, y2: {y2}')
                face = img[y1:y2, x1:x2].copy()
                # cv2.imwrite(f'./myface/th_{cnt}.jpg', face)
                cnt+=1
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 128), 4)
                cv2.putText(img, label,
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,  # font scale
                            (128, 0, 255),
                            2)  # line type

            white_numpy = np.full((480, 300, 3), 255, dtype=np.uint8)
            sec2 = 0

            if boxes.size(0):
                '''
                ===landmark detection
                '''
                x = cv2.resize(face.astype(np.float32), land_img_size[:2])

                if land_img_size[-1] == 1:
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) / 255
                    x = torch.from_numpy(x).unsqueeze(-1).permute(2, 0, 1).unsqueeze(0).cuda()
                else:
                    x = torch.from_numpy(x/255).permute(2, 0, 1).unsqueeze(0).cuda()

                prevTime = time.time()
                landmark = model_landmark(x)
                curTime = time.time()
                sec2 = curTime - prevTime

                # right eye : 38-54
                # left eye : 56-landmark.shape[0](70)
                # nose : 20-38
                face = cv2.resize(face, (300, 300))
                landmark = np.array(landmark.tolist()[0])
                landmark[0::2] = landmark[0::2] * face.shape[1]
                landmark[1::2] = landmark[1::2] * face.shape[0]

                if land_detect:
                    idx = 0
                    for land, prev_land in zip(landmark, prev_landmark):
                        # print(abs(land - prev_land))
                        if abs(land - prev_land) < landmark_low_pass_filter:
                            landmark[idx] = prev_landmark[idx]
                        else:
                            prev_landmark[idx] = landmark[idx]

                        idx += 1
                else:
                    land_detect = True
                    prev_landmark = landmark

                for ii in range(38, 55, 2):
                    face = cv2.circle(face,
                                      (int(landmark[ii]),
                                       int(landmark[ii+1])),
                                      1, (255, 255, 255), -1)
                for ii in range(56, landmark.shape[0], 2):
                    face = cv2.circle(face,
                                      (int(landmark[ii]),
                                       int(landmark[ii+1])),
                                      1, (255, 255, 255), -1)
                for ii in range(20, 38, 2):
                    face = cv2.circle(face,
                                      (int(landmark[ii]),
                                       int(landmark[ii+1])),
                                      1, (255, 255, 255), -1)

                # if land_detect:
                #
                # else:
                #     land_detect = True
                #     prev_landmark = landmark[0]


                # for ii in range(38, 55, 2):
                #     face = cv2.circle(face,
                #                       (int(landmark[0][ii]*face.shape[1]),
                #                        int(landmark[0][ii+1]*face.shape[0])),
                #                       1, (255, 255, 255), -1)
                # for ii in range(56, landmark.shape[1], 2):
                #     face = cv2.circle(face,
                #                       (int(landmark[0][ii]*face.shape[1]),
                #                        int(landmark[0][ii+1]*face.shape[0])),
                #                       1, (255, 255, 255), -1)
                # for ii in range(20, 38, 2):
                #     face = cv2.circle(face,
                #                       (int(landmark[0][ii]*face.shape[1]),
                #                        int(landmark[0][ii+1]*face.shape[0])),
                #                       1, (255, 255, 255), -1)
                '''
                '''
                white_numpy[:300, :] = face

            sec_sum = sec1 + sec2
            if (1/sec_sum) > 30:
                print(sec1)
                fps += 1/sec_sum
                fps_res = fps / frm_cnt
                frm_cnt += 1

            cv2.putText(img, f'{fps_res:.2f}',
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (0, 0, 255),
                        2)  # line type

            concat_img = cv2.hconcat([white_numpy, img])
            cv2.imshow('annotated', concat_img)
            # if (1/sec_sum) > 50:
                # out.write(concat_img)

            if cv2.waitKey(10) == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    # out.release()

if __name__=='__main__':
    # infer_frame()
    infer_cam()