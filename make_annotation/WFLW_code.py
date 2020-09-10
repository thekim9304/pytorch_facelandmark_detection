# https://wywu.github.io/projects/LAB/WFLW.html

# show landmarks
import os
import cv2
import csv
import numpy as np

annotation_path = 'E:/DB_FaceLandmark/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train' \
                  '.txt '
image_path = 'E:/DB_FaceLandmark/WFLW/WFLW_images/'

save_path = '../dataset'

if not os.path.exists(save_path):
    os.mkdir(save_path)

fishing_landmarks = [66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151]

with open(annotation_path) as anno_txts:
    data = anno_txts.read()
annotations = data.split('\n')

f = open('../dataset/train.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['ImageID', 'landmarks'])
save_cnt = 1

for i, anno_str in enumerate(annotations):
    anno_list = list(anno_str.split(' '))

    landmarks_anno = anno_list[:196]
    landmarks_anno = np.array(landmarks_anno)[fishing_landmarks]
    landmarks_anno = list(landmarks_anno)

    bbox_anno = anno_list[196:200]
    img_anno = anno_list[-1]

    img_path = str(image_path + img_anno)

    # def make_xml(savepath, folder, filename, filepath, image_shape, bboxs, landmarks):
    folder = img_anno.split('/')[0]
    filename = img_anno.split('/')[1].split('.')[0]
    filepath = os.path.join(img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1])

    img = cv2.imread(img_path)
    image_shape = img.shape
    bboxs = np.expand_dims(bbox_anno, axis=0)
    # list to str
    landmarks = np.expand_dims([' '.join(landmarks_anno)], axis=0)

    for n_box, (landmark, bbox) in enumerate(zip(landmarks, bboxs)):
        bbox = bbox.astype(float).astype(int)
        landmark = np.array(landmark[0].split(' ')).astype(float)

        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img_path = os.path.join(save_path, 'images', f'{filename}_face{n_box}_{i}.jpg')

        cv2.imwrite(img_path, face)
        # print(f'Saved img {save_cnt} : {img_path}')

        landmark[0::2] -= bbox[0]
        landmark[1::2] -= bbox[1]
        landmark[0::2] /= (bbox[2] - bbox[0])
        landmark[1::2] /= (bbox[3] - bbox[1])

        str_landmark = ''
        for cood in landmark:
            str_landmark += str(cood)
            str_landmark += ' '

        wr.writerow([f'{filename}_face{n_box}_{i}.jpg', str_landmark])
        # ina = f'{filename}_face{n_box}.jpg'
        # print(f'Write csv {save_cnt} : {ina}')
        # print()
        # save_cnt += 1

        # img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
        # for ii in range(0, landmark.shape[0], 2):
        #     x = int(landmark[ii] * bbox[2])
        #     y = int(landmark[ii+1] * bbox[3])
        #     face = cv2.circle(face, (x, y), 2, (255, 255, 0), -1)

    #     cv2.imshow('ff', face)
    #
    # cv2.imshow('t', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # if i == 2:
    #     break

f.close()
print('{} creat done!'.format(save_path))