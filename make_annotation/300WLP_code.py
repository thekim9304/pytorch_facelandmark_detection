import os
import cv2
import csv
from scipy import io
import numpy as np

fishing_landmarks = [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 46]

db_path = 'E:/DB_FaceLandmark/300W-LP'
# save_path = 'C:/Users/th_k9/Desktop/pytorch_facelandmark_detection/dataset/300W-LP'
save_path = 'F:/07_Data/300W-LP/only_face'

f = open(f'{save_path}/train.csv', 'w', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow(['ImageID', 'landmarks'])

db_dirs = os.listdir(db_path)
db_dirs.remove('landmarks')
landmark_root = os.path.join(db_path, 'landmarks')
landmark_dirs = os.listdir(landmark_root)

add_region = 10
for landmark_dir in landmark_dirs:
    print(f'{landmark_dir} start!!')
    landmark_files = os.listdir(os.path.join(landmark_root, landmark_dir))

    for landmark_file in landmark_files:
        mat_file_abs_path = os.path.join(landmark_root, landmark_dir, landmark_file)
        mat_file = io.loadmat(mat_file_abs_path)
        landmark2d = mat_file['pts_2d']

        img_id = os.path.splitext(landmark_file)[0][:-4]
        img_name = f'{img_id}.jpg'
        img_file_abs_path = os.path.join(db_path, landmark_dir, img_name)
        img_flip_file_abs_path = os.path.join(db_path, f'{landmark_dir}_Flip', img_name)

        img = cv2.imread(img_file_abs_path)
        img_flip = cv2.imread(img_flip_file_abs_path)
        width = img.shape[1]

        xmin, ymin, xmax, ymax = 1000, 1000, 0, 0
        for i, landmark in enumerate(landmark2d):
            if landmark[0] < xmin:
                xmin = int(landmark[0])
            if landmark[0] > xmax:
                xmax = int(landmark[0])
            if landmark[1] < ymin:
                ymin = int(landmark[1])
            if landmark[1] > ymax:
                ymax = int(landmark[1])

        xmin -= add_region
        xmax += add_region
        ymax += add_region

        landmark2d = np.array(landmark2d)
        landmark2d_flip = landmark2d.copy()
        landmark2d[:, 0] -= xmin
        landmark2d[:, 1] -= ymin
        landmark2d[:, 0] /= (xmax-xmin)
        landmark2d[:, 1] /= (ymax-ymin)
        landmark2d_flip[:, 0] = abs(landmark2d_flip[:, 0] - width)
        landmark2d_flip[:, 0] -= (width-xmax)
        landmark2d_flip[:, 1] -= ymin
        landmark2d_flip[:, 0] /= (abs(width-xmin) - (width - xmax))
        landmark2d_flip[:, 1] /= (ymax-ymin)

        face_region = img[ymin:ymax, xmin:xmax].copy()
        face_region_flip = img_flip[ymin:ymax, width-xmax:abs(width-xmin)].copy()

        face_region = cv2.resize(face_region, (224, 224))
        face_region_flip = cv2.resize(face_region_flip, (224, 224))

        img_save_path = os.path.join(f'{save_path}/imgs/{img_id}')
        cv2.imwrite(f'{img_save_path}.jpg', face_region)
        cv2.imwrite(f'{img_save_path}_flip.jpg', face_region_flip)

        for i, j in zip(landmark2d, landmark2d_flip):
            face_region = cv2.circle(face_region, (int(i[0] * face_region.shape[1]), int(i[1] * face_region.shape[0])), 1, (255, 255, 255), -1)
            face_region_flip = cv2.circle(face_region_flip, (int(j[0] * face_region_flip.shape[1]), int(j[1] * face_region_flip.shape[0])), 1, (255, 255, 255), -1)

        str_landmark = ''
        str_landmark_flip = ''
        for land, land_flip in zip(landmark2d, landmark2d_flip):
            str_landmark += f'{land[0] * face_region.shape[1]} {land[1] * face_region.shape[0]} '
            str_landmark_flip += f'{land_flip[0] * face_region_flip.shape[1]} {land_flip[1] * face_region_flip.shape[0]} '

        wr.writerow([f'{img_id}.jpg', str_landmark[:-1]])
        wr.writerow([f'{img_id}_flip.jpg', str_landmark_flip[:-1]])

        '''
        check
        '''

        cv2.imshow('t', img)
        cv2.imshow('c', img_flip)
        cv2.imshow('f', face_region)
        cv2.imshow('ff', face_region_flip)
        img_save_path = os.path.join(f'{save_path}/check_imgs/{img_id}')
        cv2.imwrite(f'{img_save_path}.jpg', face_region)
        cv2.imwrite(f'{img_save_path}_flip.jpg', face_region_flip)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

# f.close()