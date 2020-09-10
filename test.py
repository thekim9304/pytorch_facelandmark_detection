import cv2

# file_name = '006_12.png'
# img = cv2.imread(f'E:/DB_FaceLandmark/gi4e/gi4e_database/images/{file_name}')
#
# a = '467.19	171.48	453.52	169.79	439.71	170.57	404.95	172.53	389.97	173.31	376.04	174.09'
# a = list(map(int, (map(float, a.split('\t')))))
#
# for i in range(0, len(a)-1, 2):
#     print(a[i], a[i+1])
#     img = cv2.circle(img, (a[i], a[i+1]), 1, (0, 0, 255), -1)
#
# cv2.imshow('y', img)
# cv2.imwrite(f'C:/Users/th_k9/Desktop/{file_name}', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#


img = cv2.imread('E:/DB_BioID/BioID-FaceDatabase-V1.2/BioID_0005.pgm')

# img = cv2.circle(img, (222, 95), 1, (0, 0, 255), -1)
# img = cv2.circle(img, (151, 96), 1, (0, 0, 255), -1)
cv2.imwrite(f'C:/Users/th_k9/Desktop/ori.png', img)

cv2.imshow('y2', img)
cv2.waitKey()
cv2.destroyAllWindows()