from facenet_pytorch import MTCNN
import torch
import numpy as np
import time
import sys
import cv2


device = 'cpu'
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device, keep_all = True)


img_path = 'WIDER_train/images/'
param = 0.9
file_annot = open('wider_face_split/wider_face_train_bbx_gt.txt')

def iou(x11,y11,x12,y12,x21,y21,x22,y22):
    xi1 = max(x11,x21)
    yi1 = max(y11,y21)
    xi2 = min(x12,x22)
    yi2 = min(y12,y22)
    l1 = max(xi2-xi1+1,0)
    l2 = max(yi2-yi1+1,0)
    s_i = l1*l2
    s_1 = (x12-x11+1)*(y12-y11+1)
    s_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    s_u = s_1 + s_2 - s_i
    s_iou = s_i/s_u
    return s_iou


def if_not_intersect(x11,y11,x12,y12,faces):
    for [x,y,xx,yy] in faces:
        if iou(x11,y11,x12,y12,x,y,xx,yy) > 0.45:
            return False
    return True


file_results = open('mtcnn_test_results.txt','w')
start_time = time.time()
n = 0
for i in range(1000):
  tp = 0
  faces_annot = []
  image_name = file_annot.readline()[:-1]
  img = cv2.imread(img_path+image_name)
  ni = int(file_annot.readline())
  if ni == 0:
    print ('ni=0: '+image_name)
    ni+=1
  for k in range(ni):
    xywh = file_annot.readline().split()
    x = int(xywh[0])
    y = int(xywh[1])
    w = int(xywh[2])
    h = int(xywh[3])
    faces_annot.append((x,y,w,h))
  faces, prob = mtcnn.detect(img)
  if faces is not None:
    for (xj,yj,wj,hj) in faces_annot:
        if not if_not_intersect(xj,yj,xj+wj-1,yj+hj-1,faces):
          tp+=1
    fp = len(faces) - tp
    for [x, y, xx, yy] in faces:
     cv2.rectangle(img, (x, y), (xx, yy), (0, 255, 255), 2)
     cv2.imwrite('mtcnn_test.jpg', img)
  new_line = str(ni) + ' ' + str(tp) + ' ' + str(fp) +' '+ str(time.time() - start_time) + '\n'
  file_results.write(new_line)

print(time.time() - start_time)

file_results.close()
file_annot.close()
