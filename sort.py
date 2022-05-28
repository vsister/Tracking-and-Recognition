from sort import Sort
from facenet_pytorch import MTCNN
import torch
import numpy as np
import time
import sys
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device, keep_all = True) 

file_results = open('sort_test_results.txt','w')
K = 713 # video length, counted in advance to not use while True
ids = ['0003', '0015', '0021', '0022'] # ChokePoint IDs
capture = cv2.VideoCapture('ChokePoint/output.avi') 
w = capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # getting size to create a video writer with the same size

vid_writer = cv2.VideoWriter("ChokePoint/tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (int(w), int(h)) )
mot_tracker = Sort(max_age = 3, min_hits = 1) #create instance of SORT

start_time = time.time()

for k in range(K): 
    ret, img = capture.read()   
    dist = []
    id_found = -1
    height, width = img.shape[:2]

    faces, prob, landm = mtcnn.detect(img, landmarks=True) #detect faces and landmarks
    if faces is not None:
      det_res = []
      for i in range(len(faces)): #making correct format for tracker input
        det_res.append([faces[i][0],faces[i][1],faces[i][2],faces[i][3],prob[i]])
    
      track_bbs_ids = mot_tracker.update(np.array(det_res)) #updating tracking ids
      #print(track_bbs_ids)
      for [x1,y1,x2,y2,id] in track_bbs_ids:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(img, str(id), (int(x1),int(y1)), cv2.FONT_HERSHEY_PLAIN, 8, (0, 0, 255), 2)
      vid_writer.write(img)
      if len(track_bbs_ids)==0:
        new_line = '-1'+ '\n'
      else:
        new_line = ''
        for t in track_bbs_ids:
          new_line += str(t[4])
          new_line += ' '
        new_line +='\n'
      file_results.write(new_line)

    else: # when no faces found
      new_line = '-1'+ '\n'
      file_results.write(new_line)
      vid_writer.write(img)

vid_writer.release() 
print(time.time() - start_time)
file_results.close()