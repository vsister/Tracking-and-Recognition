from facenet_pytorch import MTCNN
import torch
import numpy as np
import time
import sys
import cv2
from yolox.tracker.byte_tracker import BYTETracker

from torchvision.transforms import functional



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device, keep_all = True) 

class args:
  track_thresh = 0.5
  track_buffer =  1
  match_thresh = 0.8
  aspect_ratio_thresh = 1.6
  min_box_area = 10
  mot20 = False

tracker = BYTETracker(args,frame_rate=1)

file_results = open('byte_test_results.txt','w')

K = 713 # video length, counted in advance to not use while True
ids = ['0003', '0015', '0021', '0022'] # ChokePoint IDs
capture = cv2.VideoCapture('ChokePoint/output.avi') 
w = capture.get(cv2.CAP_PROP_FRAME_WIDTH) 
h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) # getting size to create a video writer with the same size
vid_writer = cv2.VideoWriter("ChokePoint/bytetracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (int(w), int(h)) )
start_time = time.time()

for k in range(K): 
    ret, img = capture.read()   
    dist = []
    id_found = -1
    height, width = img.shape[:2]

    faces, prob, landm = mtcnn.detect(img, landmarks=True)#detect faces and landmarks
    if faces is not None:
      det_res = []
      for i in range(len(faces)):
        det_res.append([faces[i][0],faces[i][1],faces[i][2],faces[i][3],prob[i]])  #making correct format for tracker input

      online_targets = tracker.update(np.array(det_res), (height,width), (height,width)) #updating tracking ids
        
      #print(online_targets,'##############################')
      for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
     #   score = t.score
        cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0]+tlwh[2]),int(tlwh[1]+tlwh[3])), (255,0,0), 2)
        cv2.putText(img, str(tid), (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
      vid_writer.write(img)
      if len(online_targets)==0:
        new_line = '-1'+ '\n'
      else:
        new_line = ''
        for t in online_targets:
          new_line += str(t.track_id)
          new_line += ' '
        new_line +='\n'
      file_results.write(new_line)

    else:  # when no faces found
      
      vid_writer.write(img)
      new_line = '-1'+ '\n'
      file_results.write(new_line)

vid_writer.release() 
print(time.time() - start_time)
file_results.close()