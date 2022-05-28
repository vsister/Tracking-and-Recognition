from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import time
import sys
import cv2


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2',device = device).eval()
file_results = open('ChokePoint/results.txt','w')
tolerance = 0.84 # threshold
K = 713 # video length, counted in advance to not use while True
ids = ['0003', '0015', '0021', '0022']
capture = cv2.VideoCapture('ChokePoint/output.avi')
all_time = []
start_time = time.time()

for k in range(K):
    ret, img = capture.read()
    dist = []
    id_found = -1
    try:
      img_cropped = mtcnn(img)
      if img_cropped is not None:
        img_cropped = img_cropped.to(device)
        embedding_test = resnet(img_cropped.unsqueeze(0))
        with open('ChokePoint/embedding_facenet_torch.npy', 'rb') as f:
            for name in ids:
              embedding_known = np.load(f)
              embedding_known_new = embedding_known
              dist.append(np.linalg.norm(embedding_test.cpu().detach().numpy() - embedding_known_new, axis=1))
        results = np.array(dist)
        if np.min(results)>tolerance:
          id_found = 0
        else:
          id_found = ids[np.argmin(results)]
    except TypeError:
      print("No faces")
    new_time = time.time() - start_time
    all_time.append(new_time)
    new_line = str(k) + ' ' + str(id_found) +' '+ str(new_time)+'\n'
    file_results.write(new_line)



#print("min: ",min(all_time), "max: ", max(all_time), "avg: ", sum(all_time)/len(all_time))
print(time.time() - start_time)
file_results.close()