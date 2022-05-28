from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import time
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device) 
resnet = InceptionResnetV1(pretrained='vggface2',device = device).eval()

ids = []
all_time = []
tolerance = 0.84 # threshold
N = 0 

file_results = open('masked/test_results_facenet_pytorch_masked.txt','a')
file_test = open('masked/mfr2_labels.txt')
file_ids = open('masked/id_emb.txt')

line = file_ids.readline().split()
while (len(line)>0):
  ids.append(line[0])
  line = file_ids.readline().split()
  N+=1
print(ids)
file_ids.close()
print(N)
line = file_test.readline().split()
while (len(line)>0):
  dist = []
  id_found = -1
  name = line[0][0:-1]
  photo_num = line[1][0:-1]
  if re.search(r'no-mask', line[2]) is None:
    start_time = time.time()
    zeros = '_00'
    if int(photo_num) < 10:
      zeros = zeros+'0'
    img = Image.open('masked/' + name+'/'+name+zeros+photo_num+'.png')
    try:
      img_cropped = mtcnn(img)
      if img_cropped is not None:
        img_cropped = img_cropped.to(device)
        embedding_test = resnet(img_cropped.unsqueeze(0))
        with open('masked/embedding_facenet_torch.npy', 'rb') as f:
          for j in range(N):
            embedding_known = np.load(f)
            embedding_known_new = embedding_known
            dist.append(np.linalg.norm(embedding_test.cpu().detach().numpy() - embedding_known_new, axis=1))
        results = np.array(dist)
        if np.min(results)>tolerance:
          id_found = 0
        else:
          id_found = ids[np.argmin(results)]
    except TypeError:
      print("No faces", id)

    new_time = time.time() - start_time
    all_time.append(new_time)
    new_line = name + ' ' + photo_num + ' ' + str(id_found) +' '+ str(new_time) + '\n'
    print(new_line)
    file_results.write(new_line)
  line = file_test.readline().split()

file_results.close()
file_test.close()