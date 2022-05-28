from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import time
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device,keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2',device = device).eval()

ids = []
all_time = []
tolerance = 0.84 # threshold
N = 0 # number of ids

file_results = open('test_results_facenet_pytorch.txt','a')
file_test = open('test.txt')
file_ids = open('ids.txt')

line = file_ids.readline().split()
while (len(line)>0):
  ids.append(line[0])
  line = file_ids.readline().split()
  N+=1
ids = np.array(ids)
file_ids.close()
line = file_test.readline().split()
while (len(line)>0):
  dist = []
  id_found = -1
  name = line[0]
  id = int(line[1])
  start_time = time.time()
  img = Image.open('photo_celeba/' + name)
  try:
    img_cropped = mtcnn(img)
    if img_cropped is not None:
      img_cropped = img_cropped.to(device)
      embedding_test = resnet(img_cropped.unsqueeze(0))
      print(embedding_test.shape)
      with open('embedding_facenet_torch_new.npy', 'rb') as f:
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
  new_line = str(name) + ' ' + str(id) + ' ' + str(id_found) +' '+ str(new_time) + '\n'
  file_results.write(new_line)
  line = file_test.readline().split()


print("min: ",min(all_time), "max: ", max(all_time), "avg: ", sum(all_time)/len(all_time))
file_test.close()
file_results.close()
