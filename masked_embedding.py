from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import time
import sys
import re


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device = device).eval()

all_time = []

file_template = open('masked/mfr2_labels.txt')
line = file_template.readline().split()
file_ids = open('masked/id_emb.txt','w')

with open('masked/embedding_facenet_torch.npy', 'wb') as f:
  while (len(line)>0):
    name = line[0][0:-1]
    photo_num = line[1][0:-1]
    start_time = time.time()
    if re.search(r'no-mask', line[2]) is not None:
      print(line)
      zeros = '_00'
      if int(photo_num) < 10:
        zeros = zeros+'0'
      img = Image.open('masked/' + name+'/'+name+zeros+photo_num+'.png')
      img_cropped = mtcnn(img)
      img_cropped = img_cropped.to(device)
      embedding = resnet(img_cropped.unsqueeze(0))
      new_line = name + '\n'
      file_ids.write(new_line)
      all_time.append(time.time() - start_time)
      e = embedding.cpu().detach().numpy()
      np.save(f, e)
   
    line = file_template.readline().split()


print("min: ",min(all_time), "max: ", max(all_time), "avg: ", sum(all_time)/len(all_time))
file_template.close()
file_ids.close()