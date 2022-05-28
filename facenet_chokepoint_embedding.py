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

with open('ChokePoint/embedding_facenet_torch.npy', 'wb') as f:
  for name in ['0003', '0015', '0021', '0022']:

      img = Image.open('ChokePoint/template/'+name+'.jpg')
      img_cropped = mtcnn(img)
      img_cropped = img_cropped.to(device)
      embedding = resnet(img_cropped.unsqueeze(0))
      e = embedding.cpu().detach().numpy()
      np.save(f, e)

