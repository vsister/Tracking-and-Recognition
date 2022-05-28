from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
import time
import sys


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()

all_time = []

file_template = open('template.txt')
line = file_template.readline().split()

with open('embedding_facenet_torch_new_chetn.npy', 'wb') as f:
    while (len(line) > 0):
        name = line[0]
        id = int(line[1])
        start_time = time.time()
        img = Image.open('photo_celeba/' + name)
        img_cropped = mtcnn(img)

        embedding = resnet(img_cropped.unsqueeze(0))
        all_time.append(time.time() - start_time)
        e = embedding.detach().numpy()
        np.save(f, e)
        print(id)
        line = file_template.readline().split()

print("min: ", min(all_time), "max: ", max(all_time), "avg: ", sum(all_time) / len(all_time))
file_template.close()