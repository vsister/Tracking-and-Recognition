import face_recognition
import numpy as np
import cv2
import time

file_template = open('template.txt')
file_ids = open("ids.txt",'a')
known_faces = []
all_time = []

line = file_template.readline().split()
while (len(line)>0):
  name = line[0]
  id = int(line[1])
  start_time = time.time()
  image = face_recognition.load_image_file("photo_celeba/" + name)
  try:
    encoding = face_recognition.face_encodings(image, model='large')[0]
  except IndexError:
    print("No faces found.", id)
  all_time.append(time.time() - start_time)
  known_faces.append(encoding)
  file_ids.write(str(id) + '\n')
  line = file_template.readline().split()


np.save("encodings_large.npy", np.array(known_faces), allow_pickle=False, fix_imports=False)
print("min: ",min(all_time), "max: ", max(all_time), "avg: ", sum(all_time)/len(all_time))
file_ids.close()
file_template.close()



