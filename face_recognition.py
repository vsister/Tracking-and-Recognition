import face_recognition
import numpy as np
import cv2
import time

tolerance =  0.55  # threshold
# id_found = 0 if face found is not from existing data set; id_found = -1 if no face found.
known_faces = np.load('encodings_large.npy')

file_ids = open('ids.txt')
ids = []
all_time = []
line = file_ids.readline().split()
while (len(line)>0):
  ids.append(line[0])
  line = file_ids.readline().split()
ids = np.array(ids)
file_ids.close()



file_results = open('test_results_face_rec_large_'+str(tolerance)+'.txt','a')
file_test = open('test.txt')
line = file_test.readline().split()
while (len(line)>0):
    name = line[0]
    id = int(line[1])
    id_found = -1
    start_time = time.time()
    image = face_recognition.load_image_file("photo_celeba/" + name)

    try:
        face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
        encoding = face_recognition.face_encodings(image,face_locations,model="large")[0]
        results = face_recognition.face_distance(known_faces, encoding)
        if np.min(results)>tolerance:
        id_found = 0
        else:
        id_found = ids[np.argmin(results)]
    except IndexError:
        print("No faces found.", id)

new_time = time.time() - start_time
all_time.append(new_time)
new_line = str(name) + ' ' + str(id) + ' ' + str(id_found) +' '+ str(new_time) + '\n'
file_results.write(new_line)
line = file_test.readline().split()
file_test.close()
file_results.close()
print("min: ",min(all_time), "max: ", max(all_time), "avg: ", sum(all_time)/len(all_time))
