from torchvision.transforms import functional
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import time
import sys
import cv2
from yolox.tracker.byte_tracker import BYTETracker



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
mtcnn = MTCNN(device=device, keep_all = True)
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
class args:
    track_thresh = 0.5
    track_buffer = 1
    match_thresh = 0.8
    aspect_ratio_thresh = 1.6
    min_box_area = 10
    mot20 = False

tracker = BYTETracker(args, frame_rate=1)

d = {}
tolerance = 0.84
K = 713
ids = ['Kevin', 'Richard', 'Bob', 'Anna']
capture = cv2.VideoCapture('ChokePoint/output.avi')
w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
vid_writer = cv2.VideoWriter("ChokePoint/final_system_with_rec.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (int(w), int(h)) )
start_time = time.time()
angle = True
m_cur = 0

for k in range(K):
    ret, img = capture.read()
    dist = []
    id_found = -1
    height, width = img.shape[:2]

    faces, prob, landm = mtcnn.detect(img, landmarks=True)
    if faces is not None:
        det_res = []
        for i in range(len(faces)):
            det_res.append([faces[i][0], faces[i][1], faces[i][2], faces[i][3], prob[i]])

        online_targets = tracker.update(np.array(det_res), (height, width), (height, width))
        # angle can be calculated if needed, use function in angle.py
        for t in online_targets:
            if (m_cur < t.track_id or d[t.track_id] == 'Imposter') and angle:
                tlwh = t.tlwh
                img_cropped = cv2.resize(img[int(tlwh[1]):int(tlwh[1] + tlwh[3]), int(tlwh[0]):int(tlwh[0] + tlwh[2])],
                                         (160, 160))
                img_cropped = functional.to_tensor(np.float32(img_cropped))
                img_cropped = (img_cropped - 127.5) / 128.0
                img_cropped = img_cropped.to(device)
                embedding_test = resnet(img_cropped.unsqueeze(0))
                with open('ChokePoint/embedding_facenet_torch.npy', 'rb') as f:
                    for name in ids:
                        embedding_known = np.load(f)
                        embedding_known_new = embedding_known
                        dist.append(np.linalg.norm(embedding_test.cpu().detach().numpy() - embedding_known_new, axis=1))
                results = np.array(dist)
                if np.min(results) > tolerance:
                    res_id = 'Imposter'
                else:
                    res_id = ids[np.argmin(results)]
                d[t.track_id] = res_id
                m_cur = t.track_id

        for t in online_targets:
            tlwh = t.tlwh
            tid = d[t.track_id]
            #   score = t.score
            cv2.rectangle(img, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])),
                          (255, 0, 0), 2)
            cv2.putText(img, str(tid), (int(tlwh[0]), int(tlwh[1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        vid_writer.write(img)
    else:
        vid_writer.write(img)
        #print('No faces found')

vid_writer.release()
print(time.time() - start_time)
