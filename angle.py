import cv2
import numpy as np

def ref3DModel():
    modelPoints = [[-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [0.0, 0.0, 0.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def angle(faces, landm):
    for i in len(faces):
      face3Dmodel = ref3DModel()
      points = [landm[i][0], landm[i][1], landm[i][2], landm[i][3], landm[i][4]]
      # next 2 files need to be calculated in advance by doing calibration of camera in use (OpenCV can be used)
      camera_matrix =  np.load('camera_matrix.npy')
      dist_koeffs = np.load('dist_koeffs.npy')
      __,rot_vec,__ = cv2.solvePnP(face3Dmodel, points, camera_matrix, dist_koeffs)
      rmat,__ = cv2.Rodrigues(rot_vec)
      angles,__,__,__,__,__= cv2.RQDecomp3x3(rmat)
      appropriate_photo = True
      if abs(angles[1]) > 15:
        appropriate_photo = False
    return appropriate_photo