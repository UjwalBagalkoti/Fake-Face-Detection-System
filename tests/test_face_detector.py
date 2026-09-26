import numpy as np
from detection.face_detector import FaceDetector
def test_detector_initializes():
    d=FaceDetector(min_face_size=20); assert isinstance(d.detect(np.zeros((100,100,3),dtype=np.uint8)),list)
