from pathlib import Path

from numpy import array as nparray
from scipy.spatial.distance import euclidean as dist
import cv2
from torch import load as load_state
from torch import no_grad, round
from torch.cuda import is_available as cuda_is_available
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, Lambda

from .network import SmileNetworkBase, SmileNetworkPretrained


PARENT = Path(__file__).parent

class BaseDetector():
    def __init__(self):
        ...
    
    def __call__(self, im: nparray) -> int:
        return self._detect(im)
    
    def _detect(self, im: nparray) -> int:
        ...


class DeepSmileDetector(BaseDetector):
    def __init__(self,
                 pretrained_name='mobilenetv2',
                 weight_file=str(PARENT / 'models/model_finetune_mobilenetv2')):
        self._device = 'cuda' if cuda_is_available() else 'cpu'

        if pretrained_name is None:
            self._net = SmileNetworkBase()
        else:
            self._net = SmileNetworkPretrained(pretrained_name=pretrained_name)
        
        self._net.load_state_dict(
            load_state(
                weight_file
            )
        )
        self._net.eval()

        self._net.to(self._device)

        self._preproc = Compose([
            Lambda(lambda im: cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            Lambda(lambda im: im.unsqueeze(0).to(self._device))
        ])

        self._postproc = Compose([
            Lambda(lambda im: int(round(im).item()))
        ])
    
    def _detect(self, im):
        x = self._preproc(im)
        
        with no_grad():
            y = self._net(x)

        return self._postproc(y)


class CascadeSmileDetector(BaseDetector):
    def __init__(self):
        self._cascades = {
            'face': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'smile': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        }

    def _detect(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        faces = self._cascades['face'].detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            smiles = self._cascades['smile'].detectMultiScale(roi, 1.8, 20)

            if len(smiles) > 0:
                return 1
        
        return 0


class LandmarkBlinkDetector(BaseDetector):
    def __init__(self):
        self._cascades = {
            'face': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        }
        self._fmark = cv2.face.createFacemarkLBF()
        self._fmark.loadModel(str(PARENT / 'models/lbfmodel.yaml'))

        self._eye_thres = 0.2
        self._consec_thres = 1

        self._consec_counter = 0
    
    @staticmethod
    def _ear(lm):
        eye1 = lm[36:42]
        eye2 = lm[42:48]

        def eye_ear(eye):
            a = dist(eye[1], eye[5])
            b = dist(eye[2], eye[4])
            c = dist(eye[0], eye[3])

            return (a + b) / (2*c)
        
        return (eye_ear(eye1) + eye_ear(eye2)) / 2
    
    def _detect(self, im):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        faces = self._cascades['face'].detectMultiScale(gray, 1.3, 5)

        if len(faces) != 0:
            faces = nparray([max(faces, key=lambda f:f[2]*f[3])])
            succ, lmark = self._fmark.fit(gray, faces)

            if succ:
                ear = self._ear(lmark[0][0])

                if ear < self._eye_thres:
                    self._consec_counter += 1
                else:
                    if self._consec_counter >= self._consec_thres:
                        result = 1
                    else:
                        result = 0

                    self._consec_counter = 0
                    return result
        
        return 0
