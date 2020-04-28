import cv2
import numpy as np

import torch
from skimage.transform import SimilarityTransform

from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from models.retinaface import RetinaFace


def norm_crop(img, landmark, image_size=112):
    ARCFACE_SRC = np.array([[
        [122.5, 141.25],
        [197.5, 141.25],
        [160.0, 178.75],
        [137.5, 225.25],
        [182.5, 225.25]
    ]], dtype=np.float32)

    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = ARCFACE_SRC

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

        return min_M, min_index

    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


class FaceDetector:
    def __init__(self, device="cuda", confidence_threshold=0.8):
        self.device = device
        self.confidence_threshold = confidence_threshold

        self.cfg = cfg = cfg_re50
        self.variance = cfg["variance"]
        cfg["pretrain"] = False

        self.net = RetinaFace(cfg=cfg, phase="test").to(device).eval()
        self.decode_param_cache = {}

    def load_checkpoint(self, path):
        self.net.load_state_dict(torch.load(path))

    def decode_params(self, height, width):
        cache_key = (height, width)

        try:
            return self.decode_param_cache[cache_key]
        except KeyError:
            priorbox = PriorBox(self.cfg, image_size=(height, width))
            priors = priorbox.forward()

            prior_data = priors.data
            scale = torch.Tensor([width, height] * 2)
            scale1 = torch.Tensor([width, height] * 5)

            result = (prior_data, scale, scale1)
            self.decode_param_cache[cache_key] = result

            return result

    def detect(self, img):
        device = self.device

        prior_data, scale, scale1 = self.decode_params(*img.shape[:2])

        # REF: test_fddb.py
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device, dtype=torch.float32)

        loc, conf, landms = self.net(img)

        loc = loc.cpu()
        conf = conf.cpu()
        landms = landms.cpu()

        # Decode results
        boxes = decode(loc.squeeze(0), prior_data, self.variance)
        boxes = boxes * scale
        scores = conf.squeeze(0)[:, 1]

        landms = decode_landm(landms.squeeze(0), prior_data, self.variance)
        landms = landms * scale1

        inds = scores > self.confidence_threshold
        boxes = boxes[inds]
        landms = landms[inds]

        return boxes, landms
