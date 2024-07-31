import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import skimage
from skimage import transform
from torchvision.transforms import transforms

class FaceAlign:
    """
        Image preprocessor defined in https://github.com/deepinsight/insightface
        Given facial landmarks and image, align and crop the image.
    """
    def __init__(
          self,
          **kwargs,
    ):
        self.param = np.array(
                                [[38.2946, 51.6963],
                                [73.5318, 51.5014],
                                [56.0252, 71.7366],
                                [41.5493, 92.3655],
                                [70.7299, 92.2041]],
                                dtype=np.float32)

        self.tform= transform.SimilarityTransform()

    def transform(self,img,annotation):

        reye = [annotation['reye_x'],annotation['reye_y']]
        leye = [annotation['leye_x'],annotation['leye_y']]
        nose = [annotation['nose_x'],annotation['nose_y']]
        mouth_l = [annotation['mouthleft_x'],annotation['mouthleft_y']]
        mouth_r = [annotation['mouthright_x'],annotation['mouthright_y']]
        ldms = np.array([reye,leye,nose,mouth_r,mouth_l],dtype=np.float32)
        self.tform.estimate(ldms, self.param)
        M = self.tform.params[0:2, :]
        img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        return img

def model_loader(weight):
    """
        Load pre-trained weight to the model framework
    """
    from E5 import iresnet100
    model = iresnet100()
    model.load_state_dict(torch.load(weight,map_location="cpu", weights_only=True))
    model.eval()
    
    return model

def feature_extractor(weight):
    """
        Example code to load pre-trained weight, preprocess the example image (from RFW dataset), and forward it into the model and extract deep features, the desired shape should be (512,)
    """
    # Load model and instantiate the face cropper
    model = model_loader(weight)
    align = FaceAlign()

    # load image and facial landmarks
    img = cv2.imread("score_norm_fairness/extractor/example.jpg")
    annotation = {"path":"/test/data/African/m.0b__05j/m.0b__05j_0001.jpg","reye_x":160.5,"reye_y":180.0,"leye_x":239.5,"leye_y":180.0,"nose_x":212.4,"nose_y":232.4,"mouthleft_x":160.3,"mouthleft_y":275.9,"mouthright_x":233.9,"mouthright_y":275.9}
    
    # Preprocess the image
    image = align.transform(img,annotation)
    transform = transforms.ToTensor()
    image = transform(image)
    
    # Forward the cropped image into model and extract deep features
    feature = model(image.unsqueeze(0))[0]

    print(feature.shape)

# Please download the pretrained weight from https://github.com/Gabrielcb/DaliID and run this script
feature_extractor("./pretrained_weights/E5.pt")