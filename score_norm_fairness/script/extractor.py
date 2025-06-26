import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import skimage
from skimage import transform
from torchvision.transforms import transforms
from score_norm_fairness.extractor import iresnet50
import argparse
import os
from pathlib import Path
import h5py


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

def model_loader(weight, device):
    """
        Load pre-trained weight to the model framework
    """
    model = iresnet50()
    model.load_state_dict(torch.load(weight,map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model

def combined_landmarks():
    # Face landmark
    african_lmk = pd.read_csv("/local/scratch/ethnicity_Data/RFW/images/test/txts/African/African_lmk.txt", delimiter='\t', names=['image_path', 'label', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'nose_x', 'nose_y', 'mouthright_x', 'mouthright_y', 'mouthleft_x', 'mouthleft_y'])
    asian_lmk = pd.read_csv("/local/scratch/ethnicity_Data/RFW/images/test/txts/Asian/Asian_lmk.txt", delimiter='\t', names=['image_path', 'label', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'nose_x', 'nose_y', 'mouthright_x', 'mouthright_y', 'mouthleft_x', 'mouthleft_y'])
    caucasian_lmk = pd.read_csv("/local/scratch/ethnicity_Data/RFW/images/test/txts/Caucasian/Caucasian_lmk.txt", delimiter='\t', names=['image_path', 'label', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'nose_x', 'nose_y', 'mouthright_x', 'mouthright_y', 'mouthleft_x', 'mouthleft_y'])
    indian_lmk = pd.read_csv("/local/scratch/ethnicity_Data/RFW/images/test/txts/Indian/Indian_lmk.txt", delimiter='\t', names=['image_path', 'label', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'nose_x', 'nose_y', 'mouthright_x', 'mouthright_y', 'mouthleft_x', 'mouthleft_y'])

    # Combine the DataFrames
    combined_lmk = pd.concat([african_lmk, asian_lmk, caucasian_lmk, indian_lmk], axis=0, ignore_index=True)
    combined_lmk["path"] = combined_lmk.apply(lambda row: "/local/scratch/ethnicity_Data/RFW/images" + row['image_path'], axis=1)
    combined_lmk['annotation'] = combined_lmk.apply(lambda row: {"path": row['path'], "reye_x": row['reye_x'], "reye_y": row['reye_y'], "leye_x": row['leye_x'], "leye_y": row['leye_y'], \
                                                                 "nose_x": row['nose_x'], "nose_y": row['nose_y'], "mouthleft_x": row['mouthleft_x'], "mouthleft_y": row['mouthleft_y'], "mouthright_x": row['mouthright_x'], "mouthright_y": row['mouthright_y']}, axis=1)
    combined_lmk.drop(columns=['image_path', 'label', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'nose_x', 'nose_y', 'mouthright_x', 'mouthright_y', 'mouthleft_x', 'mouthleft_y'], inplace=True)
    return combined_lmk

def bupt_prepare_landmarks():
    # Face landmark
    lmk = pd.read_csv("/local/scratch/ethnicity_Data/Ethnicity-Aware_Face_Databases/BUPT-Balancedface/clean_annotations.csv")
    lmk["image_path"] = lmk["PATH"].apply(lambda x: x.replace("train/data/", ""))
    lmk["img_name"] = lmk["image_path"].apply(lambda x: x.split("/")[-1])
    lmk["path"] = lmk.apply(lambda row: "/local/scratch/ethnicity_Data/Ethnicity-Aware_Face_Databases/BUPT-Balancedface/images/race_per_7000/" + row['image_path'], axis=1)
    lmk['annotation'] = lmk.apply(lambda row: {"path": row['path'], "reye_x": row['reye_x'], "reye_y": row['reye_y'], "leye_x": row['leye_x'], "leye_y": row['leye_y'], \
                                                                 "nose_x": row['nose_x'], "nose_y": row['nose_y'], "mouthleft_x": row['mouthleft_x'], "mouthleft_y": row['mouthleft_y'], "mouthright_x": row['mouthright_x'], "mouthright_y": row['mouthright_y']}, axis=1)                                                         
    lmk.drop(columns=['image_path', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'nose_x', 'nose_y', 'mouthright_x', 'mouthright_y', 'mouthleft_x', 'mouthleft_y', 'quality'], inplace=True)
    return lmk


def feature_extractor(weight, device, batch_size=64):
    model = model_loader(weight, device)
    align = FaceAlign()
    
    tensor_transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])

    # ============================Test Dataset(RFW)=================================================================================
    lmk = combined_landmarks()
    print("RFW Landmarks Loaded")
    rfw_path = "/local/scratch/ethnicity_Data/RFW/images/test/data"
    output_base_path = '/local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/embedding/RFW'
    Path(output_base_path).mkdir(parents=True, exist_ok=True)


    for demo in os.listdir(rfw_path):
        demo_path = os.path.join(rfw_path, demo)
        demo_output_path = os.path.join(output_base_path, demo)
        Path(demo_output_path).mkdir(parents=True, exist_ok=True)

        for identity in os.listdir(demo_path):
            identity_path = os.path.join(demo_path, identity)
            identity_output_path = os.path.join(demo_output_path, identity)
            Path(identity_output_path).mkdir(parents=True, exist_ok=True)

            batch_images = []
            batch_paths = []
            batch_output_paths = []

            # Loop through images, accumulate batch
            for image in os.listdir(identity_path):
                image_path = os.path.join(identity_path, image)
                img = cv2.imread(image_path)
                annotation = lmk.loc[lmk['path'] == image_path, "annotation"].values[0]
                aligned_img = align.transform(img, annotation)
                tensor_img = tensor_transform(aligned_img)

                batch_images.append(tensor_img)
                batch_paths.append(image_path)
                batch_output_paths.append(os.path.join(identity_output_path, f"{os.path.splitext(image)[0]}.h5"))

                # If batch full or last image
                if len(batch_images) == batch_size:
                    batch_tensor = torch.stack(batch_images).to(device)
                    with torch.no_grad():
                        features = model(batch_tensor).cpu().numpy()

                    for feat, out_path, img_path in zip(features, batch_output_paths, batch_paths):
                        with h5py.File(out_path, "w") as h5_file:
                            h5_file.create_dataset("embedding", data=feat)

                    batch_images = []
                    batch_paths = []
                    batch_output_paths = []

            # Process remaining images in last batch
            if len(batch_images) > 0:
                batch_tensor = torch.stack(batch_images).to(device)
                with torch.no_grad():
                    features = model(batch_tensor).cpu().numpy()

                for feat, out_path, img_path in zip(features, batch_output_paths, batch_paths):
                    with h5py.File(out_path, "w") as h5_file:
                        h5_file.create_dataset("embedding", data=feat)
    
    # ============================Cohort Dataset(BuptBalancedFace)======================================================
    # Load facial landmarks
    lmk = bupt_prepare_landmarks()
    print("BUPT Landmarks Loaded")

    output_base_path = '/local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/embedding/RFW_bupt_iresnet50_arcface/norm' # Original
    Path(output_base_path).mkdir(parents=True, exist_ok=True)

    batch_images = []
    batch_paths = []
    batch_output_paths = []

    for row in lmk.itertuples(index=False):
        demo = row.race
        demo_output_path = os.path.join(output_base_path, demo)
        Path(demo_output_path).mkdir(parents=True, exist_ok=True)   # Ensure demographic folder exists

        identity = row.subject_id
        identity_output_path = os.path.join(demo_output_path,identity)
        Path(identity_output_path).mkdir(parents=True, exist_ok=True)   # Ensure identity folder exists

        image_path = row.path
        img = cv2.imread(image_path)
        aligned_img = align.transform(img,row.annotation)
        tensor_img = tensor_transform(aligned_img)  

        img_name = row.img_name.split(".")[0]
        batch_images.append(tensor_img)
        batch_paths.append(image_path)
        batch_output_paths.append(os.path.join(identity_output_path, f"{img_name}.h5"))
        
        # If batch full or last image
        if len(batch_images) == batch_size:
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                features = model(batch_tensor).cpu().numpy()

            for feat, out_path, img_path in zip(features, batch_output_paths, batch_paths):
                with h5py.File(out_path, "w") as h5_file:
                    h5_file.create_dataset("embedding", data=feat)

            batch_images = []
            batch_paths = []
            batch_output_paths = []

            # Process remaining images in last batch
    if len(batch_images) > 0:
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            features = model(batch_tensor).cpu().numpy()

        for feat, out_path, img_path in zip(features, batch_output_paths, batch_paths):
            with h5py.File(out_path, "w") as h5_file:
                h5_file.create_dataset("embedding", data=feat)



def get_args(command_line_options = None):

    parser = argparse.ArgumentParser("example_extractor",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--weight","-w", required=True, help = "download the pretrained weight and enter its saved path here")

    args = parser.parse_args(command_line_options)

    return args


# def main():
if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor(args.weight, device)
