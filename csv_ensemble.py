import os
import cv2
import csv
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import webcolors
import copy

if __name__ == "__main__":
    submission_path = []
    saved_models = "/opt/ml/segmentation/semantic-segmentation-level2-cv-02/saved/models/"
    # csv파일 경로
    # dense crf 거친 csv파일로 하는게 좋을거 같습니다.
    submission_path.append(saved_models + "DeeplabV3_DiceCE_HardTransform/1102_033252/submission.csv")
    # submission_path.append(saved_models+"Unet++_Eff7_obj_cutmix_fold/1101_063358/fold1/submission.csv") # !dense_CRF
    # submission_path.append(saved_models+"Unet++_Eff7_obj_cutmix_9:1val/1101_090806/submission.csv") # !dense_CRF
    submission_path.append(saved_models + "output11.csv")  # 한빈이꺼
    # submission_path.append(saved_models+"output11.csv") # 한빈이꺼
    submission_path.append(saved_models + "output12.csv")  # mmseg
    submission_path.append(saved_models + "output13.csv")  #
    submission_path.append(saved_models + "output14.csv")  #
    submission_path.append(saved_models + "output15.csv")  #
    # submission_path.append(saved_models+"output16.csv") #
    submission_path.append(saved_models + "output17.csv")  #

    root = "../input/data/"
    image_ids = []
    masks = []
    submission = pd.read_csv("./sample_submission.csv", index_col=None)
    read_submission = None
    for i in range(len(submission_path)):
        read_submission = pd.read_csv(submission_path[i], index_col=None)
        mas2 = read_submission["PredictionString"].values
        LE = len(mas2)
        mas = []
        print(LE, "LE", i)
        for j in range(LE):
            mas.append(list(map(int, mas2[j].split())))
        masks.append(mas)
    image_ids = list(read_submission["image_id"].values)
    image_ids = np.array(image_ids)
    # print(image_ids)
    masks = np.array(masks)
    
    for j in range(len(image_ids)):
        image_id = image_ids[j]
        ensemble = []
        mask = []
        for k in range(len(masks)):
            mask.append(masks[k][j])
        voting = np.array(mask).T
        print(j, "/", len(image_ids))
        for i in range(len(voting)):
            vot = str(np.bincount(voting[i]).argmax())
            ensemble.append(vot)
        result = " ".join(ensemble)
        submission = submission.append({"image_id": image_id, "PredictionString": result}, ignore_index=True)
    print(
        len(read_submission["PredictionString"]) == len(submission["PredictionString"]),
        len(submission) == len(read_submission),
    )
    os.makedirs(saved_models + "ensemble", exist_ok=True)
    save_path = saved_models + "/ensemble/ensemble4.csv"
    submission.to_csv(save_path, index=False)
