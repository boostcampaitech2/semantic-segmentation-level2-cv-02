# Before Use : inference with 512 size image not 256
# Usage : python pseudo_labeling.py --test_csv [path to submission.csv]
# Purpose : generate pseudo labeled json file

import json
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="make pseudo_labeled json file")
parser.add_argument(
    "--test_csv",
    default="/opt/ml/segmentation/semantic-segmentation-level2-cv-02/saved/models/HRNet_ocr_cutmix/1102_141745/submission.csv",
    type=str,
    help="Path to submission file(=pseudo labeled).",
)
parser.add_argument(
    "--save_file", default="/opt/ml/segmentation/input/data/pseudo.json", type=str, help="Path to output file"
)
args = parser.parse_args()


# csv 파일을 불러와서 coco dataset format에 맞게 변환
submission = pd.read_csv(args.test_csv, index_col=None)
# print(len(submission['PredictionString'][0])) # 131071 = (256*2) * 256 - 1

num_of_images = len(submission["PredictionString"])  # 819

image_ids = submission["image_id"][:]
predictedStrings = submission["PredictionString"][:]

size = 256

poses = [[0 for _ in range(size)] for _ in range(size)]

annotations = []

num_classes = 11
anno_idx = 0

# make annotations key in coco json
for img_idx in range(num_of_images):
    tmp_anno = [[] for _ in range(num_classes + 1)]  # class에 해당하는 idx에 좌표값 저장하는 배열

    arr_predictedStrings = predictedStrings[img_idx].split()  # df to list without blank

    for i, predictedClass in enumerate(arr_predictedStrings):
        if predictedClass != "0":  # background 0이 아닌 경우
            x = i // 256  # x 좌표
            y = i % 256  # y 좌표
            tmp_anno[int(predictedClass)].extend([x, y])

    for i in range(1, num_classes + 1):
        if tmp_anno[i]:  # 빈 값이 아니면
            tmp_dict = {
                "id": anno_idx,
                "image_id": image_ids[img_idx],
                "category_id": i,
                "segmentation": [tmp_anno[i]],
                "area": 100.0,
                "bbox": [10, 10, 100, 100],
                "iscrowd": 0,
            }
            anno_idx += 1
            annotations.append(tmp_dict)


# make category key in coco json
categories = [
    {"id": 1, "name": "General trash", "supercategory": "General trash"},
    {"id": 2, "name": "Paper", "supercategory": "Paper"},
    {"id": 3, "name": "Paper pack", "supercategory": "Paper pack"},
    {"id": 4, "name": "Metal", "supercategory": "Metal"},
    {"id": 5, "name": "Glass", "supercategory": "Glass"},
    {"id": 6, "name": "Plastic", "supercategory": "Plastic"},
    {"id": 7, "name": "Styrofoam", "supercategory": "Styrofoam"},
    {"id": 8, "name": "Plastic bag", "supercategory": "Plastic bag"},
    {"id": 9, "name": "Battery", "supercategory": "Battery"},
    {"id": 10, "name": "Clothing", "supercategory": "Clothing"},
]

# make images key in coco json
images = []
for img_idx in range(num_of_images):
    tmp_dict = {
        "license": 0,
        "url": 0,  # null
        "file_name": image_ids[img_idx],
        "height": 512,
        "width": 512,
        "date_captured": 0,  # null
        "id": img_idx,
    }
    images.append(tmp_dict)


# json 파일로 저장
with open(args.save_file, "wt", encoding="UTF-8") as coco:
    json.dump(
        {
            "images": images,
            "categories": categories,
            "annotations": annotations,
        },
        coco,
        indent=4,
    )
