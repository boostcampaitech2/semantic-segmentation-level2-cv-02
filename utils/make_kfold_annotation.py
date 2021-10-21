import json
import argparse
import funcy
import pandas as pd
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(description="Splits COCO annotations file into training and test sets.")
parser.add_argument("--annotations", default="/train_all.json", type=str, help="Path to COCO annotations file.")
parser.add_argument("--save_root", default="/opt/ml/segmentation/input/fold", type=str, help="Path to data root")
parser.add_argument("--data_root", default="/opt/ml/segmentation/input/data", type=str, help="Path to data root")
parser.add_argument(
    "--train", type=str, default=f"/train_fold{0}.json", help="Where to store COCO training annotations"
)
parser.add_argument("--test", type=str, default=f"/val_fold{0}.json", help="Where to store COCO test annotations")
parser.add_argument(
    "--split", dest="split", default=0.9, type=float, help="A percentage of a split; a number in (0, 1)"
)
parser.add_argument("--fold", type=int, default=5, help="the num of K")
parser.add_argument("--seed", type=int, default=123, help="seed")

parser.add_argument(
    "--having-annotations",
    default=True,
    dest="having_annotations",
    action="store_true",
    help="Ignore all images without annotations. Keep only these with at least one annotation",
)

args = parser.parse_args()


def make_train_df(save_root, data_root, ann):
    """
    k-fold를 위한 dataframe을 만드는 함수
    Args:
        save_root ([str]): [만든 dataframe 을 csv로 저장할 path]
        data_root ([str]): [주어진 data 저장된 root]
    """

    dd = defaultdict(list)
    classes = [
        "Background",
        "General trash",
        "Paper",
        "Paper pack",
        "Metal",
        "Glass",
        "Plastic",
        "Styrofoam",
        "Plastic bag",
        "Battery",
        "Clothing",
    ]

    with open(data_root + ann, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        # info = coco['info']
        # licenses = coco['licenses']
        # images = coco['images']
        # categories = coco['categories']
        annotations = coco["annotations"]

        for ann in annotations:
            dd["id"].append(ann["id"])
            dd["image_id"].append(ann["image_id"])
            dd["category_id"].append(ann["category_id"])
            dd["category_name"].append(classes[ann["category_id"]])
            dd["segmentation"].append(ann["segmentation"])
            dd["xmin"].append(ann["bbox"][0])
            dd["ymin"].append(ann["bbox"][1])
            dd["xmax"].append(ann["bbox"][2])
            dd["ymax"].append(ann["bbox"][3])

        print(len(dd["id"]))
        trdf = pd.DataFrame(dd)
        trdf.to_csv(save_root + "/train_original2.csv")
        print(f"saved {save_root}/train_original2.csv")
        return trdf


# k-fold
def stratified_group_k_fold(X, y, groups, k, seed=123):
    """fold 별 image 안의 label 비율이 동일하도록 train, val image index 반환

    Args:
        X (dataframe): make_train_df로 만든 csv
        y (ndarray): category id(= label id)
        groups (ndarray): image id
        k (int): n_fold
        seed (int): random seed. default 123456

    Yields:
        list: train, val 에 해당하는 image id list
    """
    labels_num = y.max()
    # https://stackoverflow.com/a/39132900/14019325
    # 기존 코드의 첫번째 loop와 동일합니다. 각 image 별 label 개수를 확인합니다.
    y_counts_per_group = (
        X.groupby(["image_id", "category_id"]).size().unstack(fill_value=0)
    )  # shape = (n_images, n_label)
    y_counts_per_fold = np.zeros((k, labels_num))

    # scale을 미리 계산하여 연산을 줄입니다.
    y_norm_counts_per_group = y_counts_per_group / y_counts_per_group.sum()
    # suffle & sort
    # shuffle 후 sort하면 같은 std값은 순서 달라짐. 이미지 당 label갯수의 std기준 내림차순 정렬
    shuffled_and_sorted_index = (
        y_norm_counts_per_group.sample(frac=1, random_state=seed).std(axis=1).sort_values(ascending=False).index
    )
    y_norm_counts_per_group = y_norm_counts_per_group.loc[shuffled_and_sorted_index]

    groups_per_fold = defaultdict(set)

    for g, y_counts in zip(y_norm_counts_per_group.index, y_norm_counts_per_group.values):
        best_fold = None
        min_eval = None
        for fold_i in range(k):
            # 기존 코드 eval_y_counts_per_fold 와 동일합니다.
            # 어떤 이미지가 fold에 속할 때 fold별 label std 의 평균이 가장 작으면 해당 fold에 이미지 넣음
            y_counts_per_fold[fold_i] += y_counts
            fold_eval = y_counts_per_fold.std(axis=0).mean()  # numpy를 활용하여 연산을 단순화 합니다.
            y_counts_per_fold[fold_i] -= y_counts
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = fold_i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
            sort_keys=True,
        )


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def train_test_split(images, dev_ind, val_ind):
    """train, val split 하는 함수

    Args:
        images (dict)): train annotation file 의 image dictionary
        dev_ind (list): train 에 들어갈 image id list
        val_ind (list): validation 에 들어갈 image id list

    Returns:
        dict: train, val에 해당하는 image annotation 정보를 담은 dictionary
    """
    x = [i for i in images if i["id"] in dev_ind]
    y = [i for i in images if i["id"] in val_ind]
    return x, y


def main(args, dev_ind, val_ind):
    with open(args.data_root + args.annotations, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        info = coco["info"]
        licenses = coco["licenses"]
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a["image_id"]), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i["id"] not in images_with_annotations, images)

        x, y = train_test_split(images, dev_ind=dev_ind, val_ind=val_ind)

        save_coco(args.data_root + args.train, info, licenses, x, filter_annotations(annotations, x), categories)
        save_coco(args.data_root + args.val, info, licenses, y, filter_annotations(annotations, y), categories)

        print(
            "Saved {} entries in {} and {} in {}".format(
                len(x), args.data_root + args.train, len(y), args.data_root + args.val
            )
        )


if __name__ == "__main__":
    trdf = make_train_df(args.save_root, args.data_root, args.annotations)

    train_x = trdf.copy()
    train_y = train_x["category_id"].values
    groups = train_x["image_id"].values
    seed = args.seed
    k = args.fold

    for fold_ind, (dev_ind, val_ind) in enumerate(stratified_group_k_fold(train_x, train_y, groups, k, seed), 1):
        dev_y, val_y = train_y[dev_ind], train_y[val_ind]
        dev_groups, val_groups = groups[dev_ind], groups[val_ind]

        assert len(set(dev_groups) & set(val_groups)) == 0, "not match"
        args.train = f"/train_fold{fold_ind}.json"
        args.val = f"/val_fold{fold_ind}.json"
        print(args)
        main(args, dev_groups, val_groups)
        print("save all k-fold annotation files")
