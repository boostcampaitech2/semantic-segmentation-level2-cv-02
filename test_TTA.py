from parse_config import ConfigParser
import torch
from tqdm import tqdm
import torch.nn.functional as F
import data_loader.data_loaders as module_data
import model as module_arch
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import pandas as pd, numpy as np
from utils import prepare_device
from pathlib import Path
import ttach as tta
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils
import multiprocessing as mp


def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])


def dense_crf(img, output_probs):
    MAX_ITER = 50
    POS_W = 3
    POS_XY_STD = 3
    Bi_W = 4
    Bi_XY_STD = 49
    Bi_RGB_STD = 5

    c = output_probs.shape[0]
    h = output_probs.shape[1]
    w = output_probs.shape[2]

    U = utils.unary_from_softmax(output_probs)
    U = np.ascontiguousarray(U)

    img = np.ascontiguousarray(img)

    d = dcrf.DenseCRF2D(w, h, c)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img, compat=Bi_W)

    Q = d.inference(MAX_ITER)
    Q = np.array(Q).reshape((c, h, w))
    return Q


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    data_loader = config.init_obj("test_data_loader", module_data)

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    models = []
    tta_tfms = tta.Compose(
        [
            tta.HorizontalFlip(),
            tta.Rotate90([0, 90]),
        ]
    )
    tta_model = tta.SegmentationTTAWrapper(model, tta_tfms, merge_mode="mean")
    tta_model.eval()
    # model.eval()

    # sample_submisson.csv 열기
    submission = pd.read_csv("./sample_submission.csv", index_col=None)

    # test set에 대한 prediction
    size = 256
    tar_size = 512
    torch.multiprocessing.set_start_method("spawn")

    transform = A.Compose([A.Resize(size, size)])
    print("Start prediction.")

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)
    resize = A.Resize(size, size)
    with torch.no_grad():
        for i, (data, info) in enumerate(tqdm(data_loader)):
            # output = tta_model(torch.stack(data).to(device))
            output = tta_model(torch.stack(data).to(device))
            final_probs = 0
            ph, pw = output.size(2), output.size(3)
            if ph != tar_size or pw != tar_size:
                output = F.interpolate(input=output, size=(tar_size, tar_size), mode="bilinear", align_corners=True)
            probs = F.softmax(output, dim=1).detach().cpu().numpy()
            pool = mp.Pool(mp.cpu_count())
            images = torch.stack(data).detach().cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            if images.shape[1] != tar_size or images.shape[2] != tar_size:
                images = np.stack([resize(image=im)["image"] for im in images], axis=0)

            final_probs += np.array(pool.map(dense_crf_wrapper, zip(images, probs)))
            pool.close()
            oms = np.argmax(final_probs.squeeze(), axis=1)
            # oms = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()
            temp_mask = []
            for d, mask in zip(np.stack(data), oms):
                transformed = transform(image=d, mask=mask)
                mask = transformed["mask"]
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i["file_name"] for i in info])

    file_names = [y for x in file_name_list for y in x]
    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": " ".join(str(e) for e in string.tolist())}, ignore_index=True
        )

    # submission.csv로 저장
    save_path = Path(config.resume).parent / "submission.csv"
    submission.to_csv(save_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)
    main(config)
