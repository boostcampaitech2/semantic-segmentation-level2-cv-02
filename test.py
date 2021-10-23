from parse_config import ConfigParser
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import data_loader.datasets as ds
import model as module_arch
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os


def main(config):
    logger = config.get_logger("test")

    # setup data_loader instances
    test_path = "/opt/ml/segmentation/input/data/test.json"

    def collate_fn(batch):
        return tuple(zip(*batch))

    test_transform = A.Compose([ToTensorV2()])
    test_dataset = ds.BasicDataset(data_dir=test_path, ann_file=None, mode="test", transform=test_transform)
    data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, num_workers=4, collate_fn=collate_fn)

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # sample_submisson.csv 열기
    submission = pd.read_csv("./sample_submission.csv", index_col=None)

    # test set에 대한 prediction
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print("Start prediction.")

    file_name_list = []
    preds_array = np.empty((0, size * size), dtype=np.long)

    with torch.no_grad():
        for i, (data, info) in enumerate(tqdm(data_loader)):
            output = model(torch.stack(data).to(device))["out"]
            oms = torch.argmax(output.squeeze(), dim=1).detach().cpu().numpy()
            temp_mask = []
            for d, mask in zip(np.stack(data), oms):
                transformed = transform(image=d, mask=mask)
                mask = transformed["mask"]
                temp_mask.append(mask)
            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], size * size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i["file_name"] for i in image_infos])

    file_names = [y for x in file_name_list for y in x]
    # PredictionString 대입
    for file_name, string in zip(file_names, preds_array):
        submission = submission.append(
            {"image_id": file_name, "PredictionString": " ".join(str(e) for e in string.tolist())}, ignore_index=True
        )

    # submission.csv로 저장
    submission.to_csv("./submission/" + os.path.basename(config.resume) + ".csv", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    config = ConfigParser.from_args(args)
    main(config)
