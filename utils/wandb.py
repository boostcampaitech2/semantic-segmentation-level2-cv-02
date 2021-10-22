import os
import datetime
import wandb
from dotenv import load_dotenv


class Wandb:
    def __init__(self, config):
        """
        wandb config의 auth key값을 사용하여 로그인
        """
        self.config = config
        self.wandb_config = config["wandb"]
        self.unique_tag = (
            self.wandb_config["unique_tag"] if self.wandb_config["unique_tag"] == "" else str(datetime.datetime.now())
        )
        self.entity = self.wandb_config["entity"]
        self.project = self.wandb_config["project"]
        dotenv_path = self.wandb_config["env_path"]
        load_dotenv(dotenv_path=dotenv_path)
        WANDB_AUTH_KEY = os.getenv("WANDB_AUTH_KEY")
        wandb.login(key=WANDB_AUTH_KEY)

    def init_wandb(self, **kwargs):
        """
        :param args: arguments변수
        :param **kwargs: wandb의 태그와 name에 추가하고싶은 내용을 넣어줌 ex_ fold=1
        """
        tags = [f"unique_tag: {self.unique_tag}"]
        name = f"{self.config['name']}_{self.unique_tag}"

        tags.extend(
            [
                f"name: {self.config['name']}",
                f"lr: {self.config['optimizer']['args']['lr']}",
                f"optimizer: {self.config['optimizer']['type']}",
                f"usertrans: {self.config['data_loader']['type']}",
            ]
        )
        name = f"{self.config['arch']['type']}, {self.unique_tag}"
        if kwargs:
            for k, v in kwargs.items():
                tags.append(f"{k}: {v}")
                name += f" {v}{k} "
        wandb.init(tags=tags, entity=self.entity, project=self.project, reinit=True)
        wandb.run.name = name
        wandb.config.update(self.config)

    def log(self, log):
        """
        wandb에 차트 그래프를 그리기 위해 로그를 찍는 함수
        :param phase: 'train' or 'valid'
        :param log: {'epoch': 1, 'loss': 0.74, 'acc': 0.744, 'mIoU': 0.14, 'val_loss': 0.61, 'val_acc': 0.82, 'val_mIoU': 0.168}
        """
        del log["epoch"]

        wandb.log(log)

    # def show_images_wandb(images, y_labels, preds):
    #     """
    #     wandb에 media로 이미지를 출력함

    #     :param images: image array를 받음 [batch,channel,width,height]
    #     :param y_labels: 실제 라벨 데이터
    #     :param preds: 예측한 데이터
    #     """
    #     for i in range(len(y_labels)):
    #         im = images[i, :, :, :]
    #         im = im.permute(1, 2, 0).cuda().cpu().detach().numpy()
    #         wandb.log(
    #             {
    #                 "image_preds": [
    #                     wandb.Image(
    #                         im, caption=f"real: {y_labels[i]}, predict: {preds[i]}"
    #                     )
    #                 ]
    #             }
    #         )
    # my_table = wandb.Table()
    # my_table.add_column("image", wandb.Image(im))
    # my_table.add_column("label", y_labels)
    # my_table.add_column("class_prediction", preds)
    # wandb.log({"image_preds_table": my_table})
