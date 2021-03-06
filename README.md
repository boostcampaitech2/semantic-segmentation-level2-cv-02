# Boostcamp Recycle Trash Semantic Segmentation Challenge
Code for 20th place solution in Boostcamp AI Tech Recycle Trash Semantic Segmentation Challenge.

## ๐ Table of content

- [ํ ์๊ฐ](#Team)<br>
- [๋ํ ๊ฐ์](#Overview)<br>
- [๋ฌธ์  ์ ์ ํด๊ฒฐ ๋ฐ ๋ฐฉ๋ฒ](#Solution)<br>
- [CODE ์ค๋ช](#Code)<br>
- [Demo ๊ฒฐ๊ณผ](#Demo)

<br></br>
## ๐ ํ ์๊ฐ <a name = 'Team'></a>
Contributors  

|๊น์์|์ด์ ์ง|์ดํ๋น|์ ์ธ์ข|์กฐํ๋|ํ์งํ|ํ์ ํ|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/swkim-sm'><img src='https://avatars.githubusercontent.com/u/58676931?v=4' width='200px'/></a> |<a href='https://github.com/Yiujin'><img src='https://avatars.githubusercontent.com/u/43367868?v=4' width='200px'/></a>|<a href='https://github.com/binlee52'><img src='https://avatars.githubusercontent.com/u/24227863?v=4' width='200px'/></a>|<a href='https://github.com/sejongjeong'><img src='https://avatars.githubusercontent.com/u/37677446?v=4' width='200px'/></a>|<a href='https://github.com/JODONG2'><img src='https://avatars.githubusercontent.com/u/61579014?v=4' width='200px'/></a>|<a href='https://github.com/hojihun5516'><img src='https://avatars.githubusercontent.com/u/32387358?v=4' width='200px'/></a>|<a href='https://github.com/herjh0405'><img src='https://avatars.githubusercontent.com/u/54921730?v=4' width='200px'/></a>



<br></br>
## โป ๋ํ ๊ฐ์ <a name = 'Overview'></a>
๋๋ ์์ฐ, ๋๋ ์๋น์ ์๋์ ์ด๋ฉฐ '์ฐ๋ ๊ธฐ ๋๋', '๋งค๋ฆฝ์ง ๋ถ์กฑ'๊ณผ ๊ฐ์ ์ฌ๋ฌ ์ฌํ ๋ฌธ์ ๋ฅผ ๋ณ๊ณ  ์๋ค.  
๋ถ๋ฆฌ์๊ฑฐ๋ ์ด๋ฌํ ํ๊ฒฝ๋ถ๋ด์ ์ค์ผ ์ ์๋ ๋ฐฉ๋ฒ์ด๋ค. ํด๋น ๋ํ๋ ์ฐ๋ ๊ธฐ๋ฅผ Segmentationํ๋ ๋ชจ๋ธ์ ๋ง๋ค์ด ์ ํํ ๋ถ๋ฆฌ์๊ฑฐ๋ฅผ ๋๋ ๊ฒ์ ๊ธฐ์ฌํ๋ค. 

- Dataset ์ค๋ช
  - 512 x 512 ํฌ๊ธฐ์ train 2617์ฅ (80%) , public test 417์ฅ (10%) , private test 420์ฅ(10%) 
  - ์ด 10๊ฐ์ class ์กด์ฌ 
     - Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - coco format์ผ๋ก images , annotations ์ ๋ณด ์กด์ฌ
    - images : id, height , width, filename
    - annotatins : id, segmentation mask , bbox, area, category_id , image_id
    
- ํ๊ฐ๋ฐฉ๋ฒ 
    - Semantic Segmentation : mIOU


<br></br>
## ๐ ๋ฌธ์  ์ ์ ๋ฐ ํด๊ฒฐ ๋ฐฉ๋ฒ <a name = 'Solution'></a>
- Template<br>
  pytorch template ํ์์ ๋ง๊ฒ ์ง์  ๊ตฌํ

- Model<br>
  FCN<br>
  UNet, UNet++<br>
  DeepLab, DeepLab++, DeepLab+++<br>
  HRNet, HRNet_OCR<br>

- Techniques<br>
  Data hard augmentation<br>
  Mixup, Cutmix<br>
  Stratified K-Fold<br>
  Pseudo Labeling<br>
  TTA<br>
  Ensemble<br>
  WandB<br>


<br></br>
## ๐ป CODE ์ค๋ช<a name = 'Code'></a>

### Archive contents
```
segmentation
โโโ input
โ   โโโ data
โโโ semantic-segmentation-level2-cv-02
โ   โโโ base
โ   โโโ data_loader
โ   โโโ logger
โ   โโโ loss
โ   โโโ model
โ   โโโ trainer
โ   โโโ utils
โโโ train.py
โโโ train_kfold.py
โโโ test.py
โโโ test_TTA.py
โโโ visulaize.py
โโโ csv_ensemble.py
โโโ create_resize_data.py
```

### Train
```
cd segmentation/semantic-segmentation-level2-cv-02
```
1. vanilla train   
```
python train.py [config path]
python train.py configs/config.json
```
2. k-fold train  
   config file์ ์๋์ ๊ฐ์ ์ธ์ ์ถ๊ฐ
   ```
       "kfold": {
        "flag": true,
        "cnt": 5,
        "train_fold": "../input/data/train_fold.json", 
        "valid_fold": "../input/data/val_fold.json"        
    },
   ```
```
python train_kfold.py [config path]
```
3. Pseudo Labeling Train & Inference   
512x512 ๋ก inference ํ csv file ์ค๋น
```
python pseudo_labeling.py --test_csv [csv file path]
```
4. Expeiment with 256x256 image 
๋น ๋ฅธ ์คํ์ ์ํ 256x256 scale image ๋ก ์คํ
```
python create_resize_data.py
```
resized data ์ฌ์ฉ ์ config ์์ train, validation data loader ์๋์ ๊ฐ์ด ์์ 
```
"data_loader": {
"type": "ResizedBasicDataLoader",
"args": {
"dataset": {
"type": "ResizedBasicDataset",
"args": {
"data_dir": "../input/resized_data_256",
"mode": "train",
"transform": {"type": "BasicTransform", "args": {}}
}
```

### Inference
```
cd segmentation/semantic-segmentation-level2-cv-02
```
1. vanilla inference  
```
python test.py -c [config path] -r [pth path]
```
1. TTA inference
```
python test_TTA.py -c [config path] -r [pth path]
```

### Visualization Result
1. test ์ด๋ฏธ์ง ์๊ฐํ<br>
   semantic-segmentation-level2-cv-02/visualize.py ์ submission_path ์ ์๊ฐํํ๊ณ ์ ํ๋ csvํ์ผ ๊ฒฝ๋ก ๋ฃ๊ณ  Run Cell 
2. validation ์ด๋ฏธ์ง ์๊ฐํ<br>
   wandb ์ฌ์ฉ, utils/wandb.py ์ show_images_wandb ํจ์๋ก validation ์ ์ด๋ฏธ์ง ์๊ฐํ


### Ensemble
ensemble_method : Hard Voting
```
cd segmentation/semantic-segmentation-level2-cv-02
```
submission.csv ๊ฒฝ๋ก ์ถ๊ฐ ํ 
```
python csv_ensemble.py
```

<br></br>
## ๐ DEMO ๊ฒฐ๊ณผ<a name = 'Demo'></a>
Backbone : EfficientNet-b4  
Segmentation Head : UNet++  
| Original Image | Ground Truth | Predicted Image |
|:--------------:|:--------------:|:--------------:| 
|<a><img src = 'https://user-images.githubusercontent.com/43367868/140676444-6d618757-5eb8-4261-8477-1edbd0e74ae6.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140676804-942640cf-df49-4ffe-b220-28f4c17f7b3e.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140676912-acd44bf1-93fd-4c07-89b7-642d1c8e87f0.png' width ='200px' ></a>|
|<a><img src = 'https://user-images.githubusercontent.com/43367868/140677109-23493624-2fc1-45e0-bfe2-b4a5c2aa57f8.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140677216-1043ab83-4286-492c-a12d-e8e390b929dd.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140677164-6fd04b07-8461-4f0c-b2cc-a3fc73059668.png' width ='200px' ></a>|


data license : Naver Boostcamp AI Tech ๋ํ๊ต์ก์ฉ ์ฌํ์ฉ ์ฐ๋ ๊ธฐ ๋ฐ์ดํฐ์. CC BY 2.0