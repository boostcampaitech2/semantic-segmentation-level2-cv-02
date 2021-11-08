# Boostcamp Recycle Trash Semantic Segmentation Challenge
Code for 20th place solution in Boostcamp AI Tech Recycle Trash Semantic Segmentation Challenge.

## 📋 Table of content

- [팀 소개](#Team)<br>
- [대회 개요](#Overview)<br>
- [문제 정의 해결 및 방법](#Solution)<br>
- [CODE 설명](#Code)<br>
- [Demo 결과](#Demo)

<br></br>
## 👋 팀 소개 <a name = 'Team'></a>
Contributors  

|김서원|이유진|이한빈|정세종|조현동|허지훈|허정훈|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/swkim-sm'><img src='https://avatars.githubusercontent.com/u/58676931?v=4' width='200px'/></a> |<a href='https://github.com/Yiujin'><img src='https://avatars.githubusercontent.com/u/43367868?v=4' width='200px'/></a>|<a href='https://github.com/binlee52'><img src='https://avatars.githubusercontent.com/u/24227863?v=4' width='200px'/></a>|<a href='https://github.com/sejongjeong'><img src='https://avatars.githubusercontent.com/u/37677446?v=4' width='200px'/></a>|<a href='https://github.com/JODONG2'><img src='https://avatars.githubusercontent.com/u/61579014?v=4' width='200px'/></a>|<a href='https://github.com/hojihun5516'><img src='https://avatars.githubusercontent.com/u/32387358?v=4' width='200px'/></a>|<a href='https://github.com/herjh0405'><img src='https://avatars.githubusercontent.com/u/54921730?v=4' width='200px'/></a>



<br></br>
## ♻ 대회 개요 <a name = 'Overview'></a>
대량 생산, 대량 소비의 시대에 살며 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있다.  
분리수거는 이러한 환경부담을 줄일 수 있는 방법이다. 해당 대회는 쓰레기를 Segmentation하는 모델을 만들어 정확한 분리수거를 돕는 것에 기여한다. 

- Dataset 설명
  - 512 x 512 크기의 train 2617장 (80%) , public test 417장 (10%) , private test 420장(10%) 
  - 총 10개의 class 존재 
     - Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
  - coco format으로 images , annotations 정보 존재
    - images : id, height , width, filename
    - annotatins : id, segmentation mask , bbox, area, category_id , image_id
    
- 평가방법 
    - Semantic Segmentation : mIOU


<br></br>
## 📝 문제 정의 및 해결 방법 <a name = 'Solution'></a>
- Template<br>
  pytorch template 형식에 맞게 직접 구현

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
## 💻 CODE 설명<a name = 'Code'></a>

### Archive contents
```
segmentation
├── input
│   └── data
├── semantic-segmentation-level2-cv-02
│   ├── base
│   ├── data_loader
│   ├── logger
│   ├── loss
│   ├── model
│   ├── trainer
│   └── utils
├── train.py
├── train_kfold.py
├── test.py
├── test_TTA.py
├── visulaize.py
├── csv_ensemble.py
└── create_resize_data.py
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
   config file에 아래와 같은 인자 추가
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
512x512 로 inference 한 csv file 준비
```
python pseudo_labeling.py --test_csv [csv file path]
```
4. Expeiment with 256x256 image 
빠른 실험을 위한 256x256 scale image 로 실험
```
python create_resize_data.py
```
resized data 사용 시 config 에서 train, validation data loader 아래와 같이 수정
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
1. test 이미지 시각화<br>
   semantic-segmentation-level2-cv-02/visualize.py 의 submission_path 에 시각화하고자 하는 csv파일 경로 넣고 Run Cell 
2. validation 이미지 시각화<br>
   wandb 사용, utils/wandb.py 의 show_images_wandb 함수로 validation 시 이미지 시각화


### Ensemble
ensemble_method : Hard Voting
```
cd segmentation/semantic-segmentation-level2-cv-02
```
submission.csv 경로 추가 후 
```
python csv_ensemble.py
```

<br></br>
## 👀 DEMO 결과<a name = 'Demo'></a>
Backbone : EfficientNet-b4  
Segmentation Head : UNet++  
| Original Image | Ground Truth | Predicted Image |
|:--------------:|:--------------:|:--------------:| 
|<a><img src = 'https://user-images.githubusercontent.com/43367868/140676444-6d618757-5eb8-4261-8477-1edbd0e74ae6.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140676804-942640cf-df49-4ffe-b220-28f4c17f7b3e.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140676912-acd44bf1-93fd-4c07-89b7-642d1c8e87f0.png' width ='200px' ></a>|
|<a><img src = 'https://user-images.githubusercontent.com/43367868/140677109-23493624-2fc1-45e0-bfe2-b4a5c2aa57f8.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140677216-1043ab83-4286-492c-a12d-e8e390b929dd.png' width ='200px' ></a>|<a><img src = 'https://user-images.githubusercontent.com/43367868/140677164-6fd04b07-8461-4f0c-b2cc-a3fc73059668.png' width ='200px' ></a>|


data license : Naver Boostcamp AI Tech 대회교육용 재활용 쓰레기 데이터셋. CC BY 2.0