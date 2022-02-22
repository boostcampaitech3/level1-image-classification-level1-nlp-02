# 나랏말싸미

<br>

### Folder Tree

<br>


```
code/
  ├── checkpoint/
  ├── .gitignore
  ├── augmentation.py
  ├── dataloader.py
  ├── dataset.py
  ├── model.py
  ├── train.py
  ├── utils.py
  └── README.md
input/
  └── data/
        ├── eval/
        ├── train/
        │     └── images/
        └── pre_processed_train.csv
```

<br>

### Training

<br>

```bash
python train.py --model_fn your_model_name --wandb_project your_wandb_project_name
```

|Name|Type|Description|
|-|-|-|
|model_fn|**Required**|When you save a model, the model_fn will be a file name|
|wandb_project|**Required**|Type your wandb project name to link log|
|batch_size|Optional|Default 64|
|n_epochs|Optional|Default 50|
|train_valid_ratio|Optional|Default 0.8|