# ScoreFollowingModel

### Installation
    same as https://github.com/CPJKU/cyolo_score_following
### Dataset
     /MiiSLab_NAS/Student_Work/xiaofu/dataset/msmd


### Training

```
python train.py --train_set ../data/msmd/msmd_train --val_set ../data/msmd/msmd_valid  --config ./models/configs/cyolo.yaml --augment --ir_path ../data/impulse response --score_type basic
```


### Evaluation

```
python eval.py --param_path ../trained_models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/msmd_test --only_onsets --score_type basic
``` 
* Augmentation File
    /MiiSLab_NAS/Student_Work/xiaofu/dataset/impulse response
* score_type 根據model是否為bipage的形式而更改
    * no bipage: basic
    * bipage train: bipage
    * bipage eval: blank_bipage
### Testing/Visualize


```
python test.py --param_path ../trained_models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/<TEST-DIR> --test_piece <PIECE> --seq_enc transformer
```
### Bipage Testing/Visualize
when model trained by bipage

```
python test_bipage.py --param_path ../trained_models/<MODEL-FOLDER>/best_model.pt --test_dir ../data/msmd/<TEST-DIR> --test_piece <PIECE> --seq_enc transformer
```

* --seq_enc 根據模型的sequence encoder設定
    * ours: transformer
    * cyolo: lstm
