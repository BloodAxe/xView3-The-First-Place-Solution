| Experiment                                                                                   :| Checkpoint :| Objectness | Vessel | Fishing | Holdout | Objectness | Vessel | Fishing | Validation | 
|-----------------------------------------------------------------------------------------------|-------------|------------|--------|---------|---------|------------|--------|---------|------------|
| 211031_10_04_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_flips_light_fold0     | Mean AUC    | 0.6 | 0.2 | 0.1 | 0.45286 | 0.65|0.3|0.1|0.46376 |
| 211031_10_04_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_flips_light_fold0     | Aggregate   | 0.5|0.6|0.4|0.43835 | 0.55|0.6|0.6|0.48258 |
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|
| 211031_19_23_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_flips_medium_fold0    | Mean AUC    | 0.6|0.5|0.5|0.41772 | 0.55|0.4|0.2|0.48087 |
| 211031_19_23_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_flips_medium_fold0    | Aggregate   | 0.55|0.3|0.4|0.43126 | 0.5|0.6|0.3|0.51830 |
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|
| 211101_02_58_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_flips_hard_fold0      | Mean AUC    |
| 211101_02_58_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_flips_hard_fold0      | Aggregate   |
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|
| 211031_10_05_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_mse_flips_light_fold0 | Mean AUC    | 0.7|0.4|0.1|0.41812 | 0.7|0.6|0.6|0.38739 |
| 211031_10_05_multilabel_r34_unet_s4_valid_sar_cv_multilabel_regularized_mse_flips_light_fold0 | Aggregate   | 0.7|0.4|0.1|0.41037 | 0.65|0.6|0.6|0.43632 |
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|
| 211031_15_55_multilabel_r34_unet_s4_valid_sar_cv_multilabel_l1_bitemp_flips_light_fold0       | Mean AUC    | 0.7|0.1|0.2|0.30509 | 0.65|0.5|0.1|0.35469 |
| 211031_15_55_multilabel_r34_unet_s4_valid_sar_cv_multilabel_l1_bitemp_flips_light_fold0       | Aggregate   | 
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|
| 211101_00_19_multilabel_r34_unet_s4_valid_sar_cv_multilabel_focal_flips_light_fold0           | Mean AUC    |
| 211101_00_19_multilabel_r34_unet_s4_valid_sar_cv_multilabel_focal_flips_light_fold0           | Aggregate   |
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|
| 211101_08_51_multilabel_r34_unet_s4_valid_sar_cv_multilabel_soft_bce_flips_light_fold0        | Mean AUC    |
| 211101_08_51_multilabel_r34_unet_s4_valid_sar_cv_multilabel_soft_bce_flips_light_fold0        | Aggregate   |
|-----------------------------------------------------------------------------------------------|-------------|---------|------------|

# 28.10.2021 - Resnet34 + Unet| stride 4

Not using houldout split| 4 folds


## Experiments






## Thresholds Config

| Fold | Obj | Vsl | Fsh | Score |
|------|-----|-----|-----|-------|
| 0    | | | | | 
| 1    | | | | | 
| 2    | | | | | 
| 3    | | | | | 


## Command

python submit_multilabel_ensemble_ddp.py\
    runs/211027_17_53_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold0/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    runs/211028_05_17_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold1/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    runs/211028_00_26_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold2/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    runs/211028_15_05_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold3/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    -bs 2 -ot 0.45 0.475 0.5 0.525 0.55 -vt 0.3 0.4 0.45 0.5 0.55 -ft 0.3 0.4 0.45 0.5 0.55 -od submissions/r34_unet_s4

python submit_multilabel_ensemble_ddp.py\
    runs/211027_17_53_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold0/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    runs/211028_05_17_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold1/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    runs/211028_00_26_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold2/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    runs/211028_15_05_multilabel_r34_unet_s4_valid_sar_cv_multilabel_bitemp_flips_light_fold3/checkpoints_metrics_conf_aggregate_score_0.50/best.pth\
    --tile-step 768 -bs 4 -ot 0.45 0.475 0.5 0.525 0.55 -vt 0.3 0.4 0.45 0.5 0.55 -ft 0.3 0.4 0.45 0.5 0.55 -od submissions/r34_unet_s4



## Result

`0024-LB-XXXX-r34_unet_s4_test_predictions_obj_0.500_vsl_0.500_fsh_0.500.csv`
LB Score: 0.51076 	0.68069 	0.39532 	0.92652 	0.70318 	0.72680

`0025-LB-50371-r34_unet_s4_test_predictions_obj_0.550_vsl_0.550_fsh_0.550.csv`
0.50371 	0.67990 	0.34116 	0.92503 	0.70387 	0.73422 	

`0025-LB-50371-r34_unet_s4_test_predictions_flips_tta_obj_0.550_vsl_0.550_fsh_0.550.csv`
0.51000 	0.68052 	0.39965 	0.92750 	0.69268 	0.72733 	

# 27.10.2021 - Resnet18 + Unet| stride 2

Not using houldout split| 4 folds


## Experiments

211025_22_31_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold0
211026_17_43_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold1
211026_22_22_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold2
211026_22_22_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold3

## Thresholds Config

| Fold | Obj | Vsl | Fsh | Score |
|------|-----|-----|-----|-------|
| 0    | | | | | 
| 1    | | | | | 
| 2    | | | | | 
| 3    | | | | | 


## Command

python submit_multilabel_ensemble_ddp.py\
    runs/211026_22_22_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold3/checkpoints_metrics_conf_aggregate_score_0.45/best.pth\
    runs/211025_22_31_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold0/checkpoints_metrics_conf_aggregate_score_0.45/best.pth\
    runs/211026_17_43_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold1/checkpoints_metrics_conf_aggregate_score_0.45/best.pth\
    runs/211026_22_22_multilabel_r18_unet_valid_sar_cv_multilabel_soft_bce_no_offset_flips_light_fold2/checkpoints_metrics_conf_aggregate_score_0.45/best.pth\
    -bs 6 -ot 0.45 0.5 0.55 -vt 0.45 0.5 0.55 -ft 0.45 0.5 0.55 --no-cache -od submissions/r18_unet_s2
