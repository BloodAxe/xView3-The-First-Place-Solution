# Winning Solution for xView3 Challenge

This repository contains source code and pretrained models for my (Eugene Khvedchenya) solution to xView 3 Challenge (https://iuu.xview.us/).
It scores the first place on the public LB with aggregate score of 0.603 and the first place on the holdout LB with aggregate score 0.617.

# Installation & Requirements

`conda create -f environment.yml`

# Prebuilt docker container

`docker pull ekhvedchenyateam/xview3:72`

# Solution approach

**Key Ingredients**:
* Single-stage point detection framework inspired by CenterNet
* High-resolution dense prediction maps (stride 2)
* Ensemble of 12 models with left-right flip TTA
* Label smoothing & entropy regularization for missing labels

## Model architecture

For this challenge, I designed a custom neural network model CircleNet. It is an Encoder-Decoder architecture inspired by CenterNet[1] and U-Net[2] architectures. In a nutshell, it's a single-stage object detection model tailored for predicting tiny (just a few pixels), tightly packed objects. The model uses a pre-trained feature extractor [5] (encoder) and U-Net decoder to produce an intermediate feature map (stride 2) that is then fed to the model's head. 
A CircleNet head predicts objectness map, offset length, and two dense classification labels - whether the object in question is a vessel and whether it's fishing or not. Since the competition data contained only object lengths, I've changed the size regression head of the CenterNet to predict only one component  - the object length (in pixels). 
The reasoning behind utilizing such a high-resolution output for dense prediction is two-fold:
Spatial resolution for SAR imagery is 10m/pixel, which causes most ships to occupy just a few pixels in the image.
Vessels/Non-Vessels can be located close to each other. 

By using a finer output map, I wanted to avoid predictions from 'sticking' and being suppressed at the non-maximum suppression. 
I further empirically proved this statement by measuring the localization f1 score for encoding-decoding ground-truth labels. The table below shows that increasing prediction maps stride leads to degraded performance.

| Stride | F1 score |
|-------:|:---------|
|     16 | 0.9672   |
|      8 | 0.9948   |
|      4 | 0.9997   |
|      2 | 0.9999   |
|      1 | 1.0      |


I experimented with several variants of CircleNet architecture during the time of competition. 
After running dozens of architecture search experiments I selected three architectures that performed the best.
In this challenge, due to the time constraint for the inference, I didn't consider heavy encoders (B7, V2L, Transformers). Instead, I used the deep ensembling [3] technique to an average output of different smaller models to reduce the variance of my predictions. And the ensemble of N smaller models has been proven to work better than a single giant network. My best performing solution is an ensemble of 12 models (3x4 scheme - Three architectures from above - B4, B5 & V2S, 4 folds for each architecture). Ensembling is also known to be a good way to fight the label noise, which was present in the training data (more on this in the next section).

* EfficientNet B4 Encoder with Unet Decoder and decoupled head with GroupNorm and PixelShuffle (b4_unet_s2)
* EfficientNet B5 Encoder with Unet Decoder and decoupled head with GroupNorm and PixelShuffle (b5_unet_s2)
* EfficientNet V2S Encoder with Unet Decoder and decoupled head with GroupNorm and PixelShuffle (v2s_unet_s2)

#### Scores of individual models in the final ensemble 

| Model       | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Split | CV                |
|-------------|--------|--------|--------|--------|-------|-------------------|
| b4_unet_s2  | 0.5829 | 0.5110 | 0.5217 | 0.6072 | Valid | 0.5557 ± 0.0467   |
| v2s_unet_s2 | 0.5999 | 0.5127 | 0.5285 | 0.6189 | Valid | 0.5650 ± 0.0522   |
| b5_unet_s2  | 0.6209 | 0.5104 | 0.5252 | 0.5975 | All   | 0.5635 ± 0.0539   |


At the inference stage, I split each scene into 2048x2048px tiles with a step size of 1536px and accumulated predictions for each tile into final output objectness/length/classification feature maps. Processing data in overlapping tiles helped to reduce artifacts near tiles boundaries. It also slightly increased model accuracy (Can be seen as a variant of TTA). However, left-right flip TTA was also employed since it. After final predictions has been assembled,  a CenterNet NMS postprocessing step was applied to obtain a set of detection candidates for that scene. 		
After getting a list of candidate objects with the corresponding objectness, vessel, and fishing scores global thresholding was applied to get the list of final candidates. Three thresholds: objectness, vessel, and a fishing threshold were hyper-parameters that were optimized on the holdout dataset (To avoid confusion - this is not the holdout used for final scoring. I'm referring to a small subset of data that I kept for hyperparameter tuning).
It is worth noting that all inference and postprocessing were performed entirely on the GPU using pure PyTorch, and could be easily ported to ONNX / TensorRT to get even better runtime performance.

## Data acquisition, processing, and manipulation

**Key Insights**:
* Only SAR data (2-channel input)
* Custom SAR normalization to [0..1] range 

After performing exploratory data analysis, it became clear that all present SAR images can be clustered in ~42 unique locations. 
The bad news was that a data leak between train, validation and public test set was present. 
There were scenes from the same geographical location in train, validation, and test. 
The presence of data leak caused some discrepancy in CV/LB scores, and by looking at the holdout scores, I can assume that the holdout dataset was also subject to data leak.
Initially, I tried to set up a 4-fold leak-free cross-validation split. 
However, I observed a significant CV/LB discrepancy and later switched to a leaky 4-fold split stratified by a number of fishing vessels and near-shore objects. I acknowledge that going with leaky validation was a risky move, yet after looking at the distribution of the geographic locations per train/validation/public test I was almost certain that holdout will also contain a data leak in it. 

### Dataset imbalance
    
|      Split | Not Vessel | Vessel | Not Fishing | Fishing | Off-Shore | Near Shore | Near Shore & Vessel | Near Shore & Fishing |
|-----------:|-----------:|-------:|------------:|--------:|----------:|-----------:|--------------------:|---------------------:|
|      train |      16692 |  36375 |       23865 |   12510 |     63829 |        284 |                 135 |                   57 |
| validation |       7148 |  11957 |        2930 |     961 |     13053 |       6171 |                4478 |                   25 |


After getting to 0.5+ zone with SAR-only data I tried to include supplementary data (bathymetry, wind & ice mask) into the model. Unfortunately, all my attempts failed and did not bring any gains and I ditched this effort and kept using 2-channels SAR input. I think there are a couple of reasons why extra data didn't help:
The limiting factor was labels quality, not the lack of signal in the input data. 
 * Bathymetry signal was not that important 
 * The model could infer wind speed/direction from the SAR signal implicitly
 * The low spatial resolution of the supplementary data.

As for data preprocessing I normalized SAR signal from uint16 range to [0...1] using sigmoid activation function with scale and offset parameter tuned to saturate to 0/1 in sea/land regions and have ~[0.2..0.8] range in the vicinity of ships/platforms. This data normalization worked slightly better than naive linear normalization. Regions with missing data were filled with zeros.  

## Training Methodology

**Key Insights**:
* Balanced Sampling (Vessel/Fishing/Shore)  
* Label smoothing to address label noise
* Self-supervision loss for missing labels

As mentioned earlier, the quality of the data was a key limiting factor in my opinion. There were several challenges:
* Different sources of truths with different confidence levels
* Missing annotations for clearly present objects and vice versa.
* Partially missing data for fishing/vessel labels and object lengths.  

To address possible label noise in fishing/vessel classification 
I trained classification heads with label smoothing of 0.05. Apart from label smoothing I also tried training with bi-tempered loss and excluding k% samples with the largest classification loss from a training batch. In practice, label smoothing loss demonstrated the best classification scores and more clearly visible separation in output distributions.
For many objects, ground-truth annotation for fishing/vessel was not available and although the objectness and length outputs could be trained, classification heads did not receive training signals. To alleviate this, I applied Shannon's entropy regularization for outputs of the classification head for the objects with unknown labels. This enforced model to move predictions either to 0 or 1 even for objects with missing ground-truth status. Unfortunately, I haven't figured out how to apply any self-supervised learning for length regression and simply ignored missing values for this task.
As for the errors in the ground truth, fortunately, there were not many of them in the validation dataset, so I did not address it at all. The training dataset was not used in the training of the models at all, due to the fact it was much noisier than validation. 
On cleaning the training dataset
My idea to clean the train set was to generate pseudo-labels using the final ensemble and then compute matches between ground-truth and pseudo-labels. True-positive matches could be kept as high-confidence labels and ignore unmatched objects at the level of the loss function. The reasoning behind this scheme is simple - if the model predictions agree with ground truth - that's probably a correct label. However, for false-positive and false-negative mismatches, we don't know who's wrong and a safer solution is to "turn off" training signal for such samples (One still can use Shannon's regularization for these objects and benefit from it).


For the training, I extracted 1024x1024 patches around the particular object or just the random tile from the scene with probabilities of 0.9 and 0.1 accordingly. For the first case, the object was not picked at random but rather concerning its vessel/fishing status, whether it's an on-shore of the off-shore object and how many objects are in its neighborhood. 
In simple words, we want to balance vessel/non-vessel, fishing/non-fishing, and on-shore/off-shore objects and avoid overfitting on crowded regions (like wind turbine farms). This balancing scheme also makes it possible to use BCE loss (with label smoothing) for training without any bells and whistles. Focal loss is a known method to address the class imbalance, however, in this scenario when label noise is present, it renders Focal loss practically unusable.

For the objectness head, I've used reduced focal loss from [1]. Heatmap encoding was changed from the original object-size dependent scheme to the fixed radius of 3px around each object. This worked very well in practice since the size variation in pixel space was really small. Also, using a fixed size radius solved the problem of encoding objects of unknown lengths.  

Classification heads we trained with weighted BCE loss with label smoothing of 0.05 and label radius of 2px around object center[7].  The weighting matrix put the maximum weight of 1 to pixels in the object center and decreased towards 0 for pixels outside the 2px radius. This effectively increased the number of samples to train the classifier head ~40 times and helped the model to converge faster. 

For size regression, I used MSE loss in pixel space with the same weighting approach as for classification heads and loss scale 0.1. Lastly, for the offset head, I used an MSE loss and loss scale of 0.001.
 
## Data Augmentations
During training, I applied a set of photometric & spatial augmentations to training patches:
* Random brightness & contrast change of the original SAR image
* Gauss Noise
* Horizontal & Vertical flips
* Small affine rotations (up to 5 degrees)
* Small scale change (+- 10% image size)
* Minor Elastic transformation  

Generally, data augmentation for the SAR domain is much harder than for regular RGB images. Therefore set of image augmentations is very limited and it was carefully picked to prevent changing original data distribution.
As a data augmentations pipeline, I used Albumentations[4] library for which I've written some custom-made augmentations tailored for the SAR domain (UnclippedRandomBrightnessContrast, UnclippedGaussNoise).


## Hardware & Training Time
The training was done using 2x3090 GPUs using PyTorch 1.9 in DDP/FP16/SyncBN mode.
Training time:
* B4 - ~6 hours per fold
* B5 - ~9 hours per fold
* V2S - ~6hours per fold

# References
- [1] https://arxiv.org/abs/1904.07850
- [2] https://arxiv.org/abs/1505.04597
- [3] https://arxiv.org/abs/1912.02757
- [4] https://github.com/albumentations-team/albumentations
- [5] https://github.com/rwightman/pytorch-image-models
- [6] https://github.com/BloodAxe/pytorch-toolbelt
- [7] https://arxiv.org/abs/1909.00700


## b5_unet_s2

| Model      | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Holdout | Mean    |
|------------|--------|--------|--------|--------|---------|---------|
| Objectness | 0.250  | 0.325  | 0.35   | 0.3    | 0.51    | 0.30625 |
| Vessel     | 0.3673 | 0.3469 | 0.4081 | 0.1632 | 0.5306  | 0.3213  |
| Fishing    | 0.4489 | 0.6734 | 0.4489 | 0.1836 | 0.4081  | 0.4387  |
| Aggregate  | 0.6209 | 0.5104 | 0.5252 | 0.5975 | 0.6344  | 0.5635 |

## v2s_unet_s2

| Model      | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Holdout | Mean    |
|------------|--------|--------|--------|--------|---------|---------|
| Objectness | 0.4    | 0.4    | 0.4    | 0.4    | 0.51    | 0.4     |
| Vessel     | 0.6938 | 0.0816 | 0.5102 | 0.3637 | 0.5306  | 0.4123  |
| Fishing    | 0.2448 | 0.3061 | 0.6734 | 0.4081 | 0.4285  | 0.4081  |
| Aggregate  | 0.5990 | 0.5127 | 0.5284 | 0.6189 | 0.529   | 0.56475 |

## b4_unet_s2 

| Model      | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Holdout | Mean    |
|------------|--------|--------|--------|--------|---------|---------|
| Objectness | 0.475  | 0.475  | 0.5    | 0.45   | 0.57    | 0.494   |
| Vessel     | 0.1836 | 0.2040 | 0.5102 | 0.4693 | 0.3265  | 0.33872 |
| Fishing    | 0.1836 | 0.3673 | 0.3673 | 0.448  | 0.3877  | 0.35078 |
| Aggregate  | 0.5829 | 0.5110 | 0.5217 | 0.6072 | 0.4944  | 0.5557  |

# Notes

## Nms with kernel size of 3 seems to give optimal performance

heatmap_nms, ks=3
heatmap_nms, ks=5
tight_heatmap_nms, ks=3
tight_heatmap_nms, ks=5

```
loc_fscore	loc_fscore_shore	vessel_fscore	fishing_fscore	length_acc	aggregate	is_vessel_threshold	is_fishing_threshold	objectness_threshold
0.799654	0.471729			0.880184		0.725389		0.63118	    0.593101	0.714286			0.102041				0.5
0.797854	0.457613			0.880431		0.725389		0.631185	0.589553	0.714286			0.102041				0.5
0.79516 	0.477541			0.879449		0.723514		0.630893	0.590231	0.714286			0.102041				0.5
0.798902	0.470016			0.880092		0.725389		0.631419	0.592293	0.714286			0.102041				0.5

```


