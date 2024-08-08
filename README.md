# Score Normalization for Demographic Fairness in Face Recognition
This repository contains the implementation of score normalization methods mentioned in the paper: "Score Normalization for Demographic Fairness in Face Recognition" presented at the International Joint Conference on Biometrics (IJBC) 2024.
The pipeline supports various methods for score normalization based on demographic information.
In case you find this package interesting, please cite the following publication:

    @inproceedings{linghu2024score,
        title     = {Score Normalization for Demographic Fairness in Face Recognition},
        author    = {Linghu, Yu and de Freitas Pereira, Tiago and Ecabert, Christophe and Marcel, S\'ebastien and G\"unther, Manuel},
        booktitle = {International Joint Conference on Biometrics (IJCB)},
        year      = {2024},
        note      = {\textbf{to appear}}
    }


## Abstract
    Fair biometric algorithms have similar verification performance across different demographic groups given a single decision threshold.
    Unfortunately, for state-of-the-art face recognition networks, score distributions differ between demographics.
    Contrary to work that tries to align those distributions by extra training or fine-tuning, we solely focus on score post-processing methods.
    As proved, well-known sample-centered score normalization techniques, Z-norm and T-norm, do not improve fairness for high-security operating points.
    Thus, we extend the standard Z/T-norm to integrate demographic information in normalization.
    Additionally, we investigate several possibilities to incorporate cohort similarities for both genuine and impostor pairs per demographic to improve fairness across different operating points.
    We run experiments on two datasets with different demographics (gender and ethnicity) and show that our techniques generally improve the overall fairness of five state-of-the-art pre-trained face recognition networks, without downgrading verification performance.
    We also indicate that an equal contribution of False Match Rate (FMR) and False Non-Match Rate (FNMR) in fairness evaluation is required for the highest gains.


## Table of Contents
- [Installation](#installation)
- [License](#license)
- [Dataset](#dataset)
- [Usage](#usage)
- [Supplemental](#supplemental)


## Installation

```
$ git clone https://github.com/AIML-IfI/score-norm-fairness.git
$ cd score-norm-fairness
$ conda env create -f environment.yaml
$ conda activate score_norm_fairness
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Datasets & Pretrained Weights

This package can be used in various ways.
The protocol files are always required, so please [download these protocol files](https://seafile.ifi.uzh.ch/f/c1623c5b26004f56b5ba/?dl=1) and extract them into the current directory.
Linux users can also make use of our provided script:

```
$ bash scripts/download_protocol.sh
```

### Running with pre-extracted features

The easiest way is to make use of our pre-extracted features, which you can [download from here](https://seafile.ifi.uzh.ch/d/da7c9d75790b4f8498ef/)
and extract them into the current directory.
These are the exact features used to generate the results shown in the paper.
(Note: There is no extracted feature for protocol ``vgg2-full-demo``.)
Linux users can also make use of our provided script:

```
$ bash scripts/download_features.sh
```


### Extracting features by yourself

In case you want to run the system end-to-end, you can download the original data and the models yourself, and extract the features.

#### Datasets:
Get access and download the datasets:
- [RFW](http://www.whdeng.cn/RFW/index.html)
- [VGGFace2](https://github.com/ox-vgg/vgg_face2)

#### Pre-trained models:

Download the pretrained weights used in the paper:

| **Model** | **Network**   | **Training Data** | **Loss Function** |
|-----------|---------------|-------------------|-------------------|
| E1        | ResNet34      | CASIA-WebFace     | ArcFace           |
| E2        | ResNet50      | MS1M-w/o-RFW      | ArcFace           |
| E3        | IResNet100    | Webface12M        | AdaFace           |
| E4        | IResNet100    | MS1M              | MagFace           |
| E5        | IResNet100    | Webface12M        | DALIFace          |

- E1 & E2: http://www.whdeng.cn/RFW/model.html
- E3: https://github.com/mk-minchul/AdaFace
- E4: https://github.com/IrvingMeng/MagFace
- E5: DaliID: https://github.com/Gabrielcb/DaliID

[//]: # (Distortion-adaptive learned invariance for identification – a robust technique for face recognition and person re-identification.)

Example Code to extract deep features using E5 and an example image from RFW dataset, so please download the pre-trained weight as above and modify its path before run below: 

```
$ bash scripts/extractor.sh
```

The features used in the paper are extracted via [Bob framework](https://gitlab.idiap.ch/bob/bob.bio.face).
You are also welcome to try this.
The image preprocessing in Bob is not identical with the one we provide in the above script, so there will be a little difference in the extracted features.


## Usage
With the preassumption that you have the extracted features ready, we introduce the pipeline to compute the cosine similarity scores from extracted features, compute normalization statistics from cohort scores, and apply those statistics to normalize raw scores.

You can directly download the features that we used, or extract features with your own pretrained networks.

To run the pipeline, execute the config_rfw.sh/config_vgg_gender.sh/config_vgg_race.sh script with the appropriate arguments. Each script takes several arguments that control the stages of the pipeline, the normalization methods, and other settings.

```
score_norm.py --stage [train,test] --methods METHODS --demo_name [race,gender] --dataset [rfw,vgg2] --protocol PROTOCOL --data_directory /path/to/data --protocol_directory /path/to/protocols --output_directory /path/to/output
```

Besides, it is possible to compute TMRs and generate WERM report from generated csv score files. The latter also includes the demographic-specific FMRs and FNMRs at all required thresholds.



### Arguments

The following command-line arguments are supported:

  -	--stages, -s: A space-separated list of stages. Possible choices: train, test. If train, generate raw scores and cohort scores; if test, use cohort scores to compute statistics to normalize raw scores. Default: train test
  -	--methods, -m: A space-separated list of score normalization methods to be applied. Possible choices: M1, M1.1, M1.2, M2, M2.1, M2.2, M3, M4, M5. Default: M1
  -	--demo-name, -n: Specific demographic to be used for normalization. Possible choices: race, gender. Default: race
  - --dataset, -d: Dataset to be used. Possible choices: rfw, vgg2. Default: rfw
  -	--protocol, -p: Specify the protocol for dataset. Possible choices for rfw: original, random; possible choices for vgg2: vgg2-short-demo, vgg2-full-demo Default: original
  -	--data-directory, -D: Directory containing the dataset/images. Default: None (Must be provided by the user)
  -	--protocol-directory, -P: Directory containing the protocols. Default: None (Must be provided by the user)
  -	--output-directory, -o: Directory for output files. All CSV scores files, including raw scores, cohort scores, and normalized scores, will be saved here. Default: None (Must be provided by the user)


### Example

Here are examples of how to run the pipeline:

```
score-norm --stage train test --methods M1.1 M2.1 M3 M4 M5 --demo-name race --dataset rfw --protocol original --data-directory /path/to/data --protocol-directory /path/to/protocols --output-directory /path/to/output

score-norm --stage train test --methods M1.1 M2.1 M3 M4 M5 --demo-name race --dataset vgg2 --protocol vgg2-short-demo --data-directory /path/to/data --protocol-directory /path/to/protocols --output-directory /path/to/output

score-norm --stage train test --methods M1.1 M2.1 M3 M4 M5 --demo-name gender --dataset vgg2 --protocol vgg2-short-demo --data-directory /path/to/data --protocol-directory /path/to/protocols --output-directory /path/to/output
```

### Evaluation

```
# TMR
tmr-table -s /path/to/SCORES -t TITLES -T NUM_OF_THRESHOLDS -o /path/to/output

# WERM
werm-report -d rfw -s /path/to/SCORES -t TITLES -o /path/to/output

```

### Complete runs

For complete runs, please look into directory `scripts`, where you can find two scripts `vgg2.sh` and `rfw.sh` for running all experiments on the two datasets.
Also, `evaluate.sh` can be used to evaluate the results and plot the WERM report for a given dataset, protocol, demographic and selected extractors and score normalization methods.


## Supplemental

Here we show the supplemental results that are promised in the paper.

### Stability of `random` protocol for RFW dataset

We compute TMR and WERM for all methods on five randomly generated protocols, and assess their standard deviations across the five splits.
Here, we show the average of these values among all methods, as well as the maximum and minimum.

|       |              | E1    | E2    | E3    | E5    |
|-------|--------------|-------|-------|-------|-------|
| **TMR** | **Mean<sub>STD</sub>** | 1.700 | 1.009 | 0.107 | 0.266 |
|       | **Max<sub>STD</sub>**  | 2.748 | 1.293 | 0.164 | 0.478 |
|       | **Min<sub>STD</sub>**  | 1.114 | 0.197 | 0.040 | 0.180 |
| **WERM** | **Mean<sub>STD</sub>** | 0.060 | 0.088 | 0.070 | 0.056 |
|       | **Max<sub>STD</sub>**  | 0.160 | 0.122 | 0.098 | 0.077 |
|       | **Min<sub>STD</sub>**  | 0.022 | 0.058 | 0.038 | 0.035 |

### Spread of FNMR and FMR, beyond WERM values

This is an extended version of Table 4, shows the spread of FMR and FNMR, in addition to TMR and WERM values.

#### VGGFace2 Gender:

| **Network** | **E1** | **E1** | **E1** | **E1** | **E2** | **E2** | **E2** | **E2** | **E3** | **E3** | **E3** | **E3** | **E4** | **E4** | **E4** | **E4** | **E5** | **E5** | **E5** | **E5** |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Metrics** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** |
| Baseline    | 93.55  | 1.6477 | 2.5718 | 1.0557 | 95.48  | 1.4042 | 1.9011 | 1.0372 | 96.71  | 1.2908 | 1.6253 | 1.0251 | 96.88  | 1.1996 | 1.4225 | 1.0116 | 96.76  | 1.3892 | 1.8625 | 1.0361 |
| FSN         | 92.32  | 1.7575 | 3.0506 | 1.0125 | 95.36  | 1.3947 | 1.8498 | 1.0515 | 96.76  | 1.3059 | 1.6033 | 1.0636 | 96.71  | 1.1674 | 1.2918 | 1.0550 | 96.84  | 1.4596 | 2.0103 | 1.0597 |
| M1.1        | 93.43  | 1.1649 | 1.1092 | 1.2235 | 95.65  | 1.0977 | 1.0623 | 1.1343 | 96.67  | 1.0997 | 1.1052 | 1.0942 | 96.88  | 1.0986 | 1.1568 | 1.0432 | 96.92  | 1.0830 | 1.0818 | 1.0843 |
| M2.1        | 93.63  | 1.2033 | 1.1608 | 1.2474 | 95.77  | 1.0932 | 1.0662 | 1.1208 | 96.63  | 1.1092 | 1.1092 | 1.1093 | 96.92  | 1.0349 | 1.0430 | 1.0268 | 96.84  | 1.0926 | 1.0974 | 1.0878 |
| M3          | 92.98  | 1.1779 | 1.0507 | 1.3204 | 95.32  | 1.1346 | 1.1289 | 1.1404 | 96.67  | 1.0445 | 0.9971 | 1.0942 | 96.84  | 1.0366 | 1.0701 | 1.0042 | 96.80  | 1.0476 | 0.9944 | 1.1037 |
| M4          | 93.35  | 1.3064 | 1.3950 | 1.2235 | 95.36  | 1.1233 | 1.0974 | 1.1499 | 96.71  | 1.1540 | 1.2344 | 1.0789 | 96.88  | 1.1420 | 1.2892 | 1.0116 | 96.88  | 1.1289 | 1.1891 | 1.0717 |
| M5          | 93.55  | 1.3906 | 1.6739 | 1.1553 | 95.44  | 1.1811 | 1.2595 | 1.1076 | 96.67  | 1.0505 | 1.0086 | 1.0942 | 96.88  | 1.0474 | 1.0804 | 1.0153 | 96.88  | 1.1871 | 1.3150 | 1.0717 |


#### VGGFace2 Gender Balanced:

| **Network** | **E1** | **E1** | **E1** | **E1** | **E2** | **E2** | **E2** | **E2** | **E3** | **E3** | **E3** | **E3** | **E4** | **E4** | **E4** | **E4** | **E5** | **E5** | **E5** | **E5** |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Metrics** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** |
| Baseline    | 93.02  | 1.7456 | 2.8365 | 1.0742 | 95.40  | 1.4696 | 2.0760 | 1.0403 | 96.63  | 1.2624 | 1.5096 | 1.0557 | 96.80  | 1.2490 | 1.5096 | 1.0334 | 96.71  | 1.3916 | 1.8891 | 1.0251 |
| FSN         | 91.46  | 1.7486 | 2.9466 | 1.0376 | 95.11  | 1.4429 | 1.8986 | 1.0966 | 96.67  | 1.2761 | 1.5257 | 1.0674 | 96.71  | 1.1664 | 1.2896 | 1.0550 | 96.59  | 1.4267 | 1.9969 | 1.0193 |
| M1.1        | 93.35  | 1.1637 | 1.1068 | 1.2235 | 95.65  | 1.1080 | 1.0823 | 1.1343 | 96.67  | 1.0525 | 1.0124 | 1.0942 | 96.76  | 1.0811 | 1.1579 | 1.0094 | 96.92  | 1.1017 | 1.1193 | 1.0843 |
| M2.1        | 93.55  | 1.2197 | 1.1779 | 1.2631 | 95.77  | 1.1014 | 1.0823 | 1.1208 | 96.63  | 1.0686 | 1.0294 | 1.1093 | 96.92  | 1.0425 | 1.0584 | 1.0268 | 96.84  | 1.1097 | 1.1320 | 1.0878 |
| M3          | 92.98  | 1.1723 | 1.0409 | 1.3204 | 95.32  | 1.1426 | 1.1449 | 1.1404 | 96.67  | 1.0408 | 0.9901 | 1.0942 | 96.84  | 1.0455 | 1.0884 | 1.0042 | 96.80  | 1.0688 | 1.0351 | 1.1037 |
| M4          | 93.14  | 1.3244 | 1.4249 | 1.2309 | 95.32  | 1.1330 | 1.1257 | 1.1404 | 96.63  | 1.1258 | 1.1712 | 1.0822 | 96.88  | 1.1642 | 1.3398 | 1.0116 | 96.80  | 1.1633 | 1.2262 | 1.1037 |
| M5          | 93.06  | 1.4569 | 1.8298 | 1.1600 | 95.28  | 1.2022 | 1.3003 | 1.1115 | 96.67  | 1.0408 | 0.9901 | 1.0942 | 96.88  | 1.0483 | 1.0823 | 1.0153 | 96.88  | 1.2092 | 1.3645 | 1.0717 |


#### VGGFace2 Race:

| **Network** | **E1** | **E1** | **E1** | **E1** | **E2** | **E2** | **E2** | **E2** | **E3** | **E3** | **E3** | **E3** | **E4** | **E4** | **E4** | **E4** | **E5** | **E5** | **E5** | **E5** |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Metrics** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** |
| Baseline    | 93.76  | 3.8094 | 10.4621 | 1.3871 | 95.65  | 4.0101 | 12.1923 | 1.3189 | 96.80  | 1.9320 | 3.2329 | 1.1545 | 96.92  | 1.8356 | 2.6008 | 1.2956 | 96.96  | 2.3542 | 4.3319 | 1.2794 |
| FSN         | 92.69  | 3.3962 | 7.2071  | 1.6004 | 95.65  | 4.0190 | 12.1577 | 1.3286 | 96.63  | 1.9067 | 3.0015 | 1.2113 | 96.92  | 1.7605 | 2.3914 | 1.2961 | 96.84  | 4.9612 | 20.1557 | 1.2212 |
| M1.1        | 92.85  | 1.5870 | 1.5414  | 1.6340 | 95.73  | 1.3594 | 1.6506  | 1.1195 | 96.76  | 1.6203 | 2.2726 | 1.1552 | 96.84  | 2.1640 | 3.9016 | 1.2002 | 96.88  | 1.6728 | 2.2049 | 1.2690 |
| M2.1        | 92.28  | 2.1288 | 2.3411  | 1.9358 | 95.69  | 1.4253 | 1.6962  | 1.1977 | 96.71  | 1.3468 | 1.5765 | 1.1506 | 96.96  | 2.9963 | 7.3834 | 1.2160 | 96.80  | 2.0942 | 3.6694 | 1.1952 |
| M3          | 92.98  | 2.5158 | 4.0568  | 1.5601 | 95.52  | 2.6485 | 5.9303  | 1.1828 | 96.80  | 1.6644 | 2.3994 | 1.1545 | 96.88  | 1.8598 | 2.7257 | 1.2690 | 96.80  | 2.4600 | 5.2772 | 1.1467 |
| M4          | 94.00  | 2.6577 | 5.0359  | 1.4026 | 95.69  | 3.7550 | 11.7510 | 1.1999 | 96.84  | 1.7912 | 2.6282 | 1.2208 | 96.92  | 1.6990 | 2.2281 | 1.2956 | 96.96  | 1.9823 | 3.0715 | 1.2794 |
| M5          | 93.92  | 3.4323 | 8.6270  | 1.3656 | 95.65  | 3.6006 | 9.8291  | 1.3189 | 96.80  | 1.9328 | 3.2356 | 1.1545 | 96.88  | 1.9968 | 3.3070 | 1.2057 | 96.92  | 3.9164 | 12.8826 | 1.1906 |



#### VGGFace2 Race Balanced:

| **Network** | **E1** | **E1** | **E1** | **E1** | **E2** | **E2** | **E2** | **E2** | **E3** | **E3** | **E3** | **E3** | **E4** | **E4** | **E4** | **E4** | **E5** | **E5** | **E5** | **E5** |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Metrics** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** |
| Baseline    | 88.62  | 6.4436 | 27.7412 | 1.4967 | 94.25  | 6.0192 | 27.7466 | 1.3058 | 96.51  | 2.9195 | 6.8048 | 1.2526 | 96.47  | 2.7463 | 6.0926 | 1.2379 | 96.39  | 3.2103 | 8.9146 | 1.1560 |
| FSN         | 86.57  | 11.1599| 78.3731 | 1.5891 | 94.05  | 5.2970 | 22.5922 | 1.2419 | 96.35  | 3.8272 | 12.1879 | 1.2018 | 96.39  | 2.0060 | 3.2379 | 1.2428 | 96.35  | 3.9992 | 13.6824 | 1.1689 |
| M1.1        | 93.26  | 2.0317 | 2.6872  | 1.5361 | 95.85  | 2.0109 | 3.6119  | 1.1195 | 96.71  | 1.9065 | 3.1719 | 1.1460 | 96.88  | 2.5114 | 5.2326 | 1.2053 | 96.88  | 1.9237 | 2.9162 | 1.2690 |
| M2.1        | 93.18  | 2.5456 | 3.6084  | 1.7959 | 95.85  | 1.8519 | 3.0058  | 1.1410 | 96.76  | 1.7582 | 2.6868 | 1.1506 | 96.96  | 2.5224 | 5.2326 | 1.2160 | 96.88  | 2.4882 | 4.8787 | 1.2690 |
| M3          | 93.43  | 3.3406 | 8.1677  | 1.3663 | 95.69  | 2.6263 | 5.8313  | 1.1828 | 96.76  | 2.3642 | 4.8773 | 1.1460 | 96.76  | 2.9416 | 6.4759 | 1.3361 | 96.80  | 2.5019 | 5.4586 | 1.1467 |
| M4          | 92.65  | 3.6487 | 9.9694  | 1.3354 | 95.03  | 2.8246 | 5.8324  | 1.3679 | 96.51  | 1.9404 | 3.0060 | 1.2526 | 96.59  | 1.6599 | 2.3022 | 1.1968 | 96.67  | 3.0813 | 7.6354 | 1.2435 |
| M5          | 88.91  | 6.1117 | 27.7412 | 1.3465 | 93.72  | 6.0566 | 27.7466 | 1.3221 | 96.22  | 3.6257 | 11.0964| 1.1847 | 95.81  | 3.3750 | 8.9150 | 1.2777 | 95.73  | 5.6425 | 24.6961 | 1.2892 |



#### RFW Original:

| **Network** | **E1** | **E1** | **E1** | **E1** | **E2** | **E2** | **E2** | **E2** | **E3** | **E3** | **E3** | **E3** | **E4** | **E4** | **E4** | **E4** | **E5** | **E5** | **E5** | **E5** |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Metrics** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** |
| Baseline    | 24.22  | 2.5246 | 5.8430  | 1.0908 | 63.05  | 2.5402 | 5.8430  | 1.1043 | 89.14  | 3.4611 | 9.5169  | 1.2587 | ---    | ---    | ---    | ---    | 89.03  | 2.7167 | 5.8547  | 1.2606 |
| FSN         | 2.19   | 6.8737 | 46.9036 | 1.0073 | 55.81  | 3.0454 | 8.5076  | 1.0901 | 88.60  | 4.5858 | 17.1078 | 1.2292 | ---    | ---    | ---    | ---    | 87.85  | 3.6184 | 10.4813 | 1.2492 |
| M1.1        | 33.11  | 1.6129 | 2.1090  | 1.2334 | 67.13  | 2.0791 | 2.6734  | 1.6169 | 89.20  | 2.7097 | 4.6584  | 1.5761 | ---    | ---    | ---    | ---    | 88.59  | 2.4683 | 3.5565  | 1.7131 |
| M2.1        | 33.38  | 1.4072 | 1.6134  | 1.2274 | 65.35  | 2.0301 | 2.6734  | 1.5415 | 90.41  | 7.0568 | 32.2795 | 1.5427 | ---    | ---    | ---    | ---    | 89.54  | 2.2140 | 2.8658  | 1.7105 |
| M3          | 27.99  | 1.7419 | 2.6734  | 1.1349 | 62.17  | 2.2668 | 3.5565  | 1.4449 | 88.92  | 3.8758 | 9.5169  | 1.5784 | ---    | ---    | ---    | ---    | 89.56  | 2.1883 | 2.8658  | 1.6710 |
| M4          | 26.29  | 2.0549 | 3.5565  | 1.1874 | 62.25  | 2.7727 | 5.8547  | 1.3131 | 89.60  | 3.5602 | 9.5169  | 1.3319 | ---    | ---    | ---    | ---    | 89.91  | 1.8976 | 2.6734  | 1.3470 |
| M5          | 25.63  | 2.2587 | 4.7538  | 1.0732 | 62.54  | 2.5204 | 5.8430  | 1.0872 | 90.05  | 3.1712 | 7.7886  | 1.2912 | ---    | ---    | ---    | ---    | 89.50  | 1.6877 | 2.1090  | 1.3505 |


#### RFW Random:

| **Network** | **E1** | **E1** | **E1** | **E1** | **E2** | **E2** | **E2** | **E2** | **E3** | **E3** | **E3** | **E3** | **E4** | **E4** | **E4** | **E4** | **E5** | **E5** | **E5** | **E5** |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| **Metrics** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** | **TMR ↑** | **WERM ↓** | **M/M FMR ↓** | **M/M FNMR ↓** |
| Baseline    | 60.25  | 2.0418 | 3.6295  | 1.1487 | 89.66  | 2.7152 | 5.8372  | 1.2630 | 98.08  | 2.3202 | 4.7444  | 1.1347 | ---    | ---    | ---    | ---    | 98.04  | 4.3784 | 14.7772 | 1.2973 |
| FSN         | 56.15  | 3.4326 | 10.4603 | 1.1264 | 87.89  | 3.2014 | 8.2447  | 1.2431 | 98.10  | 3.0989 | 8.4906  | 1.1310 | ---    | ---    | ---    | ---    | 98.17  | 7.6021 | 46.8449 | 1.2337 |
| M1.1        | 68.35  | 1.9580 | 2.6734  | 1.4340 | 90.55  | 1.7059 | 1.6134  | 1.8036 | 98.37  | 1.5468 | 1.6086  | 1.4873 | ---    | ---    | ---    | ---    | 98.66  | 3.2601 | 6.4525  | 1.6472 |
| M2.1        | 63.53  | 1.4274 | 1.5051  | 1.3537 | 90.31  | 1.9231 | 2.1027  | 1.7588 | 97.97  | 1.7707 | 2.1048  | 1.4897 | ---    | ---    | ---    | ---    | 98.59  | 3.1609 | 6.4403  | 1.5514 |
| M3          | 64.25  | 1.7578 | 2.1090  | 1.4651 | 89.30  | 2.2723 | 2.6734  | 1.9314 | 98.09  | 1.6354 | 1.6134  | 1.6576 | ---    | ---    | ---    | ---    | 98.42  | 2.0688 | 2.3200  | 1.8448 |
| M4          | 64.11  | 1.7085 | 2.1090  | 1.3840 | 90.84  | 2.6720 | 4.7396  | 1.5064 | 98.06  | 2.5022 | 5.8430  | 1.0716 | ---    | ---    | ---    | ---    | 98.31  | 4.8942 | 20.8191 | 1.1505 |
| M5          | 60.57  | 2.5452 | 5.8430  | 1.1087 | 86.83  | 3.4545 | 9.4980  | 1.2564 | 97.46  | 6.7419 | 32.2388 | 1.4099 | ---    | ---    | ---    | ---    | 97.21  | 5.0545 | 20.8191 | 1.2272 |
