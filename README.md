# Score Normalization for Demographic Fairness in Face Recognition

Fair biometric algorithms have similar verification performance across different demographic groups given a single decision threshold. Unfortunately, for state-of-the-art face recognition networks, score distributions differ between demographics. Contrary to work that tries to align those distributions by extra training or fine-tuning, we solely focus on score post-processing methods. As proved, well-known sample-centered score normalization techniques, Z-norm and T-norm, do not improve fairness for high-security operating points. Thus, we extend the standard Z/T-norm to integrate demographic information in normalization. Additionally, we investigate several possibilities to incorporate cohort similarities for both genuine and impostor pairs per demographic to improve fairness across different operating points. We run experiment on RFW datasets with different demographics (gender and ethnicity) and show that our techniques generally improve the overall fairness of five state-of-the-art pre-trained face recognition networks, without downgrading verification performance.

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
The easiest way is to make use of our pre-extracted features, which you can download from here and extract them into the current directory. These are the exact features used to generate the results shown in the paper. Linux users can also make use of our provided script:

```
$ bash scripts/download_features.sh
```


### Extracting features by yourself

In case you want to run the system end-to-end, you can download the original data and extract features using a pre-trained model of your choice.

#### Datasets:
Get access and download the datasets:
- [RFW](http://www.whdeng.cn/RFW/index.html)

#### Pre-trained models:

Models E1 to E5 are publicly available pre-trained networks used in the paper. You can download their weights from the sources listed below. If you prefer to use your own pre-trained model, you can manually integrate it by modifying the relevant code under the `score_norm_fairness/extractor` directory, specifying your model using the `--extractors` option in `score_norm_fairness/script/score_norm.py` and register your model under `MODELS` in `score_norm_fairness/utils.py`. For example, E6 represents a custom-trained model.


| **Model** | **Network**   | **Training Data** | **Loss Function** |
|-----------|---------------|-------------------|-------------------|
| E1        | ResNet34      | CASIA-WebFace     | ArcFace           |
| E2        | ResNet50      | MS1M-w/o-RFW      | ArcFace           |
| E3        | IResNet100    | Webface12M        | AdaFace           |
| E4        | IResNet100    | MS1M              | MagFace           |
| E5        | IResNet100    | Webface12M        | DALIFace          |
| E6        | ResNet50      | BUPT-BalancedFace | ArcFace           |

- E1 & E2: http://www.whdeng.cn/RFW/model.html
- E3: https://github.com/mk-minchul/AdaFace
- E4: https://github.com/IrvingMeng/MagFace
- E5: DaliID: https://github.com/Gabrielcb/DaliID
- E6: Custom pre-trained network

[//]: # (Distortion-adaptive learned invariance for identification – a robust technique for face recognition and person re-identification.)

`score_norm_fairness/script/extractor.py` is the example code to extract deep features from RFW and BUPT-BalancedFace(as cohorts set) datasets using E6 model. Please change the image directory accordingly to where it stores the images. If you want to use another dataset, please modify the code accordingly. After the `score_norm_fairness/script/extractor.py` is correctly set, modify the `extractor.sh` with the pre-trained weight's path before run below:

`score_norm_fairness/script/extractor.py` provides example code to extract deep features from the RFW and BUPT-BalancedFace (used as the cohort set) datasets using the E6 model. Please ensure the image directory path is correctly set according to your dataset. If you're using a different dataset, adjust the code accordingly. Once the code is correctly configured, edit the `extractor.sh` script to point to the correct path of your pre-trained weights, and then run:

```
$ bash scripts/extractor.sh
```

The features used in the paper are extracted via [Bob framework](https://gitlab.idiap.ch/bob/bob.bio.face).
You are also welcome to try this.
The image preprocessing in Bob is not identical with the one we provide in the above script, so there will be a little difference in the extracted features.


## Usage
With the preassumption that you have the extracted features ready, we introduce the pipeline to compute the cosine similarity scores from extracted features, compute normalization statistics from cohort scores, and apply those statistics to normalize raw scores.

You can directly download the features that we used, or extract features with your own pretrained networks.

To run the pipeline, modify and execute the `rfw.sh` script with the appropriate arguments. The script takes several arguments that control the stages of the pipeline, the normalization methods, and other settings.

```
score_norm.py --stage [train,test] --methods METHODS --demo_name [race,gender] --dataset [rfw,vgg2] --protocol PROTOCOL --data_directory /path/to/data --protocol_directory /path/to/protocols --output_directory /path/to/output
```

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

score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E6 -n race -d rfw -p original -P /local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/protocol/protocols/RFW -D /local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/embedding -o /local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/output/20250527
```

## Evaluating Recognition and Fairness Performance

After generating the score files using the scripts (please focus only on the `_normed.csv` files and ignore the `_cohort.csv` files), you can assess the recognition and fairness performance on individual demographic data or on overall (collated scores) dataset. To perform the evaluation, run the following command:

    $ python score_norm_fairness/evaluate_performance.py <score-csv-file>

This script outputs standard recognition and fairness metrics for both overall and demographic-specific evaluations:
1. Recognition Metrics:
   - Accuracy
   - FNMR@FMR=0.001
   - TMR@FMR=0.001
   - Equal Error Rate (EER) 
2. Fairness Metrics:
   - Standard Deviation (STD)
   - Max-Min
   - Max/Min
   - Max/GeoMean
   - Gini

These metrics help assess both the system’s overall accuracy and its fairness across demographic subgroups.

----
