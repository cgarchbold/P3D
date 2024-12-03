# dp_protest

This repository holds the implementations for Privacy Preserving Protest Dynamics (P3D) project.

## Goals:
* Synthesize a large number of high-fidelity and diverse protest images with a privacy-preserving deep generative model​

* Perform an image-based downstream multi-level prediction task—violence and the common attributes associated with it (e.g., sign, group size, artifacts, children, police, etc.)

## Getting Started

### Datasets
* UCLA Dataset - (https://paperswithcode.com/dataset/ucla-protest-image)
  * Can be found in ```\scratch\datasets\UCLA-protest\```on hummingbird
* VGKG Dataset - (https://blog.gdeltproject.org/vgkg-a-massive-new-5-5-million-global-protest-image-annotations-dataset-from-worldwide-news/)
  * Can be found in ```\localdisk0\VGKG\``` on hummingbird
* Crowd Counting Datasets
  * NWPU-Crowd - (https://paperswithcode.com/dataset/nwpu-crowd)
    * Can be found in ```\localdisk0\NWPU-Crowd\``` on hummingbird
  * UCF-QNRF - (https://crcv.ucf.edu/data/ucf-qnrf/)
    * Can be found in ```\localdisk0\UCF-QNRF_ECCV18``` on hummingbird
  * JHU-Crowd - (http://www.crowd-counting.com/) 
    * Can be found in ```\localdisk0\jhu_crowd_v2.0``` on hummingbird

### Dependencies

* Python 3+
* Libraries:
  * pytorch
  * torchvision
  * pandas
  * numpy
  * wandb
  * torchmetrics
  * torch-fidelity
  * tqdm
  ...

* Full environment via conda
  * ```conda env create -f environment.yml```

### WGAN and DP-GAN
* Contained in this file tree
* View config to modify training parameters in ```/configs/```
* To train:
```
python train.py --config ./configs/ucla_gan.py
```
* To test:
  * Metrics: Frechet Inception Distance (FID), Inception Score(IS), Learned Perceptual Image Similarity (LPIPS), Kernel Inception Distance (KID)
```
python test.py --config ./configs/ucla_gan.py
```
* Trained models can be found on hummingbird ```\scratch\dp_protest_results\wgan_dpgan ```

### StyleGAN
* File tree begins in ```/stylegan3/```
* See ```/stylegan3/README.md``` for training details
* Training commands stored in ```/stylegan3/run.sh```
* Trained models can be found on hummingbird ```\scratch\dp_protest_results\saved_models\stylegan ```

### Downstream Classification + Violence Estimation
* File tree begins in ```/PROTEST-DETECTION-VIOLENCE-ESTIMATION/```
* See ```/PROTEST-DETECTION-VIOLENCE-ESTIMATION/README.md``` for training + testing details
* Trained models can be found on hummingbird ```\scratch\dp_protest_results\saved_models\downstream ```
* Model results can be found on hummingbird
```\scratch\dp_protest_results\downstream\```
* Additional Experiments
  * Membership Inference -
    * ```Membership.ipynb``` Creates splits
    * ```Membership_eval.ipynb``` Evaluates membership inference ability
  * Facial Attributes - ```face_att.ipynb```

## Authors

Contributors names and contact info

Cohen Archbold
(cohen.archbold@uky.edu)


## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
Special thanks to Dr. Abdullah-Al-Zubaer Imran and the University of Kentucky.

