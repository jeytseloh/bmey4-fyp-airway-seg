<a name="readme-top"></a>

# Enhancing Deep Learning-Based Airway Segmentation 🫁: Investigating Loss Functions and Medical Foundation Models 
## MEng Bioengineering Individual Project 2023-2024 @ Imperial College London
## 🖋️ Jey Tse Loh (MEng Biomedical Engineering)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Project Overview</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#key-results">Key Results</a></li>
      <ul>
        <li><a href="#investigate-datasets">Investigating Dataset Distribution for the Robustness and Generalisability of a Model</a></li>
        <li><a href="#comparison-study">Comparison Study of Loss Functions in Airway Segmentation</a></li>
        <li><a href="#benchmarking-sam">Benchmarking Medical SAMs on Airway Segmentation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<a name="abstract"></a>
## ℹ️ Project Overview
Pulmonary airway segmentation is vital for the diagnosis and prognosis of lung diseases, but manual segmentation of airways is labour-intensive for clinicians. Recent advancements in automatic segmentation methods have introduced various loss functions tailored for airway segmentation, which are essential for guiding deep learning models. Although various studies have studied and compared loss functions in medical image segmentation, none have specifically focused on airway segmentation. Most existing methods in airway segmentation have been proposed and evaluated with different network architectures and datasets, making comparison and benchmarking difficult.

Additionally, it has been shown that models trained on "clean" healthy data struggle to generalise to "noisy" pathological data. With the recent release of large public benchmark datasets, specifically the ATM'22 and AeroPath datasets, we can better assess the robustness and generalisability of various state-of-the-art methods.

Furthermore, this study explores the application of the Segment Anything Model (SAM), a highly generalisable and adaptable vision Foundation Model (FM), to airway segmentation. Although adaptations of SAM for medical imaging analysis have been evaluated on a wide range of segmentation tasks, few have tackled its application on treelike structures such as airways and vessels.

Therefore, the objectives of the project are summarised as follows:
- To investigate the effect of dataset distributions on the robustness and generalisability of a trained model.
- To perform a comparative analysis of various loss functions proposed for airway segmentation.
- To benchmark the performance and capacity of proposed medical SAM adaptations with and without fine-tuning on airway segmentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="getting-started"></a>
## 🚀 Getting Started
1. Create a virtual environment and activate it.
```bash
conda create -n fyp python=3.10 -y
```
2. Install [PyTorch](https://pytorch.org/get-started/locally/) according to their website instructions.
3. To install nnU-Net:
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
4. To run the custom trainers on nnU-Net, copy the custom trainers, loss functions, preprocessor, data transforms, and dataloaders to the respective directories within the cloned nnU-Net. Ensure that you **re-install** nnU-Net with the custom files.
```bash
# preprocessor
cp /{path_to}/nnunetv2/preprocessors/cl_preprocessor.py /{path_to}/nnUNet/nnunetv2/preprocessing/preprocessors
# custom transforms
cp /{path_to}/nnunetv2/training/data_augmentation/custom_transforms/deep_supervision_downsampling_cl.py /{path_to}/nnUNet/nnunetv2/training/data_augmentation/custom_transforms
cp /{path_to}/bmey4-fyp/nnunetv2/training/data_augmentation/custom_transforms/mirror_transform_cl.py /{path_to}/nnUNet/nnunetv2/training/data_augmentation/custom_transforms
cp /{path_to}/nnunetv2/training/data_augmentation/custom_transforms/spatial_transform_cl.py /{path_to}/nnUNet/nnunetv2/training/data_augmentation/custom_transforms
# data loaders
cp /{path_to}/nnunetv2/training/dataloading/base_data_loader_cl.py /{path_to}/nnUNet/nnunetv2/training/dataloading
cp /{path_to}/nnunetv2/training/dataloading/data_loader_3d_cl.py /{path_to}/nnUNet/nnunetv2/training/dataloading
cp /{path_to}/nnunetv2/training/dataloading/nnunet_dataset_cl.py /{path_to}/nnUNet/nnunetv2/training/dataloading
# trainer
cp /{path_to}/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_{loss}.py /{path_to}/nnUNet/nnunetv2/training/nnUNetTrainer
# loss
cp /{path_to}/nnunetv2/training/loss/{loss}.py /{path_to}/nnUNet/nnunetv2/training/loss

# install nnunet
cd /{path_to}/nnUNet
pip install -e .
```
5. Sample scripts to run patch extraction, preprocessing (nnU-Net), training (nnU-Net), inference (nnU-Net), and volume reconstruction are available [here](nnunetv2/scripts/).
6. Please refer to nnU-Net's documentation for more details on [dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) and [environment variables](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md).
7. For the [patch extraction](utils/extract_patches.py) step, the resulting filename convention is as follows:
```
    extracted_patches/
    ├── img
    │   ├── {case_id}_img_{patch_num}.nii.gz
    │   ├── Aero_001_img_0.nii.gz
    │   ├── Aero_001_img_1.nii.gz
    │   ├── ...
    └── label
        ├── {case_id}_label_{patch_num}.nii.gz
        ├── Aero_001_label_0.nii.gz
        ├── Aero_001_label_1.nii.gz
        ├── ...
```
Note: 
Please follow this [script](utils/file_rename.py) to convert the filenames to match the nnU-Net convention for training/inference on patches:
```
    nnUNet_raw/DatasetXXX_extractedPatches/
    ├── dataset.json
    ├── imagesTr
    │   ├── {case_id}_{patch_num}_0000.nii.gz
    │   ├── Aero_001_0_0000.nii.gz
    │   ├── Aero_001_1_0000.nii.gz
    │   ├── ...
    └── labelsTr
        ├── {case_id}_{patch_num}.nii.gz
        ├── Aero_001_0.nii.gz
        ├── Aero_001_1.nii.gz
        ├── ...
```
8. To perform inference and fine-tuning on MedSAM and SAM-Med3D, please follow the instructions listed in the [LiteMedSAM branch](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) (for 3D images) and [SAM-Med3D main](https://github.com/uni-medical/SAM-Med3D/tree/main).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="key-results"></a>
## 🏆 Key Results
An overview of the methods and key findings are presented here.

<a name="investigate-datasets"></a>
### 💡 Investigating Dataset Distribution for the Robustness and Generalisability of a Model
| **Dataset**       | **Number of Cases** |
|:---|:---:|
| ATM'22            |         239         |
| AeroPath          |          21         |
| ATM'22 + AeroPath |         260         |

**A large dataset with a good representation of true (healthy) airway structures is key for model robustness and generalisability.**

<a name="comparison-study"></a>
### 💡 Comparison Study of Loss Functions in Airway Segmentation

<div align="left">
  <img src="assets/comparison-pipeline-highres.png">
</div>

|              **Loss**              | **Link**                                                                                                                                                                                                                                                                                                                                                                 |
|:----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Combo, **L1**                         | [Combo loss: Handling input and output imbalance in multi-organ segmentation]( https://www.sciencedirect.com/science/article/pii/S0895611118305688?casa_token=4Q_znPQXFOgAAAAA:TReCw4sSZPNo-JUaMX1-eY__K7CnIZyBguDEklHBPnvUEuWDr-U9uclwJRZd5HukFX7RnU6f) <br> [Original Code Release](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2)                                                  |
| General Union (GUL), **L2**                | [Alleviating Class-Wise Gradient Imbalance for Pulmonary Airway Segmentation](https://ieeexplore.ieee.org/abstract/document/9427208?casa_token=yoN6BvlPSAYAAAAA:vh08xX4dJ4YSconamkm5eC5YciU7J4uYIQAxNrd44RXua2vx6HhSDj4Y5w-dByiPTtlBqxg&amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;amp;signout=success) <br> [Original Code Release](https://github.com/haozheng-sjtu/3d-airway-segmentation) |
| Connectivity-Aware Surrogate (CAS), **L3** | [Towards Connectivity-Aware Pulmonary Airway Segmentation](https://ieeexplore.ieee.org/document/10283811) <br> [Original Code Release](https://github.com/Puzzled-Hui/Connectivity-Aware-Airway-Segmentation)                                                                                                                                                                                  |
| Hybrid Continuity, **L4**                  | [Fuzzy Attention Neural Network to Tackle Discontinuity in Airway Segmentation](https://ieeexplore.ieee.org/abstract/document/10129972?casa_token=A6SwmZUGnLIAAAAA:YUd1adKp4yrWHD5hdXpb940u51tB-E63AFJ4XV1qDLbx1pO2VbkA6RTB1k0R_ReCszLBS4Y) <br> [Original Code Release](https://github.com/Nandayang/FANN-for-airway-segmentation)                                                            |
| Penalty Dice, **L5**                       | [NaviAirway: a Bronchiole-sensitive Deep Learning-based Airway Segmentation Pipeline](https://arxiv.org/abs/2203.04294) <br> [Original Code Release](https://github.com/AntonotnaWang/NaviAirway)                                                                                                                                                                                              |

- **Loss functions incorporating topological prior knowledge such as airway centrelines outperform generic overlap-based loss functions (e.g., Dice loss).**
- **From our experiments, CAS loss performed best, particularly in topology-based metrics (i.e., TD, BD, and CCF).**

<a name="benchmarking-sam"></a>
### 💡 Benchmarking Medical SAMs on Airway Segmentation

<div align="left">
  <img src="assets/benchmark-pipeline-highres.png">
</div>

| **Model** | **Link** |
|-----------|----------|
| MedSAM    | [Segment anything in medical images](https://www.nature.com/articles/s41467-024-44824-z) <br> [Original Repository](https://github.com/bowang-lab/MedSAM)         |
| SAM-Med3D | [SAM-Med3D](https://arxiv.org/abs/2310.15161) <br> [Original Repository](https://github.com/uni-medical/SAM-Med3D)         |

- **The performance of medical adaptations of SAM is generally inferior to state-of-the-art specialist methods for airway segmentation, but fine-tuning with airway segmentation data can improve the performance of medical SAMs.**
- **Nonetheless, further advancements are required before practical application is feasible.**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="acknowledgements"></a>
## 🙏 Acknowledgements
- Thanks to all challenge organisers and dataset owners for making their datasets public
  - [ATM'22](https://atm22.grand-challenge.org/)
  - [AeroPath](https://github.com/raidionics/AeroPath)
- Thanks to the authors of the following open-source projects
  - [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
  - [MedSAM](https://github.com/bowang-lab/MedSAM)
  - [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



