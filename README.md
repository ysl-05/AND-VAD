# Aligning Normal Representations in Diffusion for VAD （AND-VAD）
This repository contains the official implementation of the paper "Aligning Normal Representations in Diffusion Model for Video Anomaly Detection". We propose an self-supervised video anomaly detection framework based on conditional diffusion models, which leverages **normal representation alignment** and **group-supervised learning strategies** to guide the diffusion model to learn semantic patterns of normal behaviors, enhancing the discrimination ability for anomalies.


## 🔧 Environment Setup  
```bash
# Create a virtual environment
conda create -n vad_diffusion python=3.8
conda activate vad_diffusion

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Quick Start  
### 1. Data Preparation  
Organize the dataset as follows:  
```
data/
├── Avenue/
│   ├── training/frames/
│   └── testing/frames/
└── Shanghaitech/
│   ├── training/frames/
│   └── testing/frames/
└── UBnormal/
    ├── training/frames/
    └── testing/frames/
```
### 2. Group-supervised Normal Reresentation Decompose Model
```bash
# Data Preprocessing
python ./pre_process/extract_sample.py 
```
### 3. Conditional Model   
```bash
# Setting the model on Avenue dataset 
python ./diffusion_openai/conditional_diffusion.py 

```

### 4. Conditional Diffusion Model  
```bash
# Evaluate the model on the test set
python ./scripts/video_sample_AUC.py  --model_path model/cond_diffusion.pth
```


## 📋 TODO LIST  
- [x] Release core code  
- [ ] Release full code 

## Acknowledgments
We would like to thank zhianliu for open-sourcing the data preprocessing module, and openai for the foundational models:
- [DDIM](https://github.com/openai/improved-diffusion) - The fundamental framework for diffusion models (MIT License).
- [HF2VAD](https://github.com/LiUzHiAn/hf2vad) - A baseline model for video anomaly detection. We appreciate the inspiration from its data preprocessing module.

We especially thank the original authors for their contributions! If there are any copyright issues, please contact us for deletion.

## 💬 Contact Us  
For any questions or suggestions, please contact us via:  
- Email: gcy_SHU@shu.edu.cn  
