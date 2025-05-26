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
python group_supervised_data.py --config configs/avenue.yaml

# Normal representation-decomposed model Pretraining
python train.py --config configs/avenue.yaml
```
### 3. Data Processing  
```bash
# Train the model on Avenue dataset
python train.py --config configs/avenue.yaml

# Train the model on ShanghaiTech dataset
python train.py --config configs/shanghaitech.yaml
```

### 4. Training Difffusion Model   
```bash
# Train the model on ShanghaiTech dataset
python train.py --config configs/shanghaitech.yaml
```

### 3. Model Evaluation  
```bash
# Evaluate the model on the test set
python evaluate.py --config configs/avenue.yaml --checkpoint checkpoints/best_model.pth
```

## 📋 TODO LIST  
- [x] Release core code  
- [ ] Upload pre-trained models  
- [ ] Add detailed documentation and tutorials  
- [ ] Support multi-GPU training  
- [ ] Optimize inference speed  


## 💬 Contact Us  
For any questions or suggestions, please contact us via:  
- Email: gcy_SHU@shu.edu.cn  
