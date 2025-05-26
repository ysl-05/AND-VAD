# Aligning Normal Representations in Diffusion for VAD （AND-VAD）
This repository contains the official implementation of the paper "Aligning Normal Representations in Diffusion Model for Video Anomaly Detection". We propose an unsupervised video anomaly detection framework based on conditional diffusion models, which leverages **normal representation alignment** and **group-supervised learning strategies** to guide the diffusion model to learn semantic patterns of normal behaviors, thereby enhancing the discrimination ability for anomalies.


## 🔧 Environment Setup  
```bash
# Create a virtual environment (optional)
conda create -n vad_diffusion python=3.9
conda activate vad_diffusion

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Quick Start  
### 1. Data Preparation  
Organize the dataset as follows:  
```
data/
├── avenue/
│   ├── training/frames/
│   └── testing/frames/
└── ped2/
    ├── training/frames/
    └── testing/frames/
```

### 2. Model Training  
```bash
# Train the model on Avenue dataset
python train.py --config configs/avenue.yaml

# Train the model on ShanghaiTech dataset
python train.py --config configs/shanghaitech.yaml
```

### 3. Model Evaluation  
```bash
# Evaluate the model on the test set
python evaluate.py --config configs/avenue.yaml --checkpoint checkpoints/best_model.pth
```

### 4. Visualize Detection Results  
```bash
# Generate visualization results for anomaly detection
python visualize.py --config configs/avenue.yaml --checkpoint checkpoints/best_model.pth --video_idx 0
```


## 📋 TODO LIST  
- [x] Release core code  
- [ ] Upload pre-trained models  
- [ ] Add detailed documentation and tutorials  
- [ ] Support multi-GPU training  
- [ ] Optimize inference speed  


## 📜 Citation  
If you use this code or model in your research, please cite our paper:  
```bibtex
@article{your_paper,
  title={Aligning Normal Representations in Diffusion Model for Video Anomaly Detection},
  author={Your Name, et al.},
  journal={arXiv preprint arXiv:2307.00000},
  year={2023}
}
```


## 💬 Contact Us  
For any questions or suggestions, please contact us via:  
- Email: your_email@example.com  
- GitHub Issues: [Submit an issue](https://github.com/your_username/your_repo/issues)
