## ✅ Requirements

To set up the environment:

```bash
conda env create -f environment.yml
conda activate dino-ldm
```
---

## 🧠 Model Training

### 🔹 Representation-Conditioned Diffusion Model (DINOv2 and unCLIP)

Train with DINOv2 or CLIP representations:

```bash
accelerate launch rep-ldm.py --config="/path/to/config_file/.yaml"
```

> 📁 **Note**: All model configs are stored in `./configs/` and structured by dataset names.
---

## 🎨 Sampling

### Generate data
```bash
python3 sample.py --model_path="checkpoint/dinov2-ldm-unet-cifar-10-subset_final_full_model.pth" --rep_dir="cifar10_subset/rep/test/"
```
---
