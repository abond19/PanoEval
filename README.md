# PanoEval

A comprehensive evaluation framework for panoramic image generation models.

## Features

- Multiple evaluation metrics specifically designed for panoramic images:
  - FID (Fréchet Inception Distance)
  - KID (Kernel Inception Distance)
  - Inception Score (IS)
  - CLIP Score
  - FAED (Fréchet AutoEncoder Distance)
  - OmniFID (Panorama-specific FID)
  - Discontinuity Score

## Installation

```bash
pip install git+https://github.com/abond19/PanoEval.git
```

## Directory Structure

Your directories should be organized as follows:

```
gen_dir/                           # Generated images directory
├── image1.jpg
├── image2.jpg
└── ...

real_dir/                         # Real images directory (if needed)
├── real1.jpg
├── real2.jpg
└── ...

prompt_dir/                       # Text prompts directory (if needed)
├── image1.txt                    # Must match generated image names (IMPORTANT)
├── image2.txt
└── ...
```

**Important**: For CLIP score evaluation, text prompt filenames must match their corresponding generated image filenames (excluding extension).

## Usage

### Command Line Arguments

```bash
python evaluate.py [--gen_dir GEN_DIR]
                  [--real_dir REAL_DIR]
                  [--prompt_dir PROMPT_DIR]
                  [--output OUTPUT]
                  [--desired_metrics METRICS]
```

- `--gen_dir`: (Required) Directory containing generated panoramic images
- `--real_dir`: (Optional) Directory containing real panoramic images (required for FID, KID, FAED, and OmniFID)
- `--prompt_dir`: (Optional) Directory containing text prompts (required for CLIP Score)
- `--output`: (Optional) Output CSV file path [default: panorama_metrics.csv]
- `--desired_metrics`: (Optional) Comma-separated list of metrics [default: all]

### Available Metrics

| Metric    | Flag      | Description                    | Requirements          |
|-----------|-----------|--------------------------------|-----------------------|
| FID       | `fid`     | Fréchet Inception Distance     | real_dir              |
| KID       | `kid`     | Kernel Inception Distance      | real_dir              |
| IS        | `is`      | Inception Score                | -                     |
| CLIP      | `clip`    | CLIP Score                     | prompt_dir            |
| FAED      | `faed`    | Fréchet AutoEncoder Distance   | real_dir              |
| OmniFID   | `omnifid` | Panorama-specific FID          | real_dir              |
| DS        | `ds`      | Discontinuity Score            | -                     |

### Examples

1. **Evaluate All Metrics**
```bash
python evaluate.py --gen_dir ./generated_images \
                  --real_dir ./real_images \
                  --prompt_dir ./prompts
```

2. **Evaluate Specific Metrics**
```bash
python evaluate.py --gen_dir ./generated_images \
                  --real_dir ./real_images \
                  --desired_metrics fid,kid,omnifid
```

3. **Metrics Not Requiring Real Images**
```bash
python evaluate.py --gen_dir ./generated_images \
                  --desired_metrics is,ds
```

4. **Single Metric**
```bash
python evaluate.py --gen_dir ./generated_images \
                  --prompt_dir ./prompts \
                  --desired_metrics clip
```

## FAED Metric

The implementation of the FAED metric in this project is based on and inspired by the [PanFusion](https://github.com/chengzhag/PanFusion) project: "Taming Stable Diffusion for Text to 360° Panorama Image Generation" by Cheng Zhang et al. Much gratitude for their significant contributions to the field and their commitment to open-source collaboration.

**Important**: For FAED evaluation, which requires an autoencoder trained on panorama-specific images, you can either train your own model or download a pre-trained checkpoint (`faed.ckpt`) from the [PanFusion GitHub repository](https://github.com/chengzhag/PanFusion) and place it in the `/weights` directory. Don't forget to name it as `faed.ckpt`.

## Citation

```bibtex
@misc{panoeval2024,
  author = {Emir Göcen},
  title = {PanoEval: A Comprehensive Evaluation Framework for Panoramic Image Generation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/emirgocen03/PanoEval}
}
```

## License

do what you wanna dooooo but cite me 

༼ ͡ಠ ͜ʖ ͡ಠ ༽