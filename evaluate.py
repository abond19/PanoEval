import os
import pandas as pd
from typing import Optional, Dict, Any, List
from metrics.fid import compute_fid
from metrics.kid import compute_kid
from metrics.inception_score import compute_inception_score
from metrics.clip_score import compute_clip_score
from metrics.faed import compute_faed
from metrics.omnifid import compute_omnifid
from metrics.discontinuity_score import compute_discontinuity_score
from metrics.tangent_fid import compute_tangentfid
from utils.dataloader import load_images, load_text_prompts

import wandb

def evaluate_all_metrics(
    gen_dir: str,
    real_dir: Optional[str] = None,
    prompt_dir: Optional[str] = None,
    output_file: str = "panorama_metrics.csv",
    desired_metrics: Optional[List[str]] = ["fid", "kid", "is", "clip", "faed", "omnifid", "ds", "tangentfid"],
    use_wandb: bool = False
) -> Dict[str, float]:
    """
    Evaluate generated panoramic images using multiple metrics.
    
    Args:
        gen_dir (str): Path to directory containing generated panoramic images
        real_dir (str, optional): Path to directory containing real panoramic images
        prompt_dir (str, optional): Path to directory containing text prompts for CLIP score evaluation
        output_file (str): Path to save the evaluation results (CSV format)
        desired_metrics (List[str], optional): List of metrics to compute. Available options:
            - 'fid': Fréchet Inception Distance (requires real_dir)
            - 'kid': Kernel Inception Distance (requires real_dir)
            - 'is': Inception Score
            - 'clip': CLIP Score (requires prompt_dir)
            - 'faed': Fréchet AutoEncoder Distance (requires real_dir)
            - 'omnifid': OmniFID (requires real_dir)
            - 'ds': Discontinuity Score
    
    Returns:
        Dict[str, float]: Dictionary containing computed metric scores
    
    Raises:
        FileNotFoundError: If specified directories don't exist
        ValueError: If required directories are missing for selected metrics
    """   
    # Check if generated images directory exists
    if not os.path.exists(gen_dir):
        raise FileNotFoundError(f"Generated images directory not found: {gen_dir}")
    
    # Check if text prompts are provided for CLIP score evaluation
    if prompt_dir is None and "clip" in desired_metrics:
        raise ValueError("Text prompts directory (prompt_dir) is required for CLIP score evaluation.")

    # Check if metrics that require real images are provided
    # real_images = None
    # if any(metric in desired_metrics for metric in ["fid", "kid", "faed", "omnifid"]):
    #     if real_dir is None:
    #         raise ValueError("Real images directory (real_dir) is required for FID, KID, FAED, or OmniFID evaluation.")
    #     if not os.path.exists(real_dir):
    #         raise FileNotFoundError(f"Real images directory not found: {real_dir}")
    #     real_images = load_images(real_dir)
    #     print(f"Loaded {len(real_images)} real images.")

    # Load generated images
    # gen_images = load_images(gen_dir)
    # print(f"Loaded {len(gen_images)} generated images.")

    # Load text prompts if needed
    # text_prompts = None
    # if prompt_dir is not None:
    #     if not os.path.exists(prompt_dir):
    #         raise FileNotFoundError(f"Text prompts directory not found: {prompt_dir}")
    #     text_prompts = load_text_prompts(prompt_dir)
    #     print(f"Loaded {len(text_prompts)} text prompts.")

    # text_prompts is a list of text prompts, real_images is a list of PIL images, and gen_images is a list of PIL images
    # since we sort the files in the folders before loading them, the text_prompts, real_images, and gen_images are in the same order and have the same length

    # Initialize results dictionary
    results: Dict[str, float] = {}

    try:
        # Compute all metrics
        if "fid" in desired_metrics:
            print("Computing FID...")
            results["FID"] = compute_fid(real_dir, gen_dir)
        
        if "kid" in desired_metrics:
            print("Computing KID...")
            results["KID"] = compute_kid(real_dir, gen_dir)
        
        if "is" in desired_metrics:
            print("Computing Inception Score...")
            results["IS"] = compute_inception_score(gen_dir)
        
        if "clip" in desired_metrics:
            print("Computing CLIP Score...")
            results["CLIP Score"] = compute_clip_score(gen_dir, prompt_dir)
        
        if "faed" in desired_metrics:
            print("Computing FAED...")
            results["FAED"] = compute_faed(real_dir, gen_dir)
        
        if "omnifid" in desired_metrics:
            print("Computing OmniFID...")
            results["OmniFID"] = compute_omnifid(real_dir, gen_dir)
        
        if "ds" in desired_metrics:
            print("Computing Discontinuity Score...")
            results["Discontinuity Score"] = compute_discontinuity_score(gen_dir)

        if "tangentfid" in desired_metrics:
            print("Computing TangentFID...")
            results["TangentFID"] = compute_tangentfid(real_dir, gen_dir)

        if use_wandb:
            # Initialize Weights & Biases
            wandb.init(project="panorama_metrics", config={"gen_dir": gen_dir, "real_dir": real_dir, "prompt_dir": prompt_dir})
            wandb.log(results)
            wandb.finish()
        
        # Save results to CSV
        df = pd.DataFrame([results])
        df.to_csv(output_file, index=False)
        print(f"✅ Evaluation complete. Results saved to {output_file}")
        print(f"Results: {results}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate panoramic image generation quality")
    parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated images directory")
    parser.add_argument("--real_dir", type=str, default=None, help="Path to real images directory")
    parser.add_argument("--prompt_dir", type=str, default=None, help="Path to text prompts directory")
    parser.add_argument("--output", type=str, default="panorama_metrics.csv", help="Output CSV file path")
    parser.add_argument("--desired_metrics", type=str, default="fid,kid,is,clip,faed,omnifid,ds,tangentfid", 
                       help="Comma-separated list of metrics to compute")
    parser.add_argument("--use_wandb", type=bool, default=False, help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    evaluate_all_metrics(
        gen_dir=args.gen_dir,
        real_dir=args.real_dir,
        prompt_dir=args.prompt_dir,
        output_file=args.output,
        desired_metrics=args.desired_metrics.split(","),
        use_wandb=args.use_wandb
    )
