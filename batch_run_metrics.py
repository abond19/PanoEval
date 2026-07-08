#!/usr/bin/env python3
"""
Complete batch pipeline for arranging and evaluating panoramic images

This script combines both the image arrangement and metric evaluation steps
into a single pipeline that can process multiple image sets at once.
"""

import os
import sys
import csv
import tqdm
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from panoeval.evaluate import evaluate_all_metrics

def arrange_generated_images(input_dir: str, output_dir: str):
    """
    Arrange generated images from input_dir into organized structure in output_dir
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{output_dir}/matterport", exist_ok=True)
    os.makedirs(f"{output_dir}/flickr360", exist_ok=True)
    os.makedirs(f"{output_dir}/polyhaven", exist_ok=True)
    
    # Process files
    files = [f for f in os.listdir(input_dir) if f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".jpg")]
    if not files:
        print(f"Warning: No image files found in {input_dir}")
        return
    
    pbar = tqdm.tqdm(files, desc=f"Arranging {os.path.basename(input_dir)}", unit="file")
    
    copied_count = 0
    for file in pbar:
        # Skip certain files
        if "_0_0" in file or "minecraft" in file or "painting" in file:
            continue
            
        num_dashes = file.count("_")
        file_name = file.split(".")[0]
        
        try:
            if num_dashes == 1:
                # This is a flickr360 image
                os.makedirs(f"{output_dir}/flickr360/{file_name}", exist_ok=True)
                os.system(f"cp '{input_dir}/{file}' '{output_dir}/flickr360/{file_name}/{file}'")
                copied_count += 1
                
            elif num_dashes == 2:
                # This is a matterport image
                os.makedirs(f"{output_dir}/matterport/{file_name}", exist_ok=True)
                os.system(f"cp '{input_dir}/{file}' '{output_dir}/matterport/{file_name}/{file}'")
                copied_count += 1
                
            else:
                # This is a polyhaven image
                os.makedirs(f"{output_dir}/polyhaven/{file_name}", exist_ok=True)
                os.system(f"cp '{input_dir}/{file}' '{output_dir}/polyhaven/{file_name}/{file}'")
                copied_count += 1
                
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    print(f"Arranged {copied_count} images from {input_dir}")

def evaluate_single_directory(
    gen_dir: str,
    real_dir: Optional[str] = None,
    prompt_dir: Optional[str] = None,
    desired_metrics: List[str] = ["fid", "kid", "is", "clip", "faed", "omnifid", "ds", "tangentfid", "tangentis"],
    longer_captions: bool = False,
    use_matterport: bool = False
) -> Dict[str, Any]:
    """
    Evaluate metrics for a single directory
    """
    print(f"Evaluating directory: {gen_dir}")
    
    try:
        results = evaluate_all_metrics(
            gen_dir=gen_dir,
            real_dir=real_dir,
            prompt_dir=prompt_dir,
            output_file=None,  # We'll handle CSV output ourselves
            desired_metrics=desired_metrics,
            longer_captions=longer_captions,
            use_matterport=use_matterport
        )
        
        # Add path information to results
        results['gen_dir'] = gen_dir
        results['real_dir'] = real_dir if real_dir else 'N/A'
        results['prompt_dir'] = prompt_dir if prompt_dir else 'N/A'
        
        return results
        
    except Exception as e:
        print(f"Error evaluating {gen_dir}: {str(e)}")
        return {
            'gen_dir': gen_dir,
            'real_dir': real_dir if real_dir else 'N/A',
            'prompt_dir': prompt_dir if prompt_dir else 'N/A',
            'error': str(e)
        }

def save_results_to_csv(results: Dict[str, Any], output_file: str):
    """
    Save individual results to a CSV file
    """
    df = pd.DataFrame([results])
    
    # Reorder columns to put path information first
    path_cols = ['gen_dir', 'real_dir', 'prompt_dir']
    other_cols = [col for col in df.columns if col not in path_cols]
    df = df[path_cols + other_cols]
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

def save_batch_results_to_csv(all_results: List[Dict[str, Any]], output_file: str):
    """
    Save all batch results to a single CSV file
    """
    if not all_results:
        print("No results to save")
        return
    
    df = pd.DataFrame(all_results)
    
    # Reorder columns to put path information first
    path_cols = ['gen_dir', 'real_dir', 'prompt_dir']
    other_cols = [col for col in df.columns if col not in path_cols]
    df = df[path_cols + other_cols]
    
    df.to_csv(output_file, index=False)
    print(f"Batch results saved to: {output_file}")

def run_complete_pipeline(
    input_dirs: List[str],
    arranged_dirs: List[str],
    real_dir: Optional[str] = None,
    prompt_dir: Optional[str] = None,
    output_dir: str = "batch_metrics_output",
    batch_output_file: str = "batch_panorama_metrics.csv",
    desired_metrics: List[str] = ["fid", "kid", "is", "clip", "faed", "omnifid", "ds", "tangentfid", "tangentis"],
    longer_captions: bool = False,
    use_matterport: bool = False,
    save_individual: bool = True,
    arrange_only: bool = False,
    evaluate_only: bool = False
):
    """
    Run the complete pipeline: arrange images and evaluate metrics
    
    Args:
        input_dirs: List of input directories containing raw generated images
        arranged_dirs: List of output directories for arranged images
        real_dir: Path to real images directory
        prompt_dir: Path to text prompts directory
        output_dir: Directory to save individual CSV files
        batch_output_file: Path for combined batch results CSV
        desired_metrics: List of metrics to compute
        longer_captions: Use longer captions for CLIP Score
        use_matterport: Use Matterport dataset
        save_individual: Save individual CSV files for each directory
        arrange_only: Only run the arrangement step
        evaluate_only: Only run the evaluation step (assumes images are already arranged)
    """
    
    if len(input_dirs) != len(arranged_dirs):
        raise ValueError("Input and arranged directory lists must have the same length")
    
    print("="*60)
    print("PANORAMIC IMAGE EVALUATION PIPELINE")
    print("="*60)
    print(f"Processing {len(input_dirs)} image sets")
    print(f"Real dir: {real_dir}")
    print(f"Prompt dir: {prompt_dir}")
    if not arrange_only:
        print(f"Desired metrics: {', '.join(desired_metrics)}")
    print("="*60)
    
    # Step 1: Arrange images (if not evaluate_only)
    if not evaluate_only:
        print("\nSTEP 1: ARRANGING IMAGES")
        print("-" * 30)
        
        for i, (input_dir, arranged_dir) in enumerate(zip(input_dirs, arranged_dirs)):
            print(f"\n[{i+1}/{len(input_dirs)}] Arranging: {input_dir} -> {arranged_dir}")
            
            if not os.path.exists(input_dir):
                print(f"Warning: Input directory {input_dir} does not exist. Skipping...")
                continue
                
            try:
                arrange_generated_images(input_dir, arranged_dir)
            except Exception as e:
                print(f"Error arranging {input_dir}: {str(e)}")
        
        print("\nImage arrangement complete!")
    
    # Step 2: Evaluate metrics (if not arrange_only)
    if not arrange_only:
        print("\nSTEP 2: EVALUATING METRICS")
        print("-" * 30)
        
        # Create output directory
        if save_individual:
            os.makedirs(output_dir, exist_ok=True)
        
        all_results = []
        
        for i, arranged_dir in enumerate(arranged_dirs):
            print(f"\n[{i+1}/{len(arranged_dirs)}] Evaluating: {arranged_dir}")
            
            if not os.path.exists(arranged_dir):
                print(f"Warning: Directory {arranged_dir} does not exist. Skipping...")
                continue
            
            # Evaluate this directory
            results = evaluate_single_directory(
                gen_dir=arranged_dir,
                real_dir=real_dir,
                prompt_dir=prompt_dir,
                desired_metrics=desired_metrics,
                longer_captions=longer_captions,
                use_matterport=use_matterport
            )
            
            all_results.append(results)
            
            # Save individual results if requested
            if save_individual:
                safe_name = os.path.basename(arranged_dir.rstrip('/'))
                if not safe_name:
                    safe_name = f"results_{i+1}"
                individual_output = os.path.join(output_dir, f"{safe_name}_metrics.csv")
                save_results_to_csv(results, individual_output)
        
        # Save combined batch results
        if all_results:
            save_batch_results_to_csv(all_results, batch_output_file)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    if not evaluate_only:
        print(f"✓ Arranged images for {len(input_dirs)} directories")
    if not arrange_only:
        if 'all_results' in locals():
            print(f"✓ Evaluated {len(all_results)} directories")
            if save_individual:
                print(f"✓ Individual results saved in: {output_dir}")
            print(f"✓ Batch results saved to: {batch_output_file}")
    print("="*60)

if __name__ == "__main__":
    # =================================================================
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR SPECIFIC SETUP
    # =================================================================
    
    # Input directories containing raw generated images
    input_dirs = [
        # "/home/hcapuk20/stablediffusion3/REBUTTAL_EXPERIMENTS/optim_grid/final_erp",
        # "/home/hcapuk20/stablediffusion3/REBUTTAL_EXPERIMENTS/optim_grid_700/final_erp",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/rebuttal_optim_ablation_wo_circpad/700_final_erps",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/rebuttal_optim_ablation_wo_latrot/700_final_erps",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/rebuttal_optim_ablation_wo_patchdenoise/700_final_erps",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/rebuttal_optim_ablation_wo_sr/700_final_erps",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/SD3_MEDIUM_LONG/700_final_erps",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/SD3_MEDIUM_SUMMARIZED/700_final_erps",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/SD3_BASE_FINETUNED/grids",
        # "/home/hcapuk20/baselines/par/par/results/trained_summarized",
        # "/home/hcapuk20/baselines/unipano/UniPano/UniPano_SD3/trained_medium_Longer",
        # "/home/hcapuk20/baselines/unipano/UniPano/UniPano_SD3/trained_medium_s",
        # "/home/hcapuk20/baselines/stitchdiffusion/stitchdiffusion-colab/longer_caption_outputs",
        # "/home/hcapuk20/baselines/stitchdiffusion/stitchdiffusion-colab/outputs",
        # "/home/hcapuk20/baselines/Diffusion360/SD-T2I-360PanoImage/outputs",
        # "/home/hcapuk20/baselines/multidiffusion/outputs",
        # "/home/hcapuk20/baselines/multidiffusion/longer_caption_outputs",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/summarized_optim_grid/600_final_erp",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/summarized_optim_grid/700_final_erp",
        # "/home/hcapuk20/stablediffusion3/ICLR_experiments/summarized_optim_grid/800_final_erp",
        "/home/hcapuk20/baselines/unipano/UniPano/UniPano_SD3/trained_medium_Longer",
        "/home/hcapuk20/baselines/unipano/UniPano/UniPano_SD3/trained_medium_L",   
    ]
    
    # Output directories for arranged images (corresponds to input_dirs)
    arranged_dirs = [
        # "./all_arranged_images/optim_grid_800_final/",
        # "./all_arranged_images/optim_grid_700_final/",
        # "./all_arranged_images/optim_ablation_wo_circpad/",
        # "./all_arranged_images/optim_ablation_wo_latrot/",
        # "./all_arranged_images/optim_ablation_wo_patchdenoise/",
        # "./all_arranged_images/optim_ablation_wo_sr/",
        # "./all_arranged_images/sd3_medium_long/",
        # "./all_arranged_images/sd3_medium_summarized/",
        # "./all_arranged_images/sd3_base_finetuned/",
        # "./all_arranged_images/par_trained_summarized/",
        # "./all_arranged_images/unipano_sd3_trained_medium_longer/",
        # "./all_arranged_images/unipano_sd3_trained_medium_s/",
        # "./all_arranged_images/stitchdiffusion_longer_captions/",
        # "./all_arranged_images/stitchdiffusion_regular/",
        # "./all_arranged_images/diff360_regular/",
        # "./all_arranged_images/multidiff_regular/",
        # "./all_arranged_images/multidiff_longer_captions/"
        # "/home/hcapuk20/baselines/stitchdiffusion/stitchdiffusion-colab/longer_caption_outputs",
        # "/home/hcapuk20/baselines/stitchdiffusion/stitchdiffusion-colab/outputs",
        # "/home/hcapuk20/baselines/Diffusion360/SD-T2I-360PanoImage/outputs",
        # "/home/hcapuk20/baselines/multidiffusion/outputs",
        # "/home/hcapuk20/baselines/multidiffusion/longer_caption_outputs",
        # "./all_arranged_images/summarized_optim_grid_600",
        # "./all_arranged_images/summarized_optim_grid_700",
        # "./all_arranged_images/summarized_optim_grid_800",
        "./all_arranged_images/unipano_trained_medium_longer",
        "./all_arranged_images/unipano_trained_medium_L"
    ]
    
    # These directories remain constant across all evaluations
    real_dir = "/home/hcapuk20/baselines/test_gts/"
    prompt_dir = "/home/hcapuk20/baselines/summarized_captions/"
    
    # Evaluation configuration
    desired_metrics = ["fid", "kid", "is", "clip", "faed", "omnifid", "ds", "tangentfid", "tangentis"]
    
    # =================================================================
    # RUN THE PIPELINE
    # =================================================================
    
    # Uncomment ONE of the following options:
    
    # Option 1: Run complete pipeline (arrange + evaluate)
    run_complete_pipeline(
        input_dirs=input_dirs,
        arranged_dirs=arranged_dirs,
        real_dir=real_dir,
        prompt_dir=prompt_dir,
        output_dir="batch_metrics_output",
        batch_output_file="batch_panorama_metrics.csv",
        desired_metrics=desired_metrics,
        longer_captions=False,
        use_matterport=True,
        save_individual=True
    )
    
    # Option 2: Only arrange images
    # run_complete_pipeline(
    #     input_dirs=input_dirs,
    #     arranged_dirs=arranged_dirs,
    #     arrange_only=True
    # )
    
    # Option 3: Only evaluate metrics (assumes images are already arranged)
    # run_complete_pipeline(
    #     input_dirs=input_dirs,
    #     arranged_dirs=arranged_dirs,
    #     real_dir=real_dir,
    #     prompt_dir=prompt_dir,
    #     output_dir="batch_metrics_output",
    #     batch_output_file="batch_panorama_metrics.csv",
    #     desired_metrics=desired_metrics,
    #     longer_captions=False,
    #     use_matterport=False,
    #     save_individual=True,
    #     evaluate_only=True
    # )
    
    print("Please modify the paths in the configuration section and uncomment one of the run options.")
    print("\nAvailable options:")
    print("1. Complete pipeline (arrange + evaluate)")
    print("2. Arrange images only")
    print("3. Evaluate metrics only")