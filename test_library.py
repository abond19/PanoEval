from panoeval.evaluate import evaluate_all_metrics
import argparse
import os

os.environ["TRANSMORERS_CACHE"] = "1"
os.environ["RETURN_DICT"] = "False"

parser = argparse.ArgumentParser(description="Evaluate panoramic image generation quality")
parser.add_argument("--gen_dir", type=str, required=True, help="Path to generated images directory")
parser.add_argument("--real_dir", type=str, default=None, help="Path to real images directory")
parser.add_argument("--prompt_dir", type=str, default=None, help="Path to text prompts directory")
parser.add_argument("--output", type=str, default="panorama_metrics.csv", help="Output CSV file path")
parser.add_argument("--desired_metrics", type=str, default="fid,kid,is,clip,faed,omnifid,ds,tangentfid,tangentis", 
                    help="Comma-separated list of metrics to compute")
parser.add_argument("--longer_captions", default=False, type=bool, 
                    help="Use longer captions for CLIP Score (default: False)")
parser.add_argument("--use_matterport", default=False, type=bool)

args = parser.parse_args()

evaluate_all_metrics(
    gen_dir=args.gen_dir,
    real_dir=args.real_dir,
    prompt_dir=args.prompt_dir,
    output_file=args.output,
    desired_metrics=args.desired_metrics.split(","),
    longer_captions=args.longer_captions,
    use_matterport=args.use_matterport
)