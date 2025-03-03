import argparse
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_metrics(evaluation_dirs):
    metrics = {}
    for model_name, eval_dir in evaluation_dirs.items():
        eval_file = os.path.join(eval_dir, "evaluation.json")
        if os.path.exists(eval_file):
            logging.info(f"Loading metrics for {model_name} from {eval_file}")
            with open(eval_file, "r") as f:
                metrics[model_name] = json.load(f)
        else:
            logging.warning(f"Evaluation file {eval_file} does not exist for {model_name}")
    return metrics

def compare_models(evaluation_dirs, output_path):
    logging.info("Starting model comparison")
    metrics = load_metrics(evaluation_dirs)
    
    if not metrics:
        logging.error("No metrics loaded. Exiting.")
        return
    
    # Rank models by highest F1-score
    best_model = max(metrics, key=lambda m: metrics[m]["f1_score"])
    logging.info(f"Best model determined: {best_model}")
    
    comparison_results = {
        "models": metrics,
        "best_model": best_model
    }
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "model_comparison.json")
    with open(output_file, "w") as f:
        json.dump(comparison_results, f)
    logging.info(f"Comparison results saved to {output_file}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--model-names", type=str, nargs="+", required=True)
    parser.add_argument("--output-path", type=str, required=True)
    
    args = parser.parse_args()
    evaluation_dirs = dict(zip(args.model_names, args.evaluation_dirs))
    logging.info(f"Evaluation directories: {evaluation_dirs}")
    logging.info(f"Output path: {args.output_path}")
    compare_models(evaluation_dirs, args.output_path)
