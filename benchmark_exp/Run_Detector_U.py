# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import random, argparse, time, os, logging, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.evaluation.basic_metrics import basic_metricor
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

try:
    import torch
except ModuleNotFoundError:
    torch = None

# seeding
seed = 2024
np.random.seed(seed)
random.seed(seed)
if torch is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("CUDA available: ", torch.cuda.is_available())
    print("cuDNN version: ", torch.backends.cudnn.version())
else:
    print("PyTorch available: False")


def compute_selected_metrics(score, label):
    grader = basic_metricor()
    return {
        'F1': float(grader.metric_PointF1(label, score)),
        'ROC-AUC': float(grader.metric_ROC(label, score)),
        'ECE': float(grader.metric_ECE(label, score, n_bins=10, clip=True, from_raw_score=True)),
    }


def summarize_selected_metrics(metric_history):
    summary = {}
    for metric_name in ['F1', 'ROC-AUC', 'ECE']:
        values = metric_history.get(metric_name, [])
        summary[metric_name] = float(np.nanmean(values)) if values else float('nan')
    return summary

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='./Datasets/TSB-AD-U/')
    parser.add_argument('--file_list', type=str, default='./Datasets/File_List/no_seq_anomaly_files.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--force', action='store_true', help='Recompute scores even if cached .npy files already exist.')
    parser.add_argument('--max_files', type=int, default=None, help='Optional limit for quick smoke tests.')
    parser.add_argument('--AD_Name', nargs='+', default=['IForest'])
    args = parser.parse_args()

    requested_models = args.AD_Name if isinstance(args.AD_Name, list) else [args.AD_Name]
    resolved_models = []
    for requested_name in requested_models:
        canonical_name = resolve_model_name(requested_name)
        if canonical_name is None:
            available = ', '.join(get_supported_model_names())
            raise ValueError(f"Unknown AD_Name '{requested_name}'. Available models: {available}")
        resolved_models.append(canonical_name)

    file_list = pd.read_csv(args.file_list)['file_name'].dropna().astype(str).tolist()
    if args.max_files is not None:
        file_list = file_list[:args.max_files]

    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)

    for ad_name in resolved_models:
        target_dir = os.path.join(args.score_dir, ad_name)
        os.makedirs(target_dir, exist_ok=True)
        logging.basicConfig(
            filename=f'{target_dir}/000_run_{ad_name}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True,
        )

        optimal_det_hp = Optimal_Uni_algo_HP_dict.get(ad_name, {})
        print(f'Running {ad_name} on {len(file_list)} files from {args.file_list}')
        print('Optimal_Det_HP: ', optimal_det_hp)

        metric_history = {'F1': [], 'ROC-AUC': [], 'ECE': []}
        write_csv = []

        for filename in file_list:
            print(f'Processing: {filename} by {ad_name}')

            file_path = os.path.join(args.dataset_dir, filename)
            df = pd.read_csv(file_path).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df['Label'].astype(int).to_numpy()
            slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
            train_index = filename.split('.')[0].split('_')[-3]
            data_train = data[:int(train_index), :]

            score_path = os.path.join(target_dir, filename.split('.')[0] + '.npy')
            output = None
            run_time = 0.0

            if os.path.exists(score_path) and not args.force:
                output = np.load(score_path)
                logging.info(f'Loaded cached score for {filename} using {ad_name}')
            else:
                start_time = time.time()

                if ad_name in Semisupervise_AD_Pool:
                    output = run_Semisupervise_AD(ad_name, data_train, data, **optimal_det_hp)
                elif ad_name in Unsupervise_AD_Pool:
                    output = run_Unsupervise_AD(ad_name, data, **optimal_det_hp)
                else:
                    raise Exception(f"{ad_name} is not defined")

                run_time = time.time() - start_time

                if isinstance(output, np.ndarray):
                    logging.info(f'Success at {filename} using {ad_name} | Time cost: {run_time:.3f}s at length {len(label)}')
                    np.save(score_path, output)
                else:
                    logging.error(f'At {filename}: {output}')
                    continue

            try:
                output = np.asarray(output).ravel()
                selected_metrics = compute_selected_metrics(output, label)
                for metric_name, metric_value in selected_metrics.items():
                    metric_history[metric_name].append(metric_value)

                print(
                    f"{ad_name} | {filename} | "
                    f"F1={selected_metrics['F1']:.4f}, "
                    f"ROC-AUC={selected_metrics['ROC-AUC']:.4f}, "
                    f"ECE={selected_metrics['ECE']:.4f}"
                )

                if args.save:
                    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                    row = {'file': filename, 'Time': run_time, **evaluation_result}
                    write_csv.append(row)
                    pd.DataFrame(write_csv).to_csv(f'{args.save_dir}/{ad_name}.csv', index=False)
            except Exception as exc:
                logging.exception(f'Failed to evaluate {filename} using {ad_name}: {exc}')
                print(f'Failed to evaluate {filename} by {ad_name}: {exc}')

        summary = summarize_selected_metrics(metric_history)
        processed_files = len(metric_history['F1'])
        print(
            f"{ad_name} mean metrics across {processed_files} files: "
            f"F1={summary['F1']:.4f}, "
            f"ROC-AUC={summary['ROC-AUC']:.4f}, "
            f"ECE={summary['ECE']:.4f}"
        )

    print(f'Total elapsed time: {time.time() - Start_T:.2f}s')
