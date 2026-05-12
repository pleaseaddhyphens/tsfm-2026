# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import random, argparse, time, os, logging, sys, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TSB_AD.evaluation.basic_metrics import basic_metricor
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict

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
    uncertainty_metrics = grader.metric_uncertainty_suite(
        label, score, n_bins=10, clip=True, from_raw_score=True, pred_threshold=0.5
    )
    return {
        'F1': float(grader.metric_PointF1(label, score)),
        'ROC-AUC': float(grader.metric_ROC(label, score)),
        **uncertainty_metrics,
    }


SELECTED_METRIC_NAMES = ['F1', 'ROC-AUC', 'ECE', 'MCE', 'Adaptive-ECE', 'Brier', 'NLL', 'Sharpness-Std', 'ErrDet-AUROC', 'AURC', 'EAURC']


def summarize_selected_metrics(metric_history):
    summary = {}
    for metric_name in SELECTED_METRIC_NAMES:
        values = metric_history.get(metric_name, [])
        summary[metric_name] = float(np.nanmean(values)) if values else float('nan')
    return summary

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Score')
    parser.add_argument('--dataset_dir', type=str, default='./Datasets/TSB-AD-M/')
    parser.add_argument('--file_lsit', type=str, default='./Datasets/File_List/TSB-AD-M-picked.csv')
    parser.add_argument('--score_dir', type=str, default='eval/score/multi/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/multi/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='IForest')
    parser.add_argument('--run_name', type=str, default='', help='Optional custom suffix for saved metrics/log filenames.')
    parser.add_argument('--file_name', type=str, default='', help='Optional single file name from the file list to process.')
    parser.add_argument('--force', action='store_true', help='Recompute scores even if cached .npy files already exist.')
    args = parser.parse_args()


    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    run_name_suffix = args.run_name.strip()
    if run_name_suffix:
        run_name_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name_suffix)

    log_name = f"000_run_{args.AD_Name}_{run_ts}"
    if run_name_suffix:
        log_name = f"{log_name}_{run_name_suffix}"
    logging.basicConfig(
        filename=f'{target_dir}/{log_name}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    if args.file_name:
        file_list = [args.file_name]
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)

    write_csv = []
    metric_history = {name: [] for name in SELECTED_METRIC_NAMES}
    for filename in file_list:
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        # print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        score_path = target_dir+'/'+filename.split('.')[0]+'.npy'
        output = None
        run_time = 0.0

        if os.path.exists(score_path) and not args.force:
            output = np.load(score_path)
            logging.info(f'Loaded cached score for {filename} using {args.AD_Name}')
        else:
            start_time = time.time()

            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
            else:
                raise Exception(f"{args.AD_Name} is not defined")

            end_time = time.time()
            run_time = end_time - start_time

            if isinstance(output, np.ndarray):
                logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
                np.save(score_path, output)
            else:
                logging.error(f'At {filename}: '+output)
                continue

        ### whether to save the evaluation result
        try:
            output = np.asarray(output).ravel()
            selected_metrics = compute_selected_metrics(output, label)
            for metric_name, metric_value in selected_metrics.items():
                metric_history[metric_name].append(metric_value)

            print(
                f"{args.AD_Name} | {filename} | "
                f"F1={selected_metrics['F1']:.4f}, "
                f"ROC-AUC={selected_metrics['ROC-AUC']:.4f}, "
                f"ECE={selected_metrics['ECE']:.4f}, "
                f"MCE={selected_metrics['MCE']:.4f}, "
                f"AdECE={selected_metrics['Adaptive-ECE']:.4f}, "
                f"Brier={selected_metrics['Brier']:.4f}, "
                f"NLL={selected_metrics['NLL']:.4f}, "
                f"ErrDet-AUROC={selected_metrics['ErrDet-AUROC']:.4f}, "
                f"AURC={selected_metrics['AURC']:.4f}, "
                f"EAURC={selected_metrics['EAURC']:.4f}"
            )

            if args.save:
                row_metrics = {name: selected_metrics.get(name, float('nan')) for name in SELECTED_METRIC_NAMES}
                row = {'file': filename, 'Time': run_time, **row_metrics}
                write_csv.append(row)
                out_name = f"{args.AD_Name}_{run_ts}"
                if run_name_suffix:
                    out_name = f"{out_name}_{run_name_suffix}"
                pd.DataFrame(write_csv).to_csv(f'{args.save_dir}/{out_name}.csv', index=False)
        except Exception as exc:
            logging.exception(f'Failed to evaluate {filename} using {args.AD_Name}: {exc}')
            print(f'Failed to evaluate {filename} by {args.AD_Name}: {exc}')

    summary = summarize_selected_metrics(metric_history)
    processed_files = len(metric_history['F1'])
    print(
        f"{args.AD_Name} mean metrics across {processed_files} files: "
        f"F1={summary['F1']:.4f}, "
        f"ROC-AUC={summary['ROC-AUC']:.4f}, "
        f"ECE={summary['ECE']:.4f}, "
        f"MCE={summary['MCE']:.4f}, "
        f"AdECE={summary['Adaptive-ECE']:.4f}, "
        f"Brier={summary['Brier']:.4f}, "
        f"NLL={summary['NLL']:.4f}, "
        f"ErrDet-AUROC={summary['ErrDet-AUROC']:.4f}, "
        f"AURC={summary['AURC']:.4f}, "
        f"EAURC={summary['EAURC']:.4f}"
    )
