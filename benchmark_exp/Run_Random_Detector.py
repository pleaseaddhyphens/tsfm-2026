# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore

class RandomDetector(BaseDetector):

    def __init__(self, HP, normalize=True):
        super().__init__()
        self.HP = HP
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        For the random baseline, `fit` simply assigns a random score to each
        sample in `X` and stores it in `decision_scores_`, matching the
        unsupervised interface used in the TSB-AD framework.
        """
        n_samples, _ = X.shape
        self.decision_scores_ = np.random.rand(n_samples)
        return self

    def decision_function(self, X):
        """This is not used by RandomDetector the BaseDetector requires that method to be implemented.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        n_samples, n_features = X.shape
        scores = np.random.rand(n_samples)
        self.decision_scores_ = scores
        return scores


def run_RandomDetector_Unsupervised(data, HP):
    clf = RandomDetector(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

# semisupervised interface not required in current implementation 
# def run_RandomDetector_Semisupervised(data_train, data_test, HP):
#     clf = RandomDetector(HP=HP)
#     clf.fit(data_train)
#     score = clf.decision_function(data_test)
#     score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
#     return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running RandomDetector')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='RandomDetector')
    args = parser.parse_args()

    RandomDetector_HP = {
        'HP': ['HP'],
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    start_time = time.time()

    # Use the unsupervised variant of the random detector by default
    output = run_RandomDetector_Unsupervised(data, **RandomDetector_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)