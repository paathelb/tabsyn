import numpy as np
import torch 
import pandas as pd
import os 
import sys

import json
from mle.mle import get_evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataname', type=str, default='adult')
parser.add_argument('--model', type=str, default='real')
parser.add_argument('--path', type=str, default = None, help='The file path of the synthetic data')

args = parser.parse_args()

# def preprocess(train, test, info)

#     def norm_data(data, )

if __name__ == '__main__':
    import pdb; pdb.set_trace()
    dataname = args.dataname
    model = args.model
    
    if not args.path:
        train_path = f'synthetic/{dataname}/{model}.csv'
    else:
        train_path = args.path
    test_path = f'synthetic/{dataname}/test.csv'

    train = pd.read_csv(train_path).to_numpy()
    test = pd.read_csv(test_path).to_numpy()

    with open(f'data/{dataname}/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    evaluator = get_evaluator(task_type)

    # changed by HP
    cls_experiment = False
    if cls_experiment:
        from scipy.stats import norm, wasserstein_distance
        from scipy.spatial.distance import mahalanobis
        from scipy.stats import multivariate_normal
        import matplotlib.pyplot as plt
        import numpy as np
        import numpy as np
        from scipy.stats import norm
        # Assign feature std based on the std from training data
        dummy_std = (train/train.mean(0)).std(0)[info['num_col_idx']]

        # ----------------------------------------------------------
        # classify the test sample

        # Function to calculate Wasserstein distance between two normal distributions
        def calculate_wasserstein_distance(mean1, std_dev1, tabsyn_data):       #(mean2, std_dev2)
            # Generate samples from the two normal distributions
            samples1 = np.random.normal(mean1, std_dev1, 1000) # np.random.normal(loc=mean1, scale=std_dev1, size=(1000, len(mean1))) #
            # samples2 = np.random.normal(mean2, std_dev2, 1000)

            # Calculate the Wasserstein distance
            # w_distance = 0
            # for i in range(samples1.shape[0]):
            w_distance = wasserstein_distance(samples1, tabsyn_data)
            # w_distance /= len(tabsyn_data)

            return w_distance

        def fisher_divergence(mean1, std1, dist2_samples):
            # Calculate the probability density function (PDF) of dist1
            dist1 = norm(loc=mean1, scale=std1)
            
            # Calculate the log likelihoods for each sample in dist2
            log_likelihoods = dist1.logpdf(dist2_samples)
            
            # Calculate the Fisher Divergence
            fisher_divergence = -2 * log_likelihoods.mean()
            
            return fisher_divergence

        
        # ----------------------------------------------------------
        
        # Directly load the generated data instead of the mean and std
        # Specify the path to your CSV file
        csv_file_path = "/home/hpaat/imbalanced_data/tabsyn/synthetic/default/tabsyn.csv"

        # Load the data into a numpy array, skipping the first row
        tabsyn_data = np.loadtxt(csv_file_path, delimiter=',', skiprows=1)
        tabsyn_data_0 = tabsyn_data[tabsyn_data[:,-1]==0][:,info['num_col_idx']].astype(float)
        tabsyn_data_0 /= tabsyn_data.mean(0)[info['num_col_idx']]
        
        final_w_d = []
        import pdb; pdb.set_trace()
        for j in range(len(test)):
            # Get dummy test sample we want to classify either 0 or 1
            test_sample = test[j,info['num_col_idx']]       # idx 0 has label 0, idx 9 has label 1
            test_sample /= test[:,info['num_col_idx']].mean(0)
            
            # Calculate the wasserstein distance for each feature
            w_d = []
            wasserstein, mahalanobis_d, multi_normal=False, False, False 
            fisher = True
            if wasserstein:
                for i in range(len(info['num_col_idx'])):
                    w_d.append(calculate_wasserstein_distance(test_sample[i], dummy_std[i], tabsyn_data_0[:,i]))     #tabsyn_std[i]
            elif mahalanobis_d:
                inv_cov0 = np.linalg.inv(np.cov(tabsyn_data_0, rowvar=False))
                w_d.append(mahalanobis(test_sample, tabsyn_data_0.mean(0), inv_cov0))
            elif fisher:
                for i in range(len(info['num_col_idx'])):
                    w_d.append(fisher_divergence(test_sample[i], dummy_std[i], tabsyn_data_0[:,i]))
            else:
                w_d.append(multivariate_normal.pdf(test_sample, tabsyn_data_0.mean(0), np.cov(tabsyn_data_0, rowvar=False)))
                
            final_w_d.append(w_d)

        import pdb; pdb.set_trace()
        final_w_d = np.array(final_w_d, dtype=np.float32)
        final_w_d_1 = final_w_d[[index for index, value in enumerate(test[:,-1]) if value == 1]] #,:]
        final_w_d_0 = final_w_d[[index for index, value in enumerate(test[:,-1]) if value == 0]] #,:]

        # Create a naive decision rules
        # Create a code to train a neural network decide if 0 or 1. The input would be the Wasserstein distance. 
    else:
        pass
        # import sys
        # sys.exit() 

    if task_type == 'regression':
        best_r2_scores, best_rmse_scores = evaluator(train, test, info)
        
        overall_scores = {}
        for score_name in ['best_r2_scores', 'best_rmse_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    else:
        # example is to train XGBClassifier with the generated data x with target y         # changed by HP
        best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info)

        overall_scores = {}
        for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
            overall_scores[score_name] = {}
            
            scores = eval(score_name)
            for method in scores:
                name = method['name']  
                method.pop('name')
                overall_scores[score_name][name] = method 

    if not os.path.exists(f'eval/mle/{dataname}'):
        os.makedirs(f'eval/mle/{dataname}')
    
    save_path = f'eval/mle/{dataname}/{model}.json'
    print('Saving scores to ', save_path)
    with open(save_path, "w") as json_file:
        json.dump(overall_scores, json_file, indent=4, separators=(", ", ": "))

        