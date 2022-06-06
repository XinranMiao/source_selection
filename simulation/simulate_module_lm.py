import itertools
import os
import pandas as pd
from pathlib import Path
import numpy as np

import random
import torch
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
np.random.seed(1234)

# Generate data
def random_functions(n_tasks, k_clusters, sigma_between, sigma_within):
    np.random.seed(1234)
    betas, zs = gaussian_mixture(n_tasks, k_clusters, sigma_between, sigma_within)
    functions = []
    for beta in betas:
        functions.append(spline_fun(beta))
        
    return functions, betas, zs

def gaussian_mixture(n_samples, k_clusters, sigma_between, sigma_within, dim=31):
    means = np.random.normal(0, sigma_between, (k_clusters, dim))
    cluster_props = np.random.dirichlet(k_clusters * [1.5])
    
    betas, zs = [], []
    for task in range(n_samples):
        z = np.random.choice(range(k_clusters), p = cluster_props)
        delta = np.random.normal(0, sigma_within, (1, dim))
        betas.append(means[z] + delta)
        zs.append(z)
        
    return np.vstack(betas), zs
        
def spline_fun(beta):
    def f(x):
        return np.matmul(basis(x), beta)
    return f
    
def basis(x, knots=None, degree = 3):
    if knots is None:
        knots = np.linspace(0, 1, 10)
        
    H = [np.ones(len(x))]
    for k in range(len(knots)):
        for j in range(1, degree + 1):
            H.append(pos(x - knots[k]) ** j)
    
    return np.vstack(H).T

def pos(x):
    x[x < 0] = 0
    return x



# Data manipulation
def get_key(my_dict, val):
    """
    Obtaining key of a dictionary by a value
    """
    for k, v in my_dict.items():
         if val in v:
             return k
    return "There is no such key"


class pre():
    def __init__(self, raw_data):
        self.raw_data = raw_data
    def normalize(self, raw, method = "min-max"):
        if method == "min-max":
            if type(raw) == list:
                processed = [(v - min(raw)) / (max(raw) - min(raw)) for v in raw]
            if type(raw) == np.ndarray:
                processed = (raw - raw.min(axis = 0)) / (raw.max(axis = 0) - raw.min(axis = 0))
                if len(processed.shape) > 1:
                    processed[:, 0] = 1
        return processed
    def normalize_by_key(self, key_name, by_key, method = "min-max"):
        processed = self.raw_data[key_name]
        if type(processed) == list:
            processed = np.array(processed)
        by_key_values = list(set(self.raw_data[by_key]))
        for by_v in by_key_values:
            # indices for those with value == by_v on key by_key
            idx = [i for (i, v) in enumerate(self.raw_data[key_name]) if self.raw_data[by_key][i] == by_v]
            processed[idx] = self.normalize(raw = processed[idx] , method = method)
        return processed
    def pre_process(self, key_names = None, method = "min-max", by_key = "task"):
        processed_data = self.raw_data
        if type(self.raw_data) == dict:
            for key_name in key_names:
                if by_key is None:
                    processed_data[key_name] = self.normalize(raw = processed_data[key_name], method = method)
                else:
                    processed_data[key_name] = self.normalize_by_key(key_name = key_name,
                                                                     by_key = by_key,
                                                                     method = method)
        else:
            processed_data = self.normalize(raw = self.raw_data, method = method)
        return processed_data


def subset_data(data_dict, key_name = "task", key_value = 0, test_size = 0.33):
    """
    Subsetting data by the value of a key.
    
    Parameters
    ---
    data_dict: dict
        the dictionary one wants to subset
    key_name: str
        the key one wants to subset on
    key_value: list / int / str
        the value of the key desirable in the output subset
    test_size: float
        how to split the resulting subset; if set to zero, then the output won't be splitted

    Returns
    ---
    
    """
    if type(data_dict[key_name]) == list:
        values = data_dict[key_name]
    else:
        values = list(data_dict[key_name].values())
    
    n_task = max(values) + 1    
    if type(key_value) != list:
        idx_task = [i for (i, v) in enumerate(values) if v == key_value]
    else:
        idx_task = [i for (i, v) in enumerate(values) if v in key_value]
        
    tasks = [data_dict['task'][i] for i in idx_task]
    
    
    x = [data_dict['x'][i] for i in idx_task]
    y = np.array([data_dict['y'][i] for i in idx_task])
    X = np.array([np.ones(len(idx_task)), np.array(x)]).T
    
    if test_size == 0:
        return X, y, tasks
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size,
                                                        random_state = 123)
    return X_train, X_test, y_train, y_test


def mse(y_true, y_predict):
    mse = np.mean((y_predict - y_true) ** 2)
    return mse


# d dict, keys: original clusters; values: tasks (bandits)
def prepare_input(data_dict, target_task, target_test_size, preprocess = True):
    """
    Preparing input data for bandit selection
    
    Parameters
    ---
    data_dict: dict
        all data, including source and target
    target_task: int
        data with data_dict["task"] equals to target_task will be in the target
    target_test_size: float
        within [0, 1) indicating the proportion of the validation + test set.
    
    Returns
    ---
    input_data: dict
        keys including data_dict, source_dict,
                        source_task, source_cluster,
                        X_target_train, X_target_test, X_target_val, y_target_train, y_target_test, y_target_val
    """
    
    n_tasks = max(data_dict["task"]) + 1


    input_data = {"data_dict": data_dict}
    input_data["X_target_train"], input_data["X_target_test"], input_data["y_target_train"], input_data["y_target_test"] = subset_data(data_dict, key_value = target_task, key_name = "task",
                                                                                                                                      test_size = target_test_size)
    input_data["X_target_val"], input_data["X_target_test"], input_data["y_target_val"], input_data["y_target_test"] = train_test_split(input_data["X_target_test"], input_data["y_target_test"], 
                                                        test_size = .5,
                                                        random_state = 123)
        
    input_data["source_task"] = [v for v in range(n_tasks) if v != target_task]
    
    
    idx_source = [i for (i, v) in enumerate(data_dict['task']) if v != target_task]
    
    # source data
    input_data["source_dict"] = {}
    for key_name in data_dict.keys():
        input_data["source_dict"][key_name] = [data_dict[key_name][i] for i in idx_source]
    
    
    if preprocess:
        input_data["data_dict"] = pre(raw_data = input_data["data_dict"]).pre_process(key_names = ["y", "x", "f"], by_key = "task")
        input_data["source_dict"] = pre(raw_data = input_data["source_dict"]).pre_process(key_names = ["y", "x"], by_key = "task")
        input_data = pre(raw_data = input_data).pre_process(key_names = ["X_target_test", "X_target_val", "X_target_train",
                                          "y_target_train", "y_target_val", "y_target_test"], by_key = None)
    
    return(input_data)


# TS bandit selection
def get_bandit(input_data, alpha, beta, t, pi, key_name = "source_task"):
    source_cluster = alpha.keys()
    for cluster in source_cluster:
        if t == 0:
            pi[cluster] = [np.random.beta(alpha[cluster][t], beta[cluster][t])]
        else:
            pi[cluster].append(np.random.beta(alpha[cluster][t], beta[cluster][t]))
    pi_list = [pi[cluster][-1] for cluster in input_data[key_name]]
    bandit = get_key(pi, max(pi_list))
    return(bandit, pi)

def update_hyper_para(alpha, beta, t, losses, bandit_current, thres = -1):
    # for selected bandits
    if losses[-1] < losses[-2]:
    #if losses[-1] < np.mean(losses):
    #if losses[-1] < np.quantile(losses, .25):
        alpha[bandit_current] = alpha[bandit_current] + [alpha[bandit_current][-1] + 1]
        beta[bandit_current] = beta[bandit_current] + [beta[bandit_current][-1]]
    elif losses[-1] > thres:
        alpha[bandit_current] = alpha[bandit_current] + [1]
        beta[bandit_current] = beta[bandit_current] + [100]
    else:
        alpha[bandit_current]  = alpha[bandit_current] + [alpha[bandit_current][-1]]
        beta[bandit_current] = beta[bandit_current] + [beta[bandit_current][-1] + 1]
    # for unselected bandits
    for bandit in alpha.keys():
        if len(alpha[bandit]) < len(alpha[bandit_current]):#t + 2:
           alpha[bandit] = alpha[bandit] + [alpha[bandit][-1]]
           beta[bandit] = beta[bandit] + [beta[bandit][-1]]
    return alpha, beta

def pred_ensemble(X_new, y_new, predict_old, predict_new, decay_rate):
    pre1 = predict_old(X_new)
    pre2 = predict_new(X_new)
    return (1 - decay_rate) * pre1 + decay_rate * pre2

def avg_loss(bandit_selects, losses, bandit_current):
    j = 0
    s = 0
    for b, l in zip(bandit_selects, losses):
        if (not b == bandit_current) and (not b is None):
            s += l
            j = j + 1
    if j == 0:
        return 100000
    else:
        return s/j    



def baseline(input_data, pi, N, alpha, beta, model, pred_ensemble, loss):
    final_loss = dict.fromkeys(["bandit", "all_source", "target_train", "random_source"], [])
    
    # weighted all source, by bandit selection parameters ----
    X_end, y_end, tasks = subset_data(input_data["data_dict"], key_value = input_data["source_task"], key_name = "task", test_size = 0)
    
    X_end = np.concatenate((X_end, input_data["X_target_val"]), axis = 0)
    y_end = np.concatenate((y_end, input_data["y_target_val"]), axis = 0)
    #mod_train = model.fit(X_end, y_end, [pi[t][-1] for t in tasks])
    weights = [alpha[t][-1] / (alpha[t][-1] + beta[t][-1]) for t in tasks]
    weights = weights + [sum(weights) / len(input_data["y_target_val"])] * len(input_data["y_target_val"])
    mod_train = model.fit(X_end, y_end, weights)
    mod_pred = pred_ensemble(input_data["X_target_test"], input_data["X_target_test"],
                                 mod_train.predict, mod_train.predict, decay_rate = 1)
    final_loss["bandit"] = [loss(input_data["y_target_test"], mod_pred)]
    
    # All source ----
    mod_train = model.fit(X_end, y_end)
    mod_pred = pred_ensemble(input_data["X_target_test"], input_data["X_target_test"],
                                 mod_train.predict, mod_train.predict, decay_rate = 1)
    final_loss["all_source"] = [loss(input_data["y_target_test"], mod_pred)]
    
    # target train
    mod_train = model.fit(input_data["X_target_train"], input_data["y_target_train"])
    mod_pred = pred_ensemble(input_data["X_target_test"], input_data["X_target_test"],
                                 mod_train.predict, mod_train.predict, decay_rate = 1)
    final_loss["target_train"] =[ loss(input_data["y_target_test"], mod_pred)]

    # One random source
    for n in range(N):
        # one random source
        X_random, y_random, _ = subset_data(input_data["data_dict"],
                                            key_value = random.choice(input_data["source_task"]),
                                            key_name = "task", test_size = 0)
        X_random = np.concatenate((X_end, input_data["X_target_val"]), axis = 0)
        y_random = np.concatenate((y_end, input_data["y_target_val"]), axis = 0)

        mod_train = model.fit(X_random, y_random)
        mod_pred = pred_ensemble(input_data["X_target_test"], input_data["X_target_test"],
                                     mod_train.predict, mod_train.predict, decay_rate = 1)
        final_loss["random_source"] = final_loss["random_source"] + [loss(input_data["y_target_test"], mod_pred)]
    final_loss["random_source"] = [np.mean(final_loss["random_source"])]
    
    return(final_loss)




def bandit_source_train(input_data, model, batch_size, decay_rate, n_it, loss, conservative = False):
    bandit_selects = [None]
    # initialize hyperparameters
    alpha = dict.fromkeys(input_data["source_task"], [1])
    beta = dict.fromkeys(input_data["source_task"], [1])
    pi = dict.fromkeys(input_data["source_task"], [0])
    
    # initialize model from target training data
    mod_train = model.fit(input_data["X_target_train"], input_data["y_target_train"])
    mod_pred = mod_train.predict(input_data["X_target_val"])
    losses = [loss(mod_pred, input_data["y_target_val"])]
    
    
    for t in range(n_it):
        # select bandit
        bandit_current, pi = get_bandit(input_data, alpha, beta,t, pi)
        bandit_selects.append(bandit_current)
        # set training data at this iteration
        X_current, y_current, _ = subset_data(input_data["source_dict"], 
                                   key_value = bandit_current,
                                   key_name = "task", test_size = 0)
        batch_id = random.choices(list(range(0, len(y_current))), k = batch_size)
        X_current, y_current = X_current[batch_id, :], y_current[batch_id]
        
        X_current = np.concatenate((X_current, input_data["X_target_val"]), axis = 0)
        y_current = np.concatenate((y_current, input_data["y_target_val"]), axis = 0)
        
        # train model
        mod_old = mod_train
        mod_train = model.fit(X_current, y_current)
        mod_pred = pred_ensemble(input_data["X_target_val"], input_data["X_target_val"],
                             mod_old.predict, mod_train.predict, decay_rate)
        
        # evaluate model
        l = loss(input_data["y_target_val"], mod_pred)
        losses += [l]
        
        
        # update bandit parameters
        if conservative:
            thres = 100000
        else:
            thres = avg_loss(bandit_selects, losses, bandit_current)
        alpha, beta = update_hyper_para(alpha, beta, t, losses,
                                        bandit_current,
                                        thres = thres
                                       )
    # baseline   
    _, prob = get_bandit(input_data, alpha, beta,t, pi)
    #bandit_weights = prob
    #prob = list(pi.values())
    #prob = list(np.concatenate(prob).flat)
    bl = baseline(input_data = input_data, pi=pi, alpha = alpha, beta = beta,
                  N = 10, model = model, pred_ensemble = pred_ensemble, loss = loss)
    bandit_weights = [prob[bd][-1] for bd in list(prob.keys())]
    
    return losses, alpha, beta, bandit_selects, pi, bl, bandit_weights


def save_files(output_dir, alpha, beta, losses, bandit_selects, pi, bl, bandit_weights):
    output_dir.mkdir(exist_ok = True)
    pd.DataFrame.from_dict(alpha).to_csv(output_dir / "alpha.csv")
    pd.DataFrame.from_dict(beta).to_csv(output_dir / "beta.csv")
    pd.DataFrame.from_dict({"losses": [l.item() for l in losses], "bandit_selects": bandit_selects}).to_csv(output_dir / "losses.csv")
    pd.DataFrame.from_dict(pi).to_csv(output_dir / "pi.csv")
    pd.DataFrame.from_dict(bl).to_csv(output_dir / "baseline.csv")

def draw_weighted_samples(input_data, alpha, beta):  
    weight_dict = {t: alpha[t][-1] / (alpha[t][-1] + beta[t][-1]) for t in input_data["source_task"]}
    s = sum(weight_dict.values())
    weight_dict = {t: weight_dict[t] / s for t in weight_dict.keys()}


    n_tasks = len(weight_dict.keys())
    draw = np.random.multinomial(n = 1, pvals = list(weight_dict.values()), size = 100 )
    col_idx = 0
    X_end, y_end = None, None
    for col_idx in range(n_tasks):
        X_task, y_task, _ = subset_data(input_data["source_dict"],
                                        key_value = list(weight_dict.keys())[col_idx],
                                        key_name = "task", test_size = 0)

        non_zero_idx = np.array([i for (i, v) in enumerate(draw[:, col_idx]) if v == 1])
        if len(non_zero_idx) > 0:
            X_task = X_task[non_zero_idx, :]
            y_task = y_task[non_zero_idx]

            if X_end is None:
                X_end, y_end = X_task, y_task
            else:
                X_end = np.concatenate((X_end, X_task), axis = 0)
                y_end = np.concatenate((y_end, y_task), axis = 0)
        col_idx += 1
        
    X_end = np.concatenate((X_end, input_data["X_target_val"]), axis = 0)
    y_end = np.concatenate((y_end, input_data["y_target_val"]), axis = 0)
    
    return X_end, y_end


