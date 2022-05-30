import itertools
import os
import pandas as pd

import numpy as np

import random

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
    for k, v in my_dict.items():
         if val in v:
             return k
    return "There is no such key"

def subset_data(data_dict, key_value, key_name = "task", test_size = 0.33):
    if type(data_dict[key_name]) == list:
        values = data_dict[key_name]
    else:
        values = list(data_dict[key_name].values())
        
    if type(key_value) != list:
        idx_task = np.where(np.array(values) == key_value)
    else:
        idx_task = [v in key_value for v in np.array(values)]
        idx_task = np.where(np.array(idx_task) == True)
    idx_task = idx_task[0].tolist()
    x = [data_dict['x'][i] for i in idx_task]
    tasks = [data_dict['task'][i] for i in idx_task]
    X = np.array([np.ones(len(idx_task)), np.array(x)]).T
    y = np.array([data_dict['y'][i] for i in idx_task])
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
def prepare_input(data_dict, target_task, target_test_size, d ):
    input_data = {"data_dict": data_dict}
    input_data["X_target_train"], input_data["X_target_test"], input_data["y_target_train"], input_data["y_target_test"] = subset_data(data_dict, key_value = target_task, key_name = "task")
    input_data["X_target_val"], input_data["X_target_test"], input_data["y_target_val"], input_data["y_target_test"] = train_test_split(input_data["X_target_test"], input_data["y_target_test"], 
                                                        test_size = target_test_size,
                                                        random_state = 123)
    
    input_data["source_task"] = list(set(list(itertools.chain.from_iterable(d.values()))) - set([target_task]))
    
    source_cluster = [get_key(d, i) for i in input_data["source_task"]]
    input_data["source_cluster"] = list(set(source_cluster))
    
    idx_source = np.where(np.array(list(data_dict['task'].values())) != target_task)[0].tolist()
    
    # source data
    input_data["source_dict"] = {}
    for key_name in data_dict.keys():
        input_data["source_dict"][key_name] = [data_dict[key_name][i] for i in idx_source]

    
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



