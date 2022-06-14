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
    """
    Pre-processing input data
    """
    def __init__(self, raw_data):
        self.raw_data = raw_data
    def normalize(self, raw, method = "min-max"):
        """
        Normalizing a list or columns of a numpy array.
        """
        if method == "min-max":
            if type(raw) == list:
                #processed = [(v - min(raw)) / (max(raw) - min(raw)) for v in raw]
                processed = [(v - raw.mean()) / raw.std() for v in raw]
            if type(raw) == np.ndarray:
                processed = (raw - raw.mean(axis=0)) / raw.std(axis=0)
                if len(processed.shape) > 1:
                    processed[:, 0] = 1
        return processed
    def normalize_by_key(self, key_name, by_key, method = "min-max"):
        """
        Normalizing one value of a dictionary, within groups defined by another key
        """
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
        """
        Normalizing multiple values of a dictionary, (optional) within groups defined by another key.
        """
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
    """
    Updating hyper parameters at a bandit iteration
    """
    # for selected bandits
    if losses[-1] < losses[-2]:
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





def save_files(output_dir, alpha, beta, losses, bandit_selects, pi, bl):
    output_dir.mkdir(exist_ok = True)
    pd.DataFrame.from_dict(alpha).to_csv(output_dir / "alpha.csv")
    pd.DataFrame.from_dict(beta).to_csv(output_dir / "beta.csv")
    pd.DataFrame.from_dict({"losses": [l.item() for l in losses], "bandit_selects": bandit_selects}).to_csv(output_dir / "losses.csv")
    pd.DataFrame.from_dict(pi).to_csv(output_dir / "pi.csv")
    pd.DataFrame.from_dict(bl).to_csv(output_dir / "baseline.csv")

def draw_weighted_samples(input_data, alpha, beta):
    """
    Drawing weighted samples from source
    """  
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


class lm():
    """
    Linear regression model
    """
    def __init__(self):
        self.model = LinearRegression()
    def initialize(self):
        self.model = LinearRegression()
        return self
    def prepare_data(self, x, y):
        if len(x.shape) <= 1:
            x = np.array([np.ones(x.shape), np.array(x)]).T
        return x, y
    def fit(self, x_train, y_train, loss_fn = None):
        self.model.fit(x_train, y_train)
        return self.model
    def evaluate(self, x_test, y_test, loss_fn):
        y_hat = self.model.predict(x_test)
        l = loss_fn(y_test, y_hat)
        return l
    def combine_with_old(self, model_old, decay_rate = .5):
        self.model.coef_ = decay_rate * model_old.coef_ + (1 - decay_rate) * self.model.coef_
        return self.model
    def pred(self, x_new):
        y_hat = self.model.predict(x_new)
        return y_hat
    def save(self, path = ".", x_new = None, y_new = None, para = True):
        path = Path(path)
        path.mkdir(parents = True, exist_ok = True)
        if para is True:
            pd.DataFrame.from_dict({"coef": self.model.coef_}).to_csv(path / Path("coef.csv"))
        if not x_new is None:
            pd.DataFrame.from_dict({"x": x_new[:, 1], "y": y_new, "y_hat": self.pred(x_new)}).to_csv(path / Path("fitted.csv"))
        return para

class nn():
    """
    Neural network
    """
    def __init__(self, n_inputs = 1, n_outputs = 1, H = 200):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, n_outputs),
        )
    def initialize(self, n_inputs = 1, n_outputs = 1, H = 200):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, n_outputs),
        )
        return self
    def prepare_data(self, x, y):
        if type(x) != torch.Tensor:
            if len(x.shape) > 1:
                x = torch.tensor(x[:, 1:]).float()
            else:
                x = torch.tensor(x).float()
        if type(y) != torch.Tensor:
            y = torch.tensor(y).float()
        return x, y
    def fit(self, x_train, y_train, loss_fn = torch.nn.MSELoss(), n_epochs = 10, lr = 1e-4):
        model = self.model
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        for epoch in range(n_epochs):
            # get loss
            optimizer.zero_grad()
            y_hat = self.model(x_train[:, np.newaxis])
            loss = loss_fn(y_train, y_hat)

            # update weights
            loss.backward()
            optimizer.step()
        return self
            
        return model
    def evaluate(self, x_test, y_test, loss_fn = torch.nn.MSELoss()):
        with torch.no_grad():
            y_hat = self.model(x_test[:, np.newaxis])
            l = loss_fn(y_test, y_hat)
        return l
    def pred(self, x_new):
        with torch.no_grad():
            y_hat = self.model(x_new)
        return y_hat
    def combine_with_old(self, model_old, decay_rate = .5):
        for i in range(len(model_old)):
            if "weight" in dir(model_old[i]):
                self.model[i].weight = torch.nn.Parameter(decay_rate * model_old[i].weight + (1 - decay_rate) * self.model[i].weight)
                self.model[i].bias = torch.nn.Parameter(decay_rate * model_old[i].bias + (1 - decay_rate) * self.model[i].bias)
    def save(self, path = ".", x_new = None, y_new = None, para = True):
        x_new, y_new = self.prepare_data(x_new, y_new)
        path = Path(path)
        path.mkdir(parents = True, exist_ok = True)
        y_hat = self.pred(x_new)
        if not x_new is None:
            pd.DataFrame.from_dict({"x": [item[0] for item in x_new.tolist()], 
                        "y": y_new,
                        "y_hat": [item[0] for item in y_hat.tolist()]
                       }).to_csv(path / Path("fitted.csv"))

def baseline(input_data, alpha, beta, model,  loss_fn, N):
    """
    Baseline models of out-of-domain generalization
    """
    final_loss = dict.fromkeys(["bandit", "all_source", "target_train", "random_source"], [])
    
    # weighted all source, by bandit selection parameters ----
    mod = model
    X_end, y_end = draw_weighted_samples(input_data, alpha, beta)
    X_end, y_end = mod.prepare_data(X_end, y_end)

    mod.fit(X_end, y_end, loss_fn)
    test_x, test_y = mod.prepare_data(input_data["X_target_test"], input_data["y_target_test"])
    final_loss["bandit"] = [mod.evaluate(test_x, test_y, loss_fn = loss_fn).item()]
    
    # All sources----
    mod = model
    X_sources, y_sources = mod.prepare_data(input_data["source_dict"]["x"], input_data["source_dict"]["y"])
    mod.fit(X_sources, y_sources, loss_fn)
    final_loss["all_source"] = [mod.evaluate(test_x, test_y, loss_fn = loss_fn).item()]
    
    # target train ---
    mod = model
    X_train, y_train = mod.prepare_data(input_data["X_target_train"], input_data["y_target_train"])
    mod_train = model.fit(X_train, y_train, loss_fn)
   
    final_loss["target_train"] =[ mod.evaluate(test_x, test_y, loss_fn = loss_fn).item()]

    # One random source + target train ----
    mod = model
    for n in range(N):
        # one random source
        X_random, y_random, _ = subset_data(input_data["data_dict"],
                                            key_value = random.choice(input_data["source_task"]),
                                            key_name = "task", test_size = 0)
        X_random = np.concatenate((X_random, input_data["X_target_train"]), axis = 0)
        y_random = np.concatenate((y_random, input_data["y_target_train"]), axis = 0)
        X_random, y_random = mod.prepare_data(X_random, y_random)

        mod.fit(X_random, y_random, loss_fn)
        final_loss["random_source"] =[ mod.evaluate(test_x, test_y, loss_fn = loss_fn)]
    final_loss["random_source"] = [np.mean(final_loss["random_source"])]
    
    return(final_loss)






def bandit_source_train(input_data, model, batch_size, decay_rate, n_it, loss_fn, conservative = False):
    bandit_selects = [None]
    # initialize hyperparameters
    alpha = dict.fromkeys(input_data["source_task"], [1])
    beta = dict.fromkeys(input_data["source_task"], [1])
    pi = dict.fromkeys(input_data["source_task"], [0])
    
    mod = model
    val_x, val_y = mod.prepare_data(input_data["X_target_val"], input_data["y_target_val"])

    # initialize model from target training data
    X_current, y_current = mod.prepare_data(input_data["X_target_train"], input_data["y_target_train"])
    mod.fit( X_current, y_current)
    l = mod.evaluate(val_x, val_y, loss_fn = loss_fn)
    losses = [l]
    model_old = mod.model
    
    for t in range(n_it):
        mod = model

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
        X_current, y_current = mod.prepare_data(X_current, y_current)

        # train model
        mod.fit(X_current, y_current, loss_fn = loss_fn)

        # combine parameters with previous model
        mod.combine_with_old(model_old, decay_rate = decay_rate)

        # evaluate model
        l = mod.evaluate(val_x, val_y, loss_fn = loss_fn)
        losses += [l]
        model_old = mod.model
        
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
    #bl = baseline(input_data = input_data, alpha = alpha, beta = beta,
    #              N = 10, model = model, loss_fn = loss_fn)
    return losses, alpha, beta, bandit_selects, pi#, bl


