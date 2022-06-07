#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import numpy as np
np.random.seed(1234)

from simulate_module import *


# In[2]:


sigma_setting = {"high_bw": [10, .2],
                "medium_bw": [1, .2],
                "low_bw": [.5, .2]}


# In[3]:


args = {
    "n_tasks": 15,
    "conservative": False,
    "target_test_size": 0.8,
    "model_type": "linear_regression"
}


# In[4]:


if args["model_type"] == "linear_regression":
    model_class = lm()
elif args["model_type"] == "nn":
    model_class = nn()


# In[ ]:


for s in sigma_setting:
    # set directory
    
    if args["conservative"]:
        data_path = Path(args["model_type"] + "/conservative_derived_data")
    else:
        data_path = Path(args["model_type"] + "/derived_data")


    data_path = Path(data_path)
    working_path = data_path / s
    working_path.mkdir(parents = True, exist_ok = True)

    # generate data ------------------------------------------------
    np.random.seed(1234)
    f, betas, zs = random_functions(args["n_tasks"], 6,
                                    sigma_between = sigma_setting[s][0],
                                    sigma_within = sigma_setting[s][-1])
    result = []

    for i, fi in enumerate(f):
        x = np.random.uniform(0, 1, 100)
        result.append({
            "task": i,
            "x": x,
            "f": fi(x),
            "y": fi(x) + np.random.normal(0, .1, len(x))
        })

    # save data
    data_df = pd.concat([pd.DataFrame(r) for r in result])
    data_df.to_csv(working_path / "tasks.csv", index = False)
    data_df = data_df.reset_index()


    betas_df = np.hstack([np.arange(args["n_tasks"])[:, np.newaxis], np.array(zs)[:, np.newaxis], betas])
    betas_df = pd.DataFrame(betas_df)
    betas_df.columns = ["task", "cluster"] + [f"beta{i}" for i in range(betas.shape[1])]
    betas_df.to_csv(working_path / "betas.csv", index = False)


    data_dict = data_df.to_dict(orient = "list")

    # bandit selection ------------------------------------------------
    for target_task in range(args["n_tasks"]):
        # prepare input
        input_data = prepare_input(data_dict,
                                   target_task = target_task,
                                   target_test_size = args["target_test_size"],
                                  preprocess = True)
        pd.DataFrame.from_dict(input_data["data_dict"]).to_csv(working_path / "tasks_processed.csv", index = False)
        # bandit selection
        losses, alpha, beta, bandit_selects, pi, bl = bandit_source_train(input_data = input_data,
                                                                          model = model_class,
                                                                          batch_size = 8,
                                                                          decay_rate = .5, n_it = 100,
                                                                          loss_fn =  mse,
                                                                          conservative = args["conservative"])
        
        # save outputs
        output_dir = working_path / f"target_{target_task}_{args['target_test_size']}"
        save_files(output_dir, alpha, beta, losses, bandit_selects, pi, bl)

