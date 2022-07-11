import pandas as pd
from pathlib import Path
import numpy as np
np.random.seed(1234)

from simulate_module import *
base_dir = Path("derived_data")
base_dir.mkdir(exist_ok = True)
metadata = []
i = 0
for conservative in [True, False]:
    for target_test_size in [.4, .8]:#np.linspace(0.2,0.8,4):
        for s in sigma_setting:
            for target_task in range(15):
                for decay_rate in [0, 0.5, 1]:
                    for model_type in ["nn", "lm"]:
                        metadata.append({
                            "path": "exp" + str(i),
                            "n_tasks": 15,
                            "conservative": conservative,
                            "target_test_size": target_test_size,
                            "model_type": model_type,
                            "s": s,
                            "target_task": target_task,
                            "decay_rate": decay_rate
                        })
                        i += 1

metadf = pd.concat([pd.DataFrame(m, index=[i]) for i, m in enumerate(metadata)])
metadf.to_csv( base_dir / "metadata.csv")

sigma_setting = {"high_bw": [10, .2],
                "medium_bw": [1, .2],
                "low_bw": [.5, .2]}

for i, args in enumerate(metadata):
    working_path = Path(base_dir / args["path"])
    working_path.mkdir(parents = True, exist_ok = True)
    
    if args["model_type"] == "lm":
        model_class = lm()
        loss_fn = mse
    elif args["model_type"] == "nn":
        model_class = nn()
        loss_fn =  torch.nn.MSELoss()
    
    
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
    data_df.to_csv(working_path.joinpath("tasks.csv"), index = False)
    data_df = data_df.reset_index()
    betas_df = np.hstack([np.arange(args["n_tasks"])[:, np.newaxis], np.array(zs)[:, np.newaxis], betas])
    betas_df = pd.DataFrame(betas_df)
    betas_df.columns = ["task", "cluster"] + [f"beta{i}" for i in range(betas.shape[1])]
    betas_df.to_csv(working_path / "betas.csv", index = False)
    data_dict = data_df.to_dict(orient = "list")
    
    input_data = prepare_input(data_dict,
                               target_task = args["target_task"],
                                target_test_size = args["target_test_size"],
                                preprocess = True)
    pd.DataFrame.from_dict(input_data["data_dict"]).to_csv(working_path / "tasks_processed.csv",
                                                               index = False)
        # bandit selection
    losses, alpha, beta, bandit_selects, pi, bl = bandit_source_train(input_data = input_data,
                                                                          model = model_class,
                                                                          batch_size = 8,
                                                                          decay_rate = args["decay_rate"],
                                                                          n_it = 100,
                                                                          loss_fn =  loss_fn,
                                                                          conservative = args["conservative"])
        # save outputs
    output_dir = working_path
    save_files(output_dir, alpha, beta, losses, bandit_selects, pi, bl)


