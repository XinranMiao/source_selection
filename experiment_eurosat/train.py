from IPython.display import clear_output
from pathlib import Path

import pandas as pd

from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import models

import matplotlib.pyplot as plt

from dataset import *

def Load_model(EuroSat_Type = "ALL"):
    model_ft = models.resnet50()#pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 10)
    if EuroSat_Type == 'RGB':
      inBands = 3
    elif EuroSat_Type == 'ALL':
      inBands = 13
    model_ft.conv1 = nn.Conv2d(inBands, 64, kernel_size=7, stride=2, padding = 3, bias = False)
    print('Model Loaded')
    return model_ft
def accuracy(gt_S,pred_S):       
    _, alp = torch.max(torch.from_numpy(pred_S), 1)
    return accuracy_score(gt_S, np.asarray(alp))#np.mean(F1score)
def validation(model, test_,):
    model.eval()
    
    #tot_acc=[]
    test_iter=0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_):
            if torch.cuda.is_available():
                data, target = cus_aug(Variable(data.cuda())), Variable(target.cuda())
            else:
                data, target = cus_aug(Variable(data)), Variable(target)
            output = model(data)
            _, pred = torch.max(output, 1)
            pred = output.data.cpu().numpy()
            gt = target.data.cpu().numpy()
            if test_iter == 0:
                all_pred = pred
                all_gt = gt
            else:
                all_pred = np.vstack((all_pred,pred))
                all_gt  = np.vstack((all_gt,gt))

            test_iter = test_iter + 1
        gt = all_gt.reshape(all_gt.shape[0] * all_gt.shape[1])
        acc = accuracy(gt, all_pred)
        predictions = torch.max(torch.from_numpy(all_pred), 1)[1]; predictions = np.asarray(predictions).tolist()
        model.train()
        return acc, gt.tolist(), predictions



def train(net, train_, val_, criterion, optimizer, epochs=None, scheduler=None, weights=None, save_epoch = 10,
plot = False):
    t0 = time.time()
    losses=[]; acc=[]; mean_losses=[]; 
    train_acc = []; val_acc=[]; train_losses = []
    iter_ = 0

    for e in range(1, epochs + 1):
        print('e=',e,'{} seconds'.format(time.time() - t0))

        net.train()
        y_train = []; yhat_train = []
        for batch_idx, (data, target) in enumerate(train_):

            if torch.cuda.is_available():
                data, target =  cus_aug(Variable(data.cuda())), Variable(target.cuda())
            else:
                data, target =  cus_aug(Variable(data)), Variable(target)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses = np.append(losses,loss.item())
            mean_losses = np.append(mean_losses, np.mean(losses[max(0,iter_-100):iter_]))

            clear_output()
            pred = output.data.cpu().numpy()#[0]
            pred = sigmoid(pred)# (16, 10)
            gt = target.data.cpu().numpy()#[0]
            acc = np.append(acc, accuracy(gt, pred))


            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}\tLearning Rate:{}'.format(
            e, epochs, batch_idx, len(train_),
        100. * batch_idx / len(train_), loss.item(), acc[-1],optimizer.param_groups[0]['lr']))

            iter_ += 1

            del(data, target, loss)

            # save predictions at the last epoch
            if e == epochs:
                y_train = y_train + np.asarray(gt).tolist()

                _, predictions = torch.max(torch.from_numpy(pred), 1)
                yhat_train = yhat_train + np.asarray(predictions).tolist()

        train_losses = np.append(train_losses, np.mean(losses))
        train_acc = np.append(train_acc, np.mean(acc))
        if scheduler is not None:
           scheduler.step()
        #if e % save_epoch == 0:
            #torch.save(net.state_dict(), '.\Eurosat{}'.format(e))

        # make predictions on the validation set
        val_acc_, y_val, yhat_val = validation(net, val_)
        val_acc = np.append(val_acc, val_acc_)
        
        train_dict = {"y_train": y_train,
                 "yhat_train": yhat_train}
        val_dict = {"y_val": y_val,
               "yhat_val": yhat_val}

    
                
    return net, val_acc, train_acc, train_losses, train_dict, val_dict



# Data manipulation
def get_key(my_dict, val):
    """
    Obtaining key of a dictionary by a value
    """
    for k, v in my_dict.items():
         if val in v:
             return k
    return "There is no such key"

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
def update_hyper_para(alpha, beta, t, accs, bandit_current, thres = -1):
    """
    Updating hyper parameters at a bandit iteration
    """
    # for selected bandits
    if accs[-1] > accs[-2]:
        alpha[bandit_current] = alpha[bandit_current] + [alpha[bandit_current][-1] + 1]
        beta[bandit_current] = beta[bandit_current] + [beta[bandit_current][-1]]
    else:
        alpha[bandit_current]  = alpha[bandit_current] + [alpha[bandit_current][-1]]
        beta[bandit_current] = beta[bandit_current] + [beta[bandit_current][-1] + 1]
    # for unselected bandits
    for bandit in alpha.keys():
        if len(alpha[bandit]) < len(alpha[bandit_current]):#t + 2:
           alpha[bandit] = alpha[bandit] + [alpha[bandit][-1]]
           beta[bandit] = beta[bandit] + [beta[bandit][-1]]
    return alpha, beta


milestones = [50, 75, 90]
criteria = torch.nn.CrossEntropyLoss()
def save_output(output_name, accs, bandit_selects):
    pd.DataFrame.from_dict({"accs": [a.item() for a in accs], "bandit_selects": bandit_selects}).to_csv(output_name)

def bandit_selection(data, input_data, n_epochs = 3, n_it = 2, algorithm = "bandit",
                     group_by = "cluster",
                     iter_samples = 160,
                     lr = .01, milestones = milestones,
                     criteria = criteria, output_path = "."):
    
    # prepare data ---

    target_val_loader =  torch.utils.data.DataLoader(torch.utils.data.Subset(data, input_data["idx_val"]), 
                                                  batch_size = 16, shuffle = True, num_workers = 0)
    target_train_loader =  torch.utils.data.DataLoader(torch.utils.data.Subset(data, input_data["idx_train"]), 
                                                      batch_size = 16, shuffle = True, num_workers = 0)
    target_test_loader =  torch.utils.data.DataLoader(torch.utils.data.Subset(data, input_data["idx_test"]), 
                                                      batch_size = 16, shuffle = True, num_workers = 0)



    # initialize hyperparameters ---
    
    train_log = []
    bandit_selects = [None]
    alpha = dict.fromkeys(input_data["source_task"], [1])
    beta = dict.fromkeys(input_data["source_task"], [1])
    pi = dict.fromkeys(input_data["source_task"], [0])


    
    # initialize model ---

    net = Load_model()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma=0.1)
    if torch.cuda.is_available():
        net=net.cuda()

    net, val_acc, _, _, _, _ = train(net, target_train_loader, target_test_loader , criteria, optimizer, epochs = n_epochs, scheduler = scheduler)

    print("Model initiated with acc ", val_acc[-1])
    accs = [val_acc[-1]]
    
    
    # initialize lists for saving model predictions
    train_list = []
    val_list = []
    
    
    # bandit selection iterations ---

    for t in range(n_it):
        
        # select arm
        
        if algorithm == "bandit":
            current_id = []
            while len(current_id) == 0:
                bandit_current, pi = get_bandit(input_data, alpha, beta,t, pi)
                bandit_selects.append(bandit_current)

                current_id = [input_data["source_dict"]["id"][i] for (i, v) in enumerate(input_data["source_dict"][group_by]) if v == bandit_current]
            current_id = random.choices(current_id, k = iter_samples)
            print("---", "At iteration ", t, ", source task is ", bandit_current, "-----\n")
        else:
            bandit_current = 0
            current_id = random.sample(input_data["idx_source"], k = iter_samples)
            bandit_selects.append(None)
        
        
        # load data for this arm
        
        current_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, current_id), 
                                                          batch_size = 16, shuffle = True, num_workers = 0)
        
        
        
        # train model
        
        net, val_acc, train_acc, train_losses, train_dict, val_dict = train(net,
                                                                            current_loader,
                                                                            target_test_loader,
                                                                            criteria, optimizer,
                                                                            epochs = n_epochs,
                                                                            scheduler = scheduler)

        print("At iteration ", t, ", source task is ", bandit_current, ", acc is ", val_acc[-1])
        accs += [val_acc[-1]]

        
        # record predictions
        
        val_dict["bandit"] = t * len(val_dict["y_val"])
        val_list.append(val_dict)
        
        train_dict["bandit"] = t * len(train_dict["y_train"])
        train_list.append(train_dict)
        
        
        # record logs
        
        train_log.append({"iter": [t for i in range(n_epochs)],
                          "target_task": [input_data["target_task"] for i in range(n_epochs)],
                          "algorithm": [algorithm for i in range(n_epochs)],
                          "target_size": [len(input_data["idx_train"]) for i in range(n_epochs)],
                          "train_acc": train_acc.tolist(),
                          "val_acc": val_acc.tolist(),
                          "train_losses": train_losses.tolist()})
        
        # update bandit parameters
        
        if algorithm == "bandit":
            alpha, beta = update_hyper_para(alpha, beta, t, accs,
                                            bandit_current
                                           )
            
        # save outputs
        
        if not output_path is None:
            if t % 10 == 0:
                torch.save(net.state_dict(), output_path / Path(str(input_data["target_task"]) + "_" + algorithm + ".pt" ))
                save_output(output_path / Path(str(input_data["target_task"]) + "_" + algorithm + "_evaluation.csv" ), accs, bandit_selects = bandit_selects)

                print(train_log)
                log_df = pd.concat([pd.DataFrame(r) for r in train_log])
                log_df.to_csv(output_path /  Path(str(input_data["target_task"]) + "_" + algorithm + "train_log.csv"))
                
                # save predictions
                pd.concat([pd.DataFrame.from_dict(d) for d in train_list]).to_csv(output_path /  Path(str(input_data["target_task"]) + "_" + algorithm + "_" + str(input_data["target_size"]) + "train_pred.csv"))
                pd.concat([pd.DataFrame.from_dict(d) for d in val_list]).to_csv(output_path /  Path(str(input_data["target_task"]) + "_" + algorithm + "_" + str(input_data["target_size"])  + "val_pred.csv"))
                
                
                # save bandit hyperparameters
                if algorithm == "bandit":
                    pd.DataFrame.from_dict(alpha).to_csv(output_path /  Path(str(input_data["target_task"]) + "_" + algorithm + "_" + str(input_data["target_size"]) + "alpha.csv"))
                    pd.DataFrame.from_dict(beta).to_csv(output_path /  Path(str(input_data["target_task"]) + "_" + algorithm + "_" + str(input_data["target_size"]) + "beta.csv"))
                    pd.DataFrame.from_dict(pi).to_csv(output_path / Path(str(input_data["target_task"]) + "_" + algorithm + "_" + str(input_data["target_size"]) +  "pi.csv"))

    return net, bandit_selects, accs, alpha, beta, pi



