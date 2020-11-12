

import numpy as np
import torch
import torchvision
from torchvision import models
from scipy import interpolate
import random
from matplotlib import pyplot as plt

import cca_core



class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []



def get_all_activations(model,layers,img,fraction=0.4):
    """Get all activation vectors over images for a model.

    :param model: A pytorch model 
    :type model: currently is Net defined by ourselves
    :param layers: One or more layers that activations are desired
    :type layers: torch.nn.modules.container.Sequential
    :param img: A 4-d tensor containing the test datapoints from which activations are desired. 
                The 1st dimension should be the number of test datapoints.
                The next 3 dimensions should match the input of the model
    :type img: torch.Tensor
    :param fraction: If defined, then a random fraction of img will be chosen.
    :type fraction: float
    :return (output): A list containing activations of all specified layers.
    """
    l = len(img)
    num_imgs = round(l*fraction)
    img_sub = img[torch.tensor(random.sample(range(l),num_imgs))]
    img_sub = torch.from_numpy(np.array(img_sub,dtype='f'))
    
    save_output = SaveOutput()
    hook_handles = []

    for layer in layers:
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
        
    out = model(img_sub)
    output = save_output.outputs.copy()
    
    del save_output,hook_handles,out
    return output



def get_svf_acts(acts = None, # should be a np.array
                 all_acts = None,
                 layer_num = 0, dim = 20):
    """Prepare the input for computing SVCCA from an existing activation.

    :acts: An 4d array of activations. If not specified, then all_acts and layer_num should be specified. 
                                         If specified, then all_acts and layer_num need not be specified.
    :type acts: numpy.ndarray
    :param all_acts: A list of 4-d activations. Usually the output of get_all_activations. 
    :type all_acts: list
    :layer_num: Defines which layer(s) of all_acts is desired
    :type layer: int
    :dim: How many dimenstions will be used.
    :return svf_acts: A 2d numpy.ndarray. The first dimension is param dim. 
    :return svb: A 2d numpy.ndarray of baselines.
    """
    if acts is None:
        acts = np.array(all_acts[torch.tensor([layer_num])])
    num_datapoints, h, w, channels = acts.shape
    
    f_acts = acts.reshape((num_datapoints*h*w, channels))
    
    # Mean subtract activations
    cf_acts = f_acts - np.mean(f_acts, axis=1, keepdims=True)
    
    # Perform SVD
    U, s, V = np.linalg.svd(cf_acts, full_matrices=False)
    
    svf_acts = np.dot(s[:dim]*np.eye(dim), V[:dim])
    
    # creating a random baseline
    b = np.random.randn(*f_acts.shape)
    cb = b - np.mean(b, axis=0, keepdims=True)
    Ub, sb, Vb = np.linalg.svd(cb, full_matrices=False)
    svb = np.dot(sb[:20]*np.eye(20), Vb[:20])
    
    return svf_acts, svb

def plot_svcca_baseline(svcca_results, svcca_baseline):
    """
    plot the output of get_svf_acts
    """
    print("Baseline", np.mean(svcca_baseline["cca_coef1"]), "and our model", np.mean(svcca_results["cca_coef1"]))

    plt.plot(svcca_baseline["cca_coef1"], lw=2.0, label="baseline")
    plt.plot(svcca_results["cca_coef1"], lw=2.0, label="Our model")
    plt.xlabel("Sorted CCA Correlation Coeff Idx")
    plt.ylabel("CCA Correlation Coefficient Value")
    plt.legend(loc="best")
    plt.grid()



def avg_cca(acts1,acts2):
    y = cca_core.get_cca_similarity(np.mean(np.array(acts1), axis=(1,2)),
                            np.mean(np.array(acts2), axis=(1,2)), 
                            epsilon=1e-10, verbose=False)
    return y

    
def reshape_acts(shape,
                 acts = None,
                 all_acts = None, 
                 layer_num = None):
    '''
    :param shape: The desired shape.
    :param acts: A 4d array. If specified, then all_acts and layer_num need not be specified. Othervise they should.
    :type acts: numpay.ndarray
    :param all_acts: A list of 4d tensors.
    :type all_acts:list
    :param layer_num: Indicates activations of which layer will be used.
    :type layer_num: int
    :return act_interp: the interpolated 4d activations with the desired shape.
    '''
    if acts is None:
        acts = np.array(all_acts[torch.tensor([layer_num])])
    
    num_d, h, w, _ = shape
    num_c = acts.shape[-1]
    acts_interp = np.zeros((num_d, h, w, num_c))
    for d in range(num_d):
        for c in range(num_c):
            # form interpolation function
            idxs1 = np.linspace(0, acts.shape[1],
                            acts.shape[1],
                            endpoint=False)
            idxs2 = np.linspace(0, acts.shape[2],
                            acts.shape[2],
                            endpoint=False)
            arr = acts[d,:,:,c]
            f_interp = interpolate.interp2d(idxs2, idxs1, arr)
        
            # creater larger arr
            large_idxs1 = np.linspace(0, acts.shape[1],
                            acts.shape[1],
                            endpoint=False)
            large_idxs2 = np.linspace(0, acts.shape[2],
                            acts.shape[2],
                            endpoint=False)
        
            acts_interp[d, :, :, c] = f_interp(large_idxs1, large_idxs2)
        
    return acts_interp
