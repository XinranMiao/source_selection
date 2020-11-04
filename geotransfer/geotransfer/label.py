import pandas as pd
import numpy as np
from .sen12ms_dataLoader import LCBands

label_idx = {'Original':np.array(range(17))+1,
        'Simplified':np.array([1,1,1,1,1,2,2,3,3,4,5,6,7,6,8,9,10])}
simply_df = pd.DataFrame(label_idx)
labels = ['Forest','Shrubland','Savanna','Grassland','Wetlands',
                  'Croplands','Urban/Built-up','Snow/Ice','Barren','Water']

simplify = lambda t:simply_df.loc[t-1,'Simplified']

class Labels:
    def simplify_labels(self):
        a = np.copy(self)
        with np.nditer(a, op_flags=['readwrite']) as it:
            for x in it:
                x[...] =np.array(simplify(x))
        return(a)

    def labels_prob(self):
        proportion = np.zeros(10)
        for i in range(10):
            #proportion.append(np.count_nonzero(self==i+1)/(256**2))
            proportion[i] = np.count_nonzero(self==i+1)/(256**2)
        return(proportion)

    def image_label(self):
        return(self.argmax()+1)

    def image_label_name(self):
        return(labels[self+1])


def get_label(filename):
    bands = LCBands.IGBP.value
    with rasterio.open(filename) as patch:
        data = patch.read(bands)
        bounds = patch.bounds
    sim = Labels.simplify_labels(data)
    prob = Labels.labels_prob(sim)
    label = Labels.image_label(prob)
    return label
