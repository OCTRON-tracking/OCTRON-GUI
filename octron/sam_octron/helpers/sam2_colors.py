# Create a continous colormap and cmap_range for each label 

import numpy as np
import cmasher as cmr

def create_label_colors(cmap='cmr.tropical', 
                        n_labels=10, 
                        n_colors_submap=250
                        ):
    """
    Create label submaps from cmap. 
    Each submap is a list of colors that represent a label.
    For this, the cmap is being divided into n_labels submaps.
    Each submap is then divided into n_colors_submap colors
    
    
    Parameters
    ----------
    cmap : str
        Name of the colormap to use.
    n_labels : int
        Number of labels to create submaps for.
    n_colors_submap : int
        Number of colors per submap.
        
    """

    slices = np.linspace(0, 1, n_labels+1)
    all_label_submaps = []
    for no in range(0, n_labels):
        label_colors = cmr.take_cmap_colors(cmap, 
                                            N=n_colors_submap, 
                                            cmap_range=(slices[no],slices[no+1]), 
                                            return_fmt='int'
                                            ) 
        label_colors = [np.concat([np.array(l) / 255., np.array([1])]) for l in label_colors]
        all_label_submaps.append(label_colors)
    return all_label_submaps


def sample_maximally_different(seq):
    """
    Given an ascending list of numbers, return a new ordering
    where each subsequent number is chosen such that its minimum
    absolute difference to all previously picked numbers is maximized.

    I added this to choose colors that are maximally different from each other,
    both for labels as well as for sub-label (same label, different suffix).

    Example:
        Input:  [1, 2, 3, 4, 5]
        Possible Output: [1, 5, 2, 4, 3]
    """
    if not seq:
        return []
    # Start with the first element.
    sample = [seq[0]]
    remaining = list(seq[1:])
    while remaining:
        # For each candidate, compute the minimum distance to any element in sample,
        # then select the candidate with the maximum such distance.
        candidate = max(remaining, key=lambda x: min(abs(x - s) for s in sample))
        sample.append(candidate)
        remaining.remove(candidate)
    return sample