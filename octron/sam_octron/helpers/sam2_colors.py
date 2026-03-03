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


def get_semantic_cmap_range(label_id, slice_width=0.25):
    """
    Compute a colormap slice for a label using golden-ratio spacing.

    Each label gets a ~20% band of the neon colormap so that objects
    within a label share a colour family while being clearly distinct
    from objects in other labels.

    Parameters
    ----------
    label_id : int or None
        Stable label index from the object organizer.  If None the
        full usable range is returned (backward compat).
    slice_width : float
        Fraction of the [0, 1] colormap range each label occupies.

    Returns
    -------
    (start, end) : tuple[float, float]
    """
    if label_id is None:
        return (0.0, 1.0)
    GOLDEN_RATIO_CONJ = 0.6180339887498949
    usable_start = 0.0
    max_start = 1.0 - slice_width  # highest allowed start
    position = (label_id * GOLDEN_RATIO_CONJ) % 1.0
    start = usable_start + position * (max_start - usable_start)
    end = start + slice_width
    return (round(start, 4), round(end, 4))


def create_semantic_colormap(n_objects, label_id=None):
    """
    Create a DirectLabelColormap for multi-ID semantic masks.
    Uses cmr.neon with maximally-different reordering so each
    object ID is visually distinct.

    When *label_id* is given the colours are drawn from a narrow
    slice of the colourmap (see ``get_semantic_cmap_range``) so
    that different labels occupy distinct colour families.

    Parameters
    ----------
    n_objects : int
        Number of object IDs (1-based) to map.
    label_id : int or None
        Stable label index used to pick the colourmap slice.

    Returns
    -------
    DirectLabelColormap
    """
    from napari.utils import DirectLabelColormap

    cmap_range = get_semantic_cmap_range(label_id)
    obj_colors = cmr.take_cmap_colors(
        'cmr.neon', N=max(n_objects, 2), cmap_range=cmap_range, return_fmt='float'
    )
    reorder = sample_maximally_different(list(range(len(obj_colors))))
    color_dict = {None: [0, 0, 0, 0]}  # unmapped labels -> transparent
    for oid in range(1, n_objects + 1):
        r, g, b = obj_colors[reorder[oid - 1]]
        color_dict[oid] = [float(r), float(g), float(b), 1.0]
    return DirectLabelColormap(color_dict=color_dict)


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