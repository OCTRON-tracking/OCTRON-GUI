# Polygon helpers for training data extraction
# Also contains helpers for mask manipulation / polygon generation
import numpy as np  
from skimage import measure
from skimage.morphology import binary_opening, disk

def find_objects_in_mask(mask, min_area=10, properties=None,
                        intensity_image=None, extra_properties=None):
    """
    Find all objects in a binary mask using connected component labeling.
    This is run initially to gain an understanding of the objects in the masks,
    i.e. know their median area, etc.
    See also:
    https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    
    
    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask where objects have value 1 
        and background has value 0
    min_area : int
        Minimum area (in pixels) for an object to be considered
    properties : tuple or list, optional
        Region properties to extract via skimage.measure.regionprops_table.
        'label' and 'centroid' are always included automatically.
        If None, only the minimum internal properties are extracted.
    intensity_image : numpy.ndarray, optional
        Intensity (original) image of same spatial shape as mask.
        Required for intensity-based properties (e.g. 'intensity_mean').
        Passed directly to skimage.measure.regionprops_table.
        Safely ignored when no intensity properties are requested.
    extra_properties : tuple of callables, optional
        Custom measurement functions passed to skimage.measure.regionprops_table.
        Each function must accept a region mask as its first argument.
        If the function requires an intensity image, it must accept it as the
        second argument. The function name becomes the property/column name.
        See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops_table
        
    Returns:
    --------
    labels : numpy.ndarray
        Labeled image where each object has a unique integer value
    regions : list
        List of region property dicts for each object. Each dict contains
        'label', 'centroid', and all requested properties.
    """
    if properties is None:
        properties = ()
    
    # 'label', 'area' and 'centroid' are always needed internally
    all_properties = tuple(dict.fromkeys(('label', 'area', 'centroid') + tuple(properties)))
    
    # Collect extra property function names for output dict construction
    extra_prop_names = tuple(fn.__name__ for fn in extra_properties) if extra_properties else ()

    binary_mask = mask > 0
    labels = measure.label(binary_mask, background=0, connectivity=2)
    
    # Use regionprops_table for efficiency
    props = measure.regionprops_table(
        labels, 
        intensity_image=intensity_image,
        properties=all_properties,
        extra_properties=extra_properties,
    )
    
    # Filter small regions
    if min_area > 0:
        valid_indices = props['area'] >= min_area
        valid_labels = props['label'][valid_indices]
        
        # Create a mapping from old labels to new labels or 0 (for removed regions)
        label_mapping = np.zeros(labels.max() + 1, dtype=np.int32)
        label_mapping[valid_labels] = np.arange(1, len(valid_labels) + 1)
        
        # Apply the mapping to relabel the image in one step
        labels = label_mapping[labels]
        
        # Filter the properties table
        for prop in props:
            props[prop] = props[prop][valid_indices]
    
    # Collect all table keys that are NOT internal ('label', 'centroid-*').
    # regionprops_table may expand multi-valued / multichannel properties into
    # suffixed columns (e.g. intensity_mean-0, moments_hu-3).  We simply pass
    # through every key the table produced.
    # Note: 'area' is only used internally for min_area filtering but is
    # included in the output if the user explicitly requested it.
    _internal_prefixes = ('label', 'centroid')
    _user_requested = set(properties) | set(extra_prop_names)
    output_keys = [k for k in props
                   if not k.startswith(_internal_prefixes)
                   and (k.split('-')[0] in _user_requested
                        or k in _user_requested)]

    num_regions = len(props['label'])
    regions_list = []
    
    for i in range(num_regions):
        region_dict = {
            'label': props['label'][i],
            'centroid': (props['centroid-0'][i], props['centroid-1'][i]),
        }
        for key in output_keys:
            region_dict[key] = props[key][i]
        regions_list.append(region_dict)
    
    return labels, regions_list


def watershed_mask(mask,
                   footprint_diameter,
                   min_size_ratio=0.1,
                   p_norm=2,
                   plot=False,
                   ):
    """
    Watershed segmentation of a mask image
    
    Parameters
    ----------
    mask : np.array : Binary mask where objects have value 1 
                      and background has value 0
    footprint_diameter : float : Diameter of the footprint for peak_local_max()
    min_size_ratio : float : Minimum size ratio of a mask towards
                            the largest mask to keep a mask
    p_norm : int : Norm to use for peak_local_max(). Deafault is 2 (Euclidean)
    plot : bool : Whether to plot the results
    
    
    Returns
    -------
    labels : np.array : Segmented mask, where 0 is background 
                        and each object has a unique integer value
    masks : list : List of binary masks for each object
    
    
    """
    try:
        from scipy import ndimage as ndi
    except ImportError:
        raise ImportError('watershed_mask() requires scipy')
    try:
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
    except ImportError:
        raise ImportError('watershed_mask() requires scikit-image')
    
    assert mask.ndim == 2, f'Mask should be 2D, but got ndim={mask.ndim}'
    assert not np.isnan(mask).any(), 'There are NaNs in input mask' # If this happens, it can be solved!
    assert set(np.unique(mask)) == set([1,0]), 'Mask should be composed of 0s and 1s'
    
    diam = int(np.round(footprint_diameter))
    assert diam > 0, 'Footprint diameter should be a positive integer'
    
    # Watershed segmentation
    # See https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/plot_watershed.html
    distance = ndi.distance_transform_edt(mask)
    diam = int(np.round(diam))
    peak_dist = int(np.round(diam/4))
    coords = peak_local_max(distance, 
                            footprint=np.ones((diam,diam)), 
                            labels=mask,
                            min_distance=peak_dist,
                            p_norm=p_norm, 
                            )
    mask_ = np.zeros(distance.shape, dtype=bool)
    mask_[tuple(coords.T)] = True
    markers, _ = ndi.label(mask_)
    labels = watershed(-distance, markers, mask=mask) # this is the segmentation
    masks = []
    areas = []
    for l in np.unique(labels):
        if l == 0:  
            # That's background
            continue
        labelmask = np.zeros_like(labels)
        labelmask[labels == l] = 1
        masks.append(labelmask)    
        areas.append(np.sum(labelmask))
        
    # If we have more than one mask, check for size disparities
    # Filter out masks that are too small
    if len(masks) > 1:
        max_area = max(areas)
        filtered_masks = []
        for mask_idx, area in enumerate(areas):
            # Keep the mask if it's at least min_size_ratio of the largest mask
            if area >= min_size_ratio * max_area:
                filtered_masks.append(masks[mask_idx])
        masks = filtered_masks
        
    # Create a new labels image
    labels = np.zeros_like(mask)
    for i, m in enumerate(masks):
        labels[m == 1] = i + 1

    if plot:
        import matplotlib.pyplot as plt
        plt.rcParams['xtick.major.size'] = 10
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['ytick.major.size'] = 10
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True

        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(mask, cmap='gray')
        ax[0].set_title('Original mask')
        ax[1].imshow(labels, cmap='nipy_spectral')
        ax[1].set_title(f'Watershed, remaining masks: {len(masks)}\ndiam:{diam}')
        plt.show()
        
    return labels, masks


def merge_multi_segment(segments):
    """
    This function is copied from ultralytics/utils/ops.py
    
    Merge multiple segments into one list by connecting the coordinates with the minimum distance between each segment.
    This function connects these coordinates with a thin line to merge all segments into one.

    Parameters
    ----------
    segments (List[np.ndarray]): A list of segments, where each segment is a NumPy array of shape (N, 2).
                             
    Returns
    -------
    s (List[np.ndarray]): A list of connected segments represented as NumPy arrays.

    """
    def _min_index(arr1, arr2):
        """
        Find a pair of indexes with the shortest distance between two arrays of 2D points.

        Parameters
        ----------
        arr1 (np.ndarray): A NumPy array of shape (N, 2) representing N 2D points.
        arr2 (np.ndarray): A NumPy array of shape (M, 2) representing M 2D points.
        
        Returns
        -------
        min_index1 (tuple): A tuple containing the indexes of the points with the 
                            shortest distance in arr1 and arr2 respectively.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)
    
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # Record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = _min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # Use two round to connect all the segments
    for k in range(2):
        # Forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # Middle segments have two indexes, reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def get_polygons(mask):
    """
    Given a mask image, extract outlines as polygon 
    
    Parameters
    ----------
    mask : np.array : Mask image, composed of 0s and 1s, where 1 

    Returns
    -------
    polygon_points : np.array : Polygon points for the extracted binary mask(s)
    
    """
    try:
        from imantics import Mask
    except ImportError:
        raise ImportError('get_polygons() requires imantics')
    
    if mask is None:
        return None 
    
    assert mask.ndim == 2, f'Image should be 2D, but got ndim={mask.ndim}'
    assert not np.isnan(mask).any(), 'There are NaNs in input image' # If this happens, it can be solved!
    assert set(np.unique(mask)) == set([1,0]), 'Image should be composed of 0s and 1s'   
    
    polygons = Mask(mask).polygons()
    polygons = (
                    np.concatenate(merge_multi_segment([x.reshape(-1, 2) for x in polygons.points]))
                    if len(polygons.points) > 1
                    else polygons.points[0].reshape(-1, 2)
                )
    return polygons


def polygon_to_mask(empty_mask, 
                    polygons, 
                    smooth_sigma=0., 
                    opening_radius=2,
                    model_imgsz=640,
                    ):
    """
    Convert a polygon to a binary mask
    
    Parameters
    ----------
    empty_mask : np.array : Empty mask to fill with the polygon
    polygons : np.array : Polygon points for the extracted binary mask(s)
    smooth_sigma : float : Sigma for Gaussian smoothing
    opening_radius : int : Radius for binary opening
    model_imgsz : int : Image size the model was using for training
                        (default is 640) 
                        This is used to scale the mask before opening
    
    Returns
    -------
    mask : np.array : Mask for the polygon(s) with dtype int8
    """
    try:
        import cv2
    except ImportError:
        raise ImportError('to_mask() requires OpenCV')
    assert isinstance(polygons, np.ndarray), 'Polygons should be a numpy array'
    
    # Smooth?
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        polygons = gaussian_filter1d(polygons, axis=0, sigma=smooth_sigma)
    
    mask = cv2.fillPoly(
        empty_mask.copy(),
        [np.round(polygons).astype(np.int32)],
        color=(1,),
        lineType=cv2.LINE_AA,
    )

    # Check if opening should be applied
    if opening_radius > 0:
        scale = 1
        h0, w0 = mask.shape
        
        # up‐scale if too small
        if max(h0, w0) < model_imgsz:
            scale = model_imgsz / max(h0, w0)
            h1 = int(round(h0 * scale))
            w1 = int(round(w0 * scale))
            mask = cv2.resize(mask, (w1, h1), interpolation=cv2.INTER_NEAREST_EXACT)
        
        disk_el = disk(radius=opening_radius,
                       strict_radius=False,
                       decomposition=None,
                       )
        mask = binary_opening(mask, footprint=disk_el).astype('uint8')
        # down‐scale back to exact original size
        if scale > 1:
            mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_AREA)
    
    # Convert mask to int8
    # This is because the zarr array is created as int8 to use -1 as a placeholder
    # ... and threshold to a binary mask
    mask = mask.astype('int8')
    return mask


def postprocess_mask(mask, opening_radius):
    """
    Postprocess mask by applying binary opening with a disk footprint.
    
    Parameters
    ----------
    opening_radius : int : Radius for binary opening
    
    Returns
    -------
    mask : np.array : Postprocessed mask with dtype uint8
    
    """
    
    if opening_radius > 0:
        disk_el = disk(radius=opening_radius,
                        strict_radius=False,
                        decomposition=None,
                        )
        mask = binary_opening(mask, footprint=disk_el).astype('int8')
    return mask
