# Plotting helpers
from loguru import logger

def make_linear_colormap(array, 
                         cmap='magma', 
                         desat=1, 
                         reference_numbers=None, 
                         categorical=False, 
                         percentile=100
                         ):
    '''
    Create color map - linear version. 
    Uses seaborn color_palette() function to create color map. 
    
    Paramters
    ---------
        - array: 1-D Numpy array to loop over and create colors for
        - cmap: Color palette name or list of colors of a (custom) palette
        - desat: seaborn specific desaturation of color palette. 1 = no desaturation.
        - reference_numbers: reference array (to use instead of input array)
        - percentile: To get rid of outlier artefacts, use percentile < 100 as maximum    
    Returns
    -------
        - colors: array of length len(array) of specified palette
    '''
    # Hide imports here 
    import numpy as np
    import seaborn as sns

    array = np.nan_to_num(array.copy())
    
    if reference_numbers is not None:
        reference_array = np.linspace(reference_numbers.min(),\
                                      reference_numbers.max(),\
                                      len(array))
    else:
        reference_array = None
        
    # Correct for negative values in array
    if not categorical: 
        if (array < 0).any():
            minimum = array.min()
            array += -1 * minimum
            if reference_array is not None: # Also correct reference number range
                reference_array += -1 * minimum
            
    if not categorical: 
        # Check what format the colormap is in 
        if isinstance(cmap, list):
            color_palette = cmap
            if len(color_palette) < 100:
                logger.warning('Less than 100 distinct colors - is that what you want?')
        elif isinstance(cmap, str):
            # ... create from scratch
            color_palette = sns.color_palette(cmap, len(array), desat)
        else:
            raise NotImplementedError(f'Type {type(cmap)} for cmap argument not understood.')

        colors = []
        for el in array:
            if reference_array is None:
                index = int(np.interp(el,[np.min(array), np.percentile(array, percentile)],[0.,1.])*len(color_palette))
            else:
                index = int((el/np.percentile(reference_array, percentile))*len(color_palette))
                
            if index > len(color_palette)-1: index = -1
            color = color_palette[index]
            colors.append(color)
    else: 
        categories = list(set(array))
        color_palette = sns.color_palette(cmap, len(categories), desat)
        colors = []
        for el in array:
            index = [no for no, category in enumerate(categories) if category == el][0]
            color = color_palette[index]
            colors.append(color)
        
    return np.array(colors)

def get_outline(mask):  
    """
    Get the outline of a binary mask using distance transform.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary mask where the object is represented by 1s and the background by 0s.
    
    Returns
    -------
    outline : np.ndarray    
        Coordinates of the outline of the object in the mask.
        
    """
    # Hide imports here 
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    
    # Create binary mask outline for the current frame
    roi_mask_outlines = np.zeros_like(mask, dtype=np.uint8)
    roi_mask_outlines[mask > 0] = 1
    distance = distance_transform_edt(roi_mask_outlines)
    distance = np.array(distance)
    distance[distance != 1] = 0
    outline = np.where(distance == 1)
    return np.stack(outline).T