# Shared constants for the YOLO / OCTRON training pipeline

# Color codes for train-mode indicators (segment vs. detect)
TASK_COLORS = {
    'segment': '#7e56c2',
    'detect':  '#5f9bdb',
}

# All available scalar region properties from skimage.measure.regionprops_table,
# grouped by category for the configuration dialog.
# 'centroid' and 'label' are always included internally and should not be listed here.
# See https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
ALL_REGION_PROPERTIES = {
    'Size and Shape': (
        'area',
        'area_convex',
        'area_filled',
        'axis_major_length',
        'axis_minor_length',
        'eccentricity',
        'equivalent_diameter_area',
        'euler_number',
        'extent',
        'feret_diameter_max',
        'moments_hu',
        'orientation',
        'perimeter',
        'perimeter_crofton',
        'solidity',
    ),
    'Intensity': (
        'intensity_max', 
        'intensity_mean',
        'intensity_min',
        'intensity_std',
    ),
}

# This set is updated by the Region Properties configuration dialog.
DEFAULT_REGION_PROPERTIES = (
    'area',
)
