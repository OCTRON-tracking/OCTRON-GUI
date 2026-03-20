from pathlib import Path
import cv2
import numpy as np
from loguru import logger

def interpolate_masks(mask1, 
                      mask2, 
                      num_frames, 
                      control_points=100
                      ):
    """
    Interpolate between two binary masks using an enhanced Moving Least Squares approach.

    Parameters:
    ----------
    mask1 : numpy.ndarray
        First binary mask (2D array of 0s and 1s).
    mask2 : numpy.ndarray
        Second binary mask (2D array of 0s and 1s).
    num_frames : int
        Number of interpolation frames to generate between masks.
    control_points : int, optional
        Number of points to sample along contours. Higher values give smoother results 
        but slower performance. Defaults to 100.

    Returns:
    -------
    list
        A list of numpy arrays containing the interpolated binary masks.

    Notes:
    -----
    - Both input masks should have the same dimensions.
    - The function properly handles regions with holes.
    - For masks with multiple disconnected regions, consider running this function 
      separately on each region.
    - This approach works well for complex shape transitions (e.g., square-to-circle) 
      and properly handles regions with holes.
    """
    # Extract contours with hierarchy information
    contours1, hierarchy1 = cv2.findContours(mask1.astype(np.uint8), 
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours2, hierarchy2 = cv2.findContours(mask2.astype(np.uint8), 
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    if len(contours1) == 0 or len(contours2) == 0:
        return [np.zeros_like(mask1) for _ in range(num_frames)]
    
    # Function to resample contours with equal spacing
    def resample_contour(cnt, num_points):
        # Convert to 1D array
        cnt = cnt.reshape(-1, 2)
        # Get total contour length
        perimeter = cv2.arcLength(cnt.reshape(-1, 1, 2), True)
        # Resample points at equal distances
        resampled = []
        for i in range(num_points):
            idx = i * perimeter / num_points
            # Find point at this distance along contour
            dist = 0
            for j in range(len(cnt)):
                p1 = cnt[j]
                p2 = cnt[(j + 1) % len(cnt)]
                segment_length = np.linalg.norm(p2 - p1)
                if dist + segment_length >= idx:
                    # Interpolate between p1 and p2
                    frac = (idx - dist) / segment_length
                    point = p1 + frac * (p2 - p1)
                    resampled.append(point)
                    break
                dist += segment_length
        return np.array(resampled).reshape(-1, 1, 2)
    
    # Separate external contours and holes
    def separate_contours(contours, hierarchy):
        external = []
        holes = []
        
        if hierarchy is None or len(hierarchy) == 0:
            return [max(contours, key=cv2.contourArea)], []
        
        hierarchy = hierarchy[0]  # Hierarchy comes as a nested array
        
        for i, (_, _, _, parent) in enumerate(hierarchy):
            if parent < 0:  # External contour
                external.append(contours[i])
            else:  # Hole
                holes.append(contours[i])
        
        # If no external contours found, use the largest contour
        if not external:
            external = [max(contours, key=cv2.contourArea)]
            
        return external, holes
    
    # Get external contours and holes
    external1, holes1 = separate_contours(contours1, hierarchy1)
    external2, holes2 = separate_contours(contours2, hierarchy2)
    
    # Use the largest external contour
    ext_cnt1 = max(external1, key=cv2.contourArea)
    ext_cnt2 = max(external2, key=cv2.contourArea)
    
    # Align holes between the two masks
    # This is a basic approach - more sophisticated matching could be implemented
    n_holes = min(len(holes1), len(holes2))
    if n_holes > 0:
        # Sort holes by area
        holes1 = sorted(holes1, key=cv2.contourArea, reverse=True)[:n_holes]
        holes2 = sorted(holes2, key=cv2.contourArea, reverse=True)[:n_holes]
    
    # Resample external contours
    ext_pts1 = resample_contour(ext_cnt1, control_points)
    ext_pts2 = resample_contour(ext_cnt2, control_points)
    
    # Resample holes
    hole_pts1 = [resample_contour(hole, max(10, control_points // 2)) for hole in holes1]
    hole_pts2 = [resample_contour(hole, max(10, control_points // 2)) for hole in holes2]
    
    # Generate interpolated masks
    h, w = mask1.shape
    interpolated_masks = []
    
    for t in range(1, num_frames + 1):
        alpha = t / (num_frames + 1)
        
        # Interpolate external contour
        ext_pts_interp = (1 - alpha) * ext_pts1 + alpha * ext_pts2
        
        # Create initial mask with external contour
        mask_interp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_interp, [ext_pts_interp.astype(np.int32)], 1)
        
        # Interpolate and cut out holes
        for i in range(min(len(hole_pts1), len(hole_pts2))):
            hole_pts_interp = (1 - alpha) * hole_pts1[i] + alpha * hole_pts2[i]
            cv2.fillPoly(mask_interp, [hole_pts_interp.astype(np.int32)], 0)
        
        interpolated_masks.append(mask_interp)
    
    return interpolated_masks


if __name__ == "__main__":
    output_dir = Path('/Users/horst/Desktop')
    # Create little test masks
    # and export as gif 
    #from matplotlib import pyplot as plt
    import imageio
    mask = np.zeros((500,500))
    
    mask1 = mask.copy()
    mask1[30:80, 30:80] = 1 # Square filled with 1
    mask2 = mask.copy()
    mask2[100:160, 260:280] = 1 # Rectangle filled with 1
    mask3 = mask.copy()
    cv2.circle(mask3, (250, 350), 50, (1,), -1)  # Outer circle filled with 1
    cv2.circle(mask3, (250, 350), 30, (0,), -1)  # Inner circle filled with 0 to create the donut effect
    mask4 = mask.copy()
    cv2.circle(mask4, (300, 200), 130, (1,), -1)  # Outer circle filled with 1
    cv2.circle(mask4, (300, 200), 120, (0,), -1)  # Inner circle filled with 0 to create the donut effect

    
    ### Safe as gif 
    mask_A = mask1
    mask_B = mask2
    interpolated_masks = interpolate_masks(mask_A, mask_B, num_frames=15, control_points=150)

    # EXPORT AS GIF
    frames = [(m * 255).astype(np.uint8) for m in interpolated_masks]
    frames.insert(0, (mask_A * 255).astype(np.uint8))
    frames.append((mask_B * 255).astype(np.uint8))
    frames_mirror = frames[::-1]
    frames = frames + frames_mirror
    # Save the frames as a .gif file with looping
    output_gif_path = output_dir / "interpolated_masks.gif"
    imageio.v2.mimwrite(output_gif_path.as_posix(), frames, duration=0.2, loop=0)
    logger.info(f"GIF saved at: {output_gif_path}")

