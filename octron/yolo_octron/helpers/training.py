"""YOLO training related helpers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from loguru import logger


def find_files_with_depth_limit(base_path, pattern, max_depth=1):
    """Find files matching a pattern with a maximum depth limit.

    Parameters
    ----------
    base_path : Path
        Base directory to start the search
    pattern : str
        File pattern to match (e.g., '*.json')
    max_depth : int
        Maximum directory depth to search (0 = base_path only)

    Returns
    -------
    list
        List of Path objects matching the pattern within the depth limit

    """
    results = []

    # A non-existent directory yields no matches. This avoids a
    # FileNotFoundError from glob/iterdir on a path that hasn't been
    # created yet.
    if not base_path.is_dir():
        return results

    # Process base directory (depth 0)
    for path in base_path.glob(pattern):
        if path.is_file():
            results.append(path)

    # Process subdirectories up to max_depth
    if max_depth > 0:
        for path in base_path.iterdir():
            if path.is_dir():
                # Recursively search subdirectories with reduced depth
                results.extend(
                    find_files_with_depth_limit(path, pattern, max_depth - 1)
                )

    return results


def load_object_organizer(file_path):
    """Load object organizer .json from disk and return its content.

    Returns the content as a dictionary. The .json file itself has been
    created via the save_to_disk method in the object_organizer class
    (octron.sam_octron.object_organizer.py).

    Parameters
    ----------
    file_path : str or Path
        Path to the .json file.

    Returns
    -------
    dict
        Dictionary containing all object organizer data.

    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"No organizer file found at {file_path}")
        return
    if file_path.suffix != ".json":
        logger.error(f"File is not a json file: {file_path}")
        return
    try:
        with open(file_path) as f:
            data = json.load(f)
        logger.info(
            f"Octron object organizer loaded from {file_path.as_posix()}"
        )
        return data
    except Exception as e:
        logger.error(f"Error loading json: {e}")
        return


def find_common_frames(frame_arrays):
    """Find frame indices that are present in all provided arrays.

    Parameters
    ----------
    frame_arrays : list of numpy.ndarrays
        Numpy arrays containing frame indices

    Returns
    -------
    numpy.ndarray
        Array containing only the frame indices present in all input arrays

    """
    if not frame_arrays:
        return np.array([], dtype=int)

    if len(frame_arrays) == 1:
        return frame_arrays[0]

    # Start with the first array
    common = frame_arrays[0]

    # Sequentially intersect with each additional array
    for frames in frame_arrays[1:]:
        common = np.intersect1d(common, frames)
        # Early exit if no common frames are found
        if len(common) == 0:
            break
    return common


def pick_random_frames(frames, n=20):
    """Pick n random frames from a frames array without replacement.

    Parameters
    ----------
    frames : numpy.ndarray
        Array of frame indices
    n : int
        Number of frames to pick

    Returns
    -------
    numpy.ndarray
        Array of randomly selected frame indices

    """
    # Determine the number of frames to pick (min of n and array length)
    num_to_pick = min(n, len(frames))

    # Pick random frames without replacement
    if num_to_pick > 0:
        random_frames = np.random.choice(
            frames, size=num_to_pick, replace=False
        )
        # Sort the frames to maintain chronological order if needed
        random_frames.sort()
        return random_frames
    else:
        return np.array([], dtype=frames.dtype)


def collect_labels(
    project_path,
    subfolder=None,
    prune_empty_labels=True,
    min_num_frames=5,
    verbose=False,
    verify_hash=False,
):
    """Extract info from project path.

    Find all the object organizer json files and load them.
    The object organizer json files contain the information about the
    annotations (the zarr arrays) as well as the associated video files.
    Both object organizer as well as zarr arrays (as attribute) contain
    the video hash. This hash is used to check if the correct video
    file is associated with the zarr array.

    Sanity checks:
    1. Data shape info in the zarr array and
       the object organizer json file match the actual video file shape
    2. The video hash in the zarr array and
       the object organizer json file match the actual video file hash
    3. The label id to label name association is congruent across all entries


    Parameters
    ----------
    project_path : str or Path : Path to the project root directory.
        The jsons are saved in subfolders.
    subfolder : str or Path, optional
        If given, only search for object organizer json files within
        this subfolder of project_path. Defaults to None (search the
        whole project root directory).
    prune_empty_labels : bool : Whether to prune frames that
                                do not have annotation across all labels.
    min_num_frames : int : Minimum number of frames required
                           for training data generation.
    verbose : bool : Whether to print debug info.
        Default is False.
    verify_hash : bool : Whether to recompute the video file hash and
        check it against the stored one. This reads the whole video
        file, so it is slow for large files. Default is False.

    Returns
    -------
    label_dict : dict : Dictionary containing project subfolders,
                        and their corresponding labels,
                        annotated frames, masks and video data.
            keys: project_subfolder
            values: dict
                keys: label_id, video
                values: dict, video
                    dict:
                        keys: label, frames, masks, color
                        values: label (str), # Name of the label
                                    corresponding to current ID
                                frames (np.array), # Annotated frame
                                    indices for the label
                                masks (list of zarr arrays), # Mask
                                    zarr arrays
                                color (list) # Color of the label
                                    (RGBA, [0,1])
                    video: FastVideoReader object

    """
    project_path = Path(project_path)
    assert project_path.exists(), (
        f"Project path not found at {project_path.as_posix()}"
    )
    assert project_path.is_dir(), (
        "Project path should be a directory, not file"
    )

    # Check whether .json files should be found in only a subfolder
    if subfolder is not None:
        json_parent_path = project_path / subfolder
        # The per-video subfolder is only created once annotation data has been
        # saved.
        if not json_parent_path.is_dir():
            return {}
    else:
        # If no subfolder is provided, search in the project root directory
        json_parent_path = project_path

    # Hiding some imports here to reduce initial loading time
    from napari_pyav._reader import FastVideoReader

    from octron.sam_octron.helpers.sam_zarr import (
        get_annotated_frames,
        load_image_zarr,
    )

    label_dict = {}
    # Create a (new) global mapping of label names to IDs for
    # consistency across directories. This is important since the
    # user might decide to add multiple labels with the same name
    # but in different orders across video projects. I.e. a label
    # "worm" might have ID 0 in one project and ID 1 in another. We
    # want to make sure that the label ID to label name association
    # is consistent across all projects.
    label_id_map = {}
    current_label_id = 0

    for object_organizer in find_files_with_depth_limit(
        json_parent_path, "object_organizer.json", 1
    ):
        if verbose:
            logger.debug(object_organizer.parent)
        organizer_dict = load_object_organizer(object_organizer)
        labels = {}
        video_hash_dict = {}

        for entry in organizer_dict["entries"].values():
            original_label_id = int(entry["label_id"])
            label = entry["label"]
            color = entry["color"]

            # Check if label already exists in labels
            if label in label_id_map:
                # Use existing ID for consistency
                label_id = label_id_map[label]
                if verbose:
                    logger.debug(
                        f"Using existing ID {label_id} for label {label}"
                    )
            else:
                # Assign a new ID and update mapping
                label_id = current_label_id
                label_id_map[label] = label_id
                current_label_id += 1
                if verbose:
                    logger.debug(
                        f"Created new ID {label_id} for label {label}"
                    )

            if verbose:
                logger.debug(f"Found label {label} with id {label_id}")
            if label_id in labels:
                assert labels[label_id]["label"] == label, (
                    "Label name vs. id do not match"
                )
            else:
                # First time we see this label
                labels[label_id] = {
                    "label": label,
                    "original_id": original_label_id,
                    "frames": [],
                    "masks": [],
                    "color": color,
                }

            # Find out which frames were annotated
            zarr_path_relative = Path(
                entry["prediction_layer_metadata"]["zarr_path"]
            )
            zarr_path = project_path / zarr_path_relative
            assert zarr_path.exists(), f"Zarr file not found at {zarr_path}"
            num_frames, image_height, image_width = entry[
                "prediction_layer_metadata"
            ]["data_shape"]
            # Feed the expected shape to the loader.
            loaded_masks, status = load_image_zarr(
                zarr_path,
                num_frames,
                image_height,
                image_width,
                num_ch=None,
                verbose=False,
            )  # Not doing hash comparison here!
            assert status
            assert loaded_masks is not None
            # Do some sanity checks
            assert num_frames == loaded_masks.shape[0]
            assert image_height == loaded_masks.shape[1]
            assert image_width == loaded_masks.shape[2]
            labels[label_id]["masks"].append(
                loaded_masks
            )  # This is the zarr array
            # Extract annotated frame indices from zarr attribute (fast path)
            annotated_indices = get_annotated_frames(loaded_masks)
            if verbose:
                logger.info(
                    f"Found {len(annotated_indices)} annotated frames "
                    f"for label {label} in "
                    f"{object_organizer.parent.name}"
                )
            # if prune_empty_labels:

            #     # Also get rid of frames where the mask is all zeros
            #     # Why?
            #     # Because frames that are not annotated and
            #     # accidentally skipped contribute to
            #     # "background" masks in YOLO. This will just spoil
            #     # the actual training success.
            #     summed = np.sum(loaded_masks, axis=(1,2))
            #     # TODO: This is a heavy computation!!
            #     annotated_indices = np.intersect1d(
            #         annotated_indices, np.where(summed > 0)[0]
            #     )
            #     if verbose:
            #         print(
            #             f'PRUNING: {len(annotated_indices)} remain '
            #             'after removing empty frames'
            #         )
            labels[label_id]["frames"].extend(annotated_indices)

            expected_video_hash_zarr = loaded_masks.attrs.get(
                "video_hash", None
            )
            expected_video_hash_organizer = entry["prediction_layer_metadata"][
                "video_hash"
            ]

            # Resolve so the same physical video referenced via
            # different path strings maps to one key (keeps the
            # single-video assert below honest).
            video_file_path = (
                project_path
                / Path(entry["prediction_layer_metadata"]["video_file_path"])
            ).resolve()
            if video_file_path not in video_hash_dict:
                assert video_file_path.exists(), (
                    f'Video file not found at "{video_file_path.as_posix()}"'
                )
                if verify_hash:
                    # Full hash: reads the entire video file — slow for
                    # large files. Only do this when explicitly
                    # requested (e.g. pre-training integrity check).
                    from octron.sam_octron.helpers.video_loader import (
                        get_vfile_hash,
                    )

                    actual_video_hash = get_vfile_hash(video_file_path)[-8:]
                    video_hash_dict[video_file_path] = actual_video_hash
                    assert (
                        expected_video_hash_zarr
                        == expected_video_hash_organizer
                        == actual_video_hash
                    ), "Video hash mismatch"
                else:
                    # Fast path: just cross-check zarr attrs vs organizer
                    # JSON. Avoids reading the entire video file on every
                    # project load.
                    video_hash_dict[video_file_path] = (
                        expected_video_hash_organizer
                    )
                    assert (
                        expected_video_hash_zarr
                        == expected_video_hash_organizer
                    ), (
                        "Hash mismatch between zarr "
                        f"({expected_video_hash_zarr}) and organizer "
                        f"({expected_video_hash_organizer}) for "
                        f"{video_file_path.name}"
                    )
            assert len(video_hash_dict) == 1, (
                "Different video files found for one object organizer json."
            )

        # An organizer with no entries leaves no video/labels — skip it instead
        # of failing later with an unbound video_file_path.
        if not video_hash_dict:
            logger.warning(
                f"Object organizer '{object_organizer.parent.name}' "
                f"has no entries; skipping."
            )
            continue

        # Maintain only unique entries in 'frames' lists
        for label_id in labels:
            _, i = np.unique(labels[label_id]["frames"], return_index=True)
            labels[label_id]["frames"] = np.array(labels[label_id]["frames"])[
                np.sort(i)
            ]
            if verbose:
                logger.debug(
                    f"Label {labels[label_id]['label']} has "
                    f"{len(labels[label_id]['frames'])} annotated "
                    f"frames"
                )

        # Prune frames that do not have annotation across all labels
        if prune_empty_labels:
            common_frames = find_common_frames(
                [f["frames"] for f in labels.values()]
            )
            for label_id in labels:
                labels[label_id]["frames"] = common_frames
                if verbose:
                    logger.debug(
                        f"PRUNING: Label {labels[label_id]['label']} "
                        f"has {len(labels[label_id]['frames'])} "
                        f"common frames"
                    )

        # Assert that there is a minimum number of frames available
        # for training data generation
        if min_num_frames > 0:
            for label_id in labels:
                no_frames_label = len(labels[label_id]["frames"])
                label = labels[label_id]["label"]
                path = object_organizer.parent.as_posix()
                assert no_frames_label >= min_num_frames, (
                    f'Not enough frames for label "{label}" in '
                    f'"{path}": {no_frames_label} < {min_num_frames}'
                )

        # Add the video file path and data to the dictionary
        # video_file_path is generated anew for every object, however,
        # we are making sure above that all videos are the same.
        video = FastVideoReader(video_file_path)
        labels["video_file_path"] = video_file_path
        labels["video"] = video
        label_dict[object_organizer.parent.as_posix()] = labels

    # Assert that label_id to label associations are congruent across
    # label_dict, i.e. the numerical label_id is always associated
    # with the same label name across all entries
    label_ids = []
    label_idnames = []
    for labels in label_dict.values():
        for label_id in labels:
            if label_id == "video" or label_id == "video_file_path":
                continue
            label_ids.append(label_id)
            label_idnames.append(f"{label_id}-{labels[label_id]['label']}")
    assert len(set(label_ids)) == len(set(label_idnames)), (
        "A label id to label name association is not congruent "
        "across label_dict"
    )

    return label_dict


def draw_polygons(
    labels,
    video_data,
    max_to_plot=2,
    randomize=False,
):
    """Draw the polygons on the video frames.

    Frames come from the labels dictionary created via the
    collect_labels() function.

    Parameters
    ----------
    labels : dict : Dictionary containing labels and their corresponding frames
            keys: label_id
            values: dict
                keys: label, frames, masks, polygons, color
                values: label (str), # Name of the label
                         frames (np.array), # Annotated frame indices
                             for the label
                         masks (list of zarr arrays), # Masks for
                             each frame
                         polygons (dict) # Polygons for each frame
                             index
                         color (list) # Color of the label
                             (RGBA, [0,1])
    video_data : np.array : Video data array
        Array of video frames to draw the polygons onto.
    max_to_plot : int : Maximum number of frames to plot per label
        Default is 2.
    randomize : bool : Whether to plot random frames
        Default is False.

    """
    # Check if cv2 is installed correctly
    if importlib.util.find_spec("cv2") is None:
        logger.error("Please install cv2 first, via pip install opencv-python")
        return
    # ... and matplotlib
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.error(
            "Please install matplotlib first, via pip install matplotlib"
        )
        return

    if max_to_plot < 1:
        max_to_plot = 1
    logger.info(f"Drawing polygons for {len(labels)} labels.")
    logger.info(f"Max {max_to_plot} frame(s) per label will be plotted.")
    # Draw the polygons on the video frames
    for entry in labels:
        if entry == "video" or entry == "video_file_path":
            continue

        label = labels[entry]["label"]
        frames = labels[entry]["frames"]
        if randomize:
            frames = pick_random_frames(frames, n=max_to_plot)
        # color = np.array(labels[entry]['color'])[:-1]*255
        counter = 1
        for frame in frames:
            current_frame = video_data[frame].copy()
            polys = labels[entry]["polygons"][frame]

            # # cv2.polylines() is fine but introduces some nasty artefacts in
            # # cases where the polygons are not closed.
            # frame_and_polys = cv2.polylines(img=current_frame,
            #                                 pts=polys,
            #                                 isClosed=True,
            #                                 color=color.tolist(),
            #                                 thickness=5,
            #                                 )

            # Draw
            _, ax = plt.subplots(1, 1)
            ax.imshow(current_frame)

            # Draw polys as dots
            for no_p, p in enumerate(polys):
                ax.scatter(p[:, 0], p[:, 1], c="w", s=2, alpha=0.5, marker="s")
                ax.scatter(
                    p[:, 0], p[:, 1], c="k", s=0.5, alpha=0.5, marker="."
                )
                ax.plot(p[:, 0], p[:, 1], c="w")
                center_coord = p.mean(axis=0)
                ax.text(
                    center_coord[0],
                    center_coord[1],
                    str(no_p),
                    color="w",  # Text color
                    fontsize=10,
                    bbox=dict(
                        facecolor="black",  # Background color
                        alpha=0.7,  # Transparency
                        edgecolor="none",  # No edge color
                        boxstyle="round,pad=0.3",  # Rounded corners
                        # with padding
                    ),
                    ha="center",  # Horizontal alignment
                    va="center",  # Vertical alignment
                )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Label: "{label}" - frame {frame} {len(polys)}')
            plt.show()
            if counter >= max_to_plot:
                break
            counter += 1


def _allocate_split_blocks(n_blocks, validation_fraction, test_fraction):
    """Return ``(n_train, n_val, n_test)`` block counts, each at least 1.

    Fractions are applied at block granularity; val and test each keep at
    least one block and are trimmed so train keeps at least one too.
    """
    n_val = max(1, round(validation_fraction * n_blocks))
    n_test = max(1, round(test_fraction * n_blocks))
    while n_val + n_test >= n_blocks:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    return n_blocks - n_val - n_test, n_val, n_test


def _blocks_to_splits(blocks, split_labels, buffer):
    """Group contiguous blocks into per-split frame-index lists.

    ``split_labels[b]`` is the split name for block ``b``. When ``buffer``
    is > 0 the leading ``buffer`` frames of a block are dropped whenever
    the previous block belongs to a different split, creating a temporal
    gap so no adjacent near-duplicate pair straddles the split.
    """
    buckets = {"train": [], "val": [], "test": []}
    for b, block in enumerate(blocks):
        blk = block
        if buffer and b > 0 and split_labels[b] != split_labels[b - 1]:
            blk = blk[buffer:]
        if len(blk):
            buckets[split_labels[b]].append(blk)
    return buckets


def _concat_blocks(parts, dtype):
    """Concatenate frame-index arrays; empty input -> empty array."""
    if parts:
        return np.concatenate(parts)
    return np.array([], dtype=dtype)


def _assert_valid_split(split_dict, sorted_frames):
    """Assert splits are non-empty, disjoint, and a subset of the input."""
    seen: set[int] = set()
    for name, frames in split_dict.items():
        assert len(frames) > 0, f"Empty {name} split"
        fset = {int(f) for f in frames}
        assert not (fset & seen), "Splits overlap"
        seen |= fset
    assert seen.issubset({int(f) for f in sorted_frames}), (
        "Split contains frames not present in the input"
    )


def _segment_episodes(sorted_frames, gap_threshold, gap_factor, block_size):
    """Split sorted frames into episodes at large temporal gaps.

    An episode boundary is a jump between consecutive annotated frames that
    is much larger than the typical spacing, so annotation bursts recorded
    in different parts of a video become separate episodes. When
    ``gap_threshold`` is None it is derived from the data as
    ``max(block_size, gap_factor * p90(diffs))`` -- a high percentile so a
    sparser episode's normal skip spacing is not mistaken for a boundary.
    """
    if len(sorted_frames) < 2:
        return [sorted_frames]
    diffs = np.diff(sorted_frames)
    if gap_threshold is None:
        typical = np.percentile(diffs, 90)
        threshold = max(block_size, round(gap_factor * typical))
    else:
        threshold = gap_threshold
    boundaries = np.where(diffs > threshold)[0]
    if len(boundaries) == 0:
        return [sorted_frames]
    return np.split(sorted_frames, boundaries + 1)


def _block_split(
    frames, training_fraction, validation_fraction, rng, block_size, buffer
):
    """Contiguous-block split of one already-sorted frame run.

    Cuts ``frames`` into at least 3 contiguous blocks (~``block_size``
    each), assigns whole blocks to train/val/test with ``rng`` (so val and
    test are spread across the run), and drops ``buffer`` frames at
    boundaries between differently-assigned blocks. ``frames`` must have at
    least 3 entries.
    """
    n = len(frames)
    n_blocks = max(3, min(n, round(n / max(1, block_size))))
    blocks = np.array_split(frames, n_blocks)
    n_blocks = len(blocks)
    eff_buffer = buffer if min(len(b) for b in blocks) > buffer else 0

    test_fraction = 1.0 - training_fraction - validation_fraction
    _, n_val, n_test = _allocate_split_blocks(
        n_blocks, validation_fraction, test_fraction
    )
    perm = rng.permutation(n_blocks)
    labels = np.array(["train"] * n_blocks, dtype="<U5")
    labels[perm[:n_val]] = "val"
    labels[perm[n_val : n_val + n_test]] = "test"
    labels = labels.tolist()

    buckets = _blocks_to_splits(blocks, labels, eff_buffer)
    out = {k: _concat_blocks(v, frames.dtype) for k, v in buckets.items()}
    # A split empties only if its single short block was fully consumed by
    # the buffer; retry without the buffer (always non-empty).
    if eff_buffer and any(len(v) == 0 for v in out.values()):
        buckets = _blocks_to_splits(blocks, labels, 0)
        out = {k: _concat_blocks(v, frames.dtype) for k, v in buckets.items()}
    return out


def _cut_into_blocks(frames, block_size):
    """Cut one contiguous frame run into ~``block_size`` blocks."""
    n_blocks = max(1, round(len(frames) / max(1, block_size)))
    return np.array_split(frames, n_blocks)


def _stratified_labels(n_blocks, n_val, n_test, rng):
    """Label blocks so val and test are spread across the timeline.

    The combined val+test *holdout* is placed one block per evenly-spaced
    stratum over the whole block sequence, then those holdout blocks are
    divided into val/test (also stratified). Spreading the holdout as a
    whole -- rather than choosing val, then test, independently -- keeps
    both splits distributed even when only one block of each is available
    (otherwise a lone val and a lone test block often clump together and
    leave a long train-only stretch). Picks use ``rng`` so the split is
    reproducible yet seed-sensitive. Every non-holdout block is ``train``.
    """

    def pick_spread(pool, k):
        return [int(rng.choice(s)) for s in np.array_split(pool, k)]

    labels = np.array(["train"] * n_blocks, dtype="<U5")
    all_idx = np.arange(n_blocks)
    holdout = np.array(sorted(pick_spread(all_idx, n_val + n_test)))
    test_slots = pick_spread(np.arange(len(holdout)), n_test)
    is_test = np.zeros(len(holdout), dtype=bool)
    is_test[test_slots] = True
    labels[holdout[is_test]] = "test"
    labels[holdout[~is_test]] = "val"
    return labels.tolist()


def _global_block_split(
    episodes, training_fraction, validation_fraction, rng, block_size, buffer
):
    """Split episodes against a single global block budget.

    Each episode is cut into ~``block_size`` contiguous blocks and all
    blocks across episodes share one train/val/test budget, so the
    realized proportions track the requested fractions regardless of how
    many (or how small) the episodes are. val/test blocks are spread
    across the timeline and the buffer is applied only within an episode
    (never across the large gap between episodes). Returns ``None`` when
    there are fewer than 3 blocks total so the caller can fall back to a
    single-group split.
    """
    episode_blocks = [_cut_into_blocks(ep, block_size) for ep in episodes]
    n_blocks = sum(len(b) for b in episode_blocks)
    if n_blocks < 3:
        return None

    test_fraction = 1.0 - training_fraction - validation_fraction
    _, n_val, n_test = _allocate_split_blocks(
        n_blocks, validation_fraction, test_fraction
    )
    labels = _stratified_labels(n_blocks, n_val, n_test, rng)

    buckets = {"train": [], "val": [], "test": []}
    pos = 0
    for blocks in episode_blocks:
        ep_labels = labels[pos : pos + len(blocks)]
        pos += len(blocks)
        eff_buffer = buffer if min(len(b) for b in blocks) > buffer else 0
        ep_buckets = _blocks_to_splits(blocks, ep_labels, eff_buffer)
        for name in buckets:
            buckets[name].extend(ep_buckets[name])
    return buckets


def train_test_val(
    frame_indices,
    training_fraction=0.8,
    validation_fraction=0.1,
    random_seed=88,
    block_size=20,
    buffer=1,
    gap_threshold=None,
    gap_factor=5.0,
    verbose=False,
):
    """Split frame indices into train/val/test by episode, then block.

    SAM-assisted annotation (forward propagation)
    lets users label many frames quickly, so the annotated set
    is dominated by temporally adjacent, correlated frames -- often
    across several annotation bursts ("episodes") in different parts of one
    video. A fully random per-frame split places a frame and its close
    neighbour on opposite sides of train/val (optimistic metrics) and lets
    a denser episode dominate val/test.

    This splits by *contiguous blocks pooled across episodes* instead:
    frames are first segmented into episodes at large temporal gaps, each
    episode is cut into small contiguous blocks, and all blocks across
    episodes share one train/val/test budget so the realized proportions
    match the requested fractions no matter how many (or how small) the
    episodes are. The val/test blocks are spread across the timeline
    (stratified) so they stay representative, temporally adjacent frames
    stay on the same side, and a buffer of frames is dropped at block
    boundaries within an episode to add a gap. Small episodes may fall
    entirely in one split (usually train); they are not force-split three
    ways.

    Parameters
    ----------
    frame_indices : np.array
        Frame indices (need not be contiguous; sorted internally).
    training_fraction : float
        Approximate fraction of blocks for training.
    validation_fraction : float
        Approximate fraction of blocks for validation; test is the
        remainder.
    random_seed : int
        Seed for reproducible (timeline-spread) block assignment.
    block_size : int
        Target frames per contiguous block. Adapted down for small
        episodes so at least 3 blocks (one per split) always exist.
    buffer : int
        Frames dropped at each boundary between differently-assigned
        blocks. Disabled automatically when blocks are too small to spare
        a frame.
    gap_threshold : int or None
        Frame gap above which a new episode begins. None derives it from
        the data (see :func:`_segment_episodes`).
    gap_factor : float
        Multiplier used when deriving the automatic ``gap_threshold``.
    verbose : bool
        Whether to log split sizes.

    Returns
    -------
    split_dict : dict : Dictionary containing the splits
        keys: 'train', 'val', 'test'
        values: np.array : sorted frame indices for each split. Buffered
        boundary frames are intentionally omitted from all splits.

    """
    assert training_fraction + validation_fraction < 1, (
        "Fractions should sum to less than 1"
    )
    assert training_fraction > validation_fraction, (
        "Training fraction should be greater than validation fraction"
    )
    frame_indices = np.asarray(frame_indices)
    n = len(frame_indices)
    assert n >= 3, (
        f"Need at least 3 frames to split into train/val/test, got {n}"
    )

    sorted_frames = np.sort(frame_indices)
    episodes = _segment_episodes(
        sorted_frames, gap_threshold, gap_factor, block_size
    )

    rng = np.random.default_rng(random_seed)
    buckets = _global_block_split(
        episodes,
        training_fraction,
        validation_fraction,
        rng,
        block_size,
        buffer,
    )
    dtype = sorted_frames.dtype
    if buckets is not None:
        split_dict = {k: _concat_blocks(v, dtype) for k, v in buckets.items()}
    else:
        split_dict = None

    # Fallback: too few blocks to spread three ways, or the buffer emptied
    # val/test -- split the whole set as one contiguous block group.
    if (
        split_dict is None
        or len(split_dict["val"]) == 0
        or len(split_dict["test"]) == 0
    ):
        split_dict = _block_split(
            sorted_frames,
            training_fraction,
            validation_fraction,
            np.random.default_rng(random_seed),
            block_size,
            buffer,
        )

    if verbose:
        logger.info(f"Total frames: {n} in {len(episodes)} episode(s)")
        logger.info(f"Training set: {len(split_dict['train'])} frames")
        logger.info(f"Validation set: {len(split_dict['val'])} frames")
        logger.info(f"Test set: {len(split_dict['test'])} frames")

    _assert_valid_split(split_dict, sorted_frames)
    return split_dict
