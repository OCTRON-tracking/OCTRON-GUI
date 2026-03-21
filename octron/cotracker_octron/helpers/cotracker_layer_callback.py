import numpy as np
from napari.utils.notifications import show_warning

from octron.cotracker_octron.helpers.cotracker_zarr import (
    create_trajectory_zarr,
    mark_keypoints_annotated,
    unmark_keypoints_annotated,
    write_frame_tracks,
)


class cotracker_octron_callbacks:
    def __init__(self, octron):
        self.octron = octron  # widget
        self.viewer = octron._viewer

        self.skeleton = self.octron.skeleton_definition
        self.object_organizer = self.octron.object_organizer  # has obj_id
        # self.octron.gui — GUI elements like the keypoint combobox

    def on_points_changed(self, event):
        """When user adds a point, keep track of its skeleton idx and
        auto-advance the combobox.

        Note: if a point is deleted, napari deletes the corresponding row in
        features_df, so no need to update that manually.
        """
        action = event.action
        points_layer = event.source
        keypoint_combobox = self.octron.keypoint_combobox

        if not self.octron.predictor:
            show_warning("No model loaded.")
            return

        # If all points are removed
        if not len(points_layer.data):
            return

        # If a point is added
        if action == "added":
            # Get skeleton index for the current keypoint
            skeleton_idx = keypoint_combobox.currentIndex()

            # keep track of the skeleton index of the added point in
            # the layer's features dataframe
            features_df = points_layer.features
            features_df.loc[len(features_df)] = {"skeleton_idx": skeleton_idx}

            # cycle to next keypoint in combobox
            next_idx = skeleton_idx + 1
            keypoint_combobox.setCurrentIndex(
                next_idx % keypoint_combobox.count(),
            )
        elif action == "removing":
            # If zarr store exists, remove annotated mark for deleted point 
            # NOTE: napari emits REMOVING before the deletion,
            # so self.data still has the original rows
            zarr_root = points_layer.metadata.get("_zarr_root")
            if zarr_root is not None:
                # For every deleted point, set annotated status to False
                for idx in event.data_indices:
                    frame = int(points_layer.data[idx, 0])
                    skeleton_idx = int(points_layer.features.iloc[idx]["skeleton_idx"])
                    unmark_keypoints_annotated(zarr_root, frame, skeleton_idx)
            return
        else:
            return

    def batch_predict(self):
        """Generator that calls predictor.propagate_in_video(start, end),
        and for each yielded (frame_idx, tracks_per_obj),
        calls write_frame_tracks() to persist to zarr, then yields
        results for napari display."""
        # Gather points from cotracker annotation layers
        list_cotracker_annot_layers = [
            ly
            for ly in self.object_organizer.get_annotation_layers()
            if ly.metadata.get("_model_type") == "cotracker"
        ]

        # If there are no cotracker layers: return
        if not list_cotracker_annot_layers:
            return

        # --------------------------
        # Initialise params
        # NOTE: reset cotracker model before registering points
        self.octron.predictor.reset_state()
        map_obj_id_to_zarr_root = {}

        # Loop thru annotation layers to create zarr store
        # per layer and add points to cotracker model
        for layer in list_cotracker_annot_layers:
            # Get layer object ID
            obj_id = layer.metadata["_obj_id"]

            # Get zarr root for this layer
            # (create if it doesnt exist)
            zarr_root = layer.metadata.get("_zarr_root", None)
            if zarr_root is None:
                # TODO: remove spaces from zarr name (kept here for consistency
                # with the rest of the codebase)
                zarr_name = f"{layer.name} tracks"
                zarr_path = self.octron.project_path_video / f"{zarr_name}.zarr"

                zarr_root = create_trajectory_zarr(
                    zarr_path,
                    self.octron.video_layer.metadata["num_frames"],
                    self.skeleton.n_keypoints,
                    video_hash_abbrev=self.octron.current_video_hash,
                )
                layer.metadata["_zarr_root"] = zarr_root

            # Link each layer to a zarr trajectory store
            map_obj_id_to_zarr_root[obj_id] = zarr_root

            # Register points in layer in cotracker model
            # We register points per frame
            unique_frames = np.unique(layer.data[:, 0])
            for frame_idx in unique_frames:
                mask_rows = layer.data[:, 0] == frame_idx
                xy = layer.data[mask_rows, 1:][:, ::-1]  # (y, x) → (x, y)

                # register in cotracker model
                self.octron.predictor.add_new_points(frame_idx, obj_id, xy)

                # Mark keypoints as annotated in zarr "annotated" array
                skeleton_idxs = layer.features.loc[mask_rows, "skeleton_idx"].astype(
                    int
                )
                mark_keypoints_annotated(
                    map_obj_id_to_zarr_root[obj_id],
                    frame_idx,
                    skeleton_idxs,
                )
        # ----------------

        # Define skip frames
        skip_frames = self.octron.skip_frames_spinbox.value()
        step_frames = skip_frames + 1

        # Get range of frames for prediction
        # self.octron.chunk_size is the number of frames
        # the user wants to predict per batch (default 15)
        current_frame = self.viewer.dims.current_step[0]
        num_frames = self.octron.video_layer.metadata["num_frames"]
        end_frame = min(
            num_frames - 1,
            current_frame + self.octron.chunk_size * step_frames,
        )

        # Run prediction on frames
        counter = 0
        for (
            frame_idx,
            tracks_per_obj_one_frame,
        ) in self.octron.predictor.propagate_in_video(
            current_frame, end_frame, step=step_frames
        ):
            counter += 1

            # Loop thru object IDs per frame
            # (object IDs map to annot layers)
            for obj_id, xyv in tracks_per_obj_one_frame.items():
                write_frame_tracks(
                    map_obj_id_to_zarr_root[obj_id]["tracks"],
                    frame_idx,
                    xyv,
                )

            yield counter, frame_idx, tracks_per_obj_one_frame

    def next_predict(self):
        "Single-frame variant (or"
        "possibly unnecessary if CoTracker always runs in "
        "batch windows)."
        pass
