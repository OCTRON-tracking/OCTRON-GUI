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
            # the layer's features dataframe (napari already added a row
            # for the new point, so we update it rather than appending)
            features_df = points_layer.features
            features_df.iloc[-1, features_df.columns.get_loc("skeleton_idx")] = skeleton_idx

            # Set face color for the new point based on keypoint identity
            kpt_colors = points_layer.metadata.get("_keypoint_colors")
            if kpt_colors is not None:
                colors = list(points_layer.face_color)
                colors[-1] = kpt_colors[skeleton_idx]
                points_layer.face_color = colors

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
        """
        1. Gather cotracker annotation layers
        2. Init predictor + zarr stores
        3. yield from predictor model and write to zarr
        """
        # Gather points from cotracker annotation layers
        list_cotracker_annot_layers = [
            ly
            for ly in self.octron.object_organizer.get_annotation_layers()
            if ly.metadata.get("_model_type") == "cotracker"
        ]

        # If there are no cotracker layers: return
        if not list_cotracker_annot_layers:
            return

        # Initialise points in predictor and zarr stores per layer
        self.map_obj_id_to_zarr_root = self._init_predictor_and_zarr_stores(
            list_cotracker_annot_layers,
        )

        # Run prediction and write results to zarr
        yield from self._propagate_and_write_to_zarr(
            list_cotracker_annot_layers, self.map_obj_id_to_zarr_root
        )

    def _init_predictor_and_zarr_stores(self, annotation_layers):
        """Reset predictor, ensure each layer has a zarr store,
        and register all annotated points in the cotracker model.

        It adds newly created zarr stores to layer.metadata["_zarr_root"].

        Returns {obj_id: zarr_root}.
        """
        # Reset predictor state (i.e. any previously registered points)
        self.octron.predictor.reset_state()

        # Loop thru layers
        map_obj_id_to_zarr_root = {}
        for layer in annotation_layers:
            obj_id = layer.metadata["_obj_id"]

            # Get zarr root for this layer
            # (create if it doesnt exist)
            zarr_root = layer.metadata.get("_zarr_root", None)
            if zarr_root is None:
                # TODO: remove spaces from zarr name
                # (kept here only for consistency with the rest of codebase)
                zarr_name = f"{layer.name} tracks"
                zarr_path = self.octron.project_path_video / f"{zarr_name}.zarr"

                n_frames_video = self.octron.video_layer.metadata["num_frames"]
                n_total_kpts = self.octron.skeleton_definition.n_keypoints
                zarr_root = create_trajectory_zarr(
                    zarr_path,
                    n_frames_video,
                    n_total_kpts,
                    video_hash_abbrev=self.octron.current_video_hash,
                )
                layer.metadata["_zarr_root"] = zarr_root

            map_obj_id_to_zarr_root[obj_id] = zarr_root

            # Register points per frame in cotracker model
            if not len(layer.data):
                continue
            unique_frames = np.unique(layer.data[:, 0])
            for frame_idx in unique_frames:
                # Get points
                mask_rows = layer.data[:, 0] == frame_idx
                xy = layer.data[mask_rows, 1:][:, ::-1].copy()  # (y, x) → (x, y)

                # Add to cotracker model
                self.octron.predictor.add_new_points(int(frame_idx), obj_id, xy)

                # Mark keypoints as annotated in zarr "annotated" array
                skeleton_idxs = layer.features.loc[mask_rows, "skeleton_idx"].astype(
                    int
                )
                mark_keypoints_annotated(
                    map_obj_id_to_zarr_root[obj_id],
                    frame_idx,
                    skeleton_idxs,
                )

        return map_obj_id_to_zarr_root

    def _propagate_and_write_to_zarr(
        self, annotation_layers, map_obj_id_to_zarr_root, chunk_override=None
    ):
        """Run predictor over the frame range and write tracks to zarr.

        Yields (counter, frame_idx, tracks_per_obj_one_frame, last_run).
        """
        # Get map from obj_id to annotation layer
        map_obj_id_to_layer = {ly.metadata["_obj_id"]: ly for ly in annotation_layers}

        # Get range of frames to run prediction
        current_frame, end_frame, step_frames = self._get_prediction_frame_range(
            chunk_override=chunk_override,
        )
        n_frames_to_process = len(range(current_frame, end_frame + 1, step_frames))

        # Run prediction
        counter = 0
        last_run = False
        for (
            frame_idx,
            tracks_per_obj_one_frame,
        ) in self.octron.predictor.propagate_in_video(
            current_frame, end_frame, step=step_frames
        ):
            # Loop thru object IDs per frame
            for obj_id, xyv in tracks_per_obj_one_frame.items():
                # Get skeleton indices for points in this layer
                layer = map_obj_id_to_layer[obj_id]
                skeleton_indices = layer.features["skeleton_idx"].astype(int).values

                # Build x,y,vis array for all keypoints in skeleton
                # (place each tracked point at its skeleton column)
                full_xyv = np.zeros(
                    (self.octron.skeleton_definition.n_keypoints, 3), dtype=np.float32
                )
                for i, skel_idx in enumerate(skeleton_indices):
                    full_xyv[skel_idx] = xyv[i].cpu().numpy()

                # Write full xyv array to zarr
                write_frame_tracks(
                    map_obj_id_to_zarr_root[obj_id]["tracks"],
                    frame_idx,
                    full_xyv,
                )

            # Check if last frame processed
            counter += 1
            last_run = counter == n_frames_to_process

            yield counter, frame_idx, tracks_per_obj_one_frame, last_run

    def _get_prediction_frame_range(self, chunk_override=None):
        """Return (current_frame, end_frame, step_frames) for prediction.

        Parameters
        ----------
        chunk_override : int, optional
            If given, use this instead of self.octron.chunk_size.
        """
        skip_frames = self.octron.skip_frames_spinbox.value()
        step_frames = skip_frames + 1

        chunk_size = chunk_override or self.octron.chunk_size
        current_frame = self.viewer.dims.current_step[0]
        num_frames = self.octron.video_layer.metadata["num_frames"]
        end_frame = min(
            num_frames - 1,
            current_frame + chunk_size * step_frames,
        )
        return current_frame, end_frame, step_frames

    def next_predict(self):
        """Predict next window of frames (CoTracker's minimum unit).

        CoTracker requires multi-frame context, so this processes
        ``predictor.step * 2`` frames (default 16) and yields
        results for all of them, rather than a single frame.
        """
        # Gather cotracker annotation layers
        list_cotracker_annot_layers = [
            ly
            for ly in self.octron.object_organizer.get_annotation_layers()
            if ly.metadata.get("_model_type") == "cotracker"
        ]

        if not list_cotracker_annot_layers:
            return

        self.map_obj_id_to_zarr_root = self._init_predictor_and_zarr_stores(
            list_cotracker_annot_layers,
        )

        yield from self._propagate_and_write_to_zarr(
            list_cotracker_annot_layers,
            self.map_obj_id_to_zarr_root,
            chunk_override=self.octron.predictor.step * 2,
        )
