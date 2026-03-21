from napari.utils.notifications import show_warning
import numpy as np


class cotracker_octron_callbacks:
    def __init__(self, octron):
        self.octron = octron  # widget
        self.viewer = octron._viewer

        # keep track of added keypoints and zarr keypoint index
        # is a list where position i = "CoTracker's point i
        # corresponds to keypoint index skeleton_idx_per_cotracker_pt[obj_id][i]"
        # self.skeleton_idx_per_cotracker_pt = defaultdict(list)
        # self.skeleton = self.octron.skeleton_definition
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

        # If action is not an added point return
        if action != "added":
            return
        else:
            # read skeleton index for the current keypoint
            skeleton_idx = keypoint_combobox.currentIndex()

            # keep track of the skeleton index of the added point in
            # the layer's features dataframe
            features_df = points_layer.features
            features_df.loc[len(features_df)] = {"skeleton_idx": skeleton_idx}

            # cycle to next keypoint in combobox
            next_idx = skeleton_idx + 1
            keypoint_combobox.setCurrentIndex(next_idx % keypoint_combobox.count())

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

        # Register points in cotracker model (reset first)
        self.octron.predictor.reset_state()
        for layer in list_cotracker_annot_layers:

            # we register per frame
            obj_id = layer.metadata["_obj_id"]
            unique_frames = np.unique(layer.data[:, 0])
            for frame_idx in unique_frames:
                mask_rows = layer.data[:, 0] == frame_idx
                xy = layer.data[mask_rows, 1:][:, ::-1]  # (y, x) → (x, y)
                self.octron.predictor.add_new_points(frame_idx, obj_id, xy)

        # Define skip frames
        skip_frames = max(1, self.octron.skip_frames_spinbox.value())
        step_frames = skip_frames + 1

        # Get range of frames for prediction
        # self.octron.chunk_size is the number of frames
        # the user wants to predict per batch (15 or 1)
        current_frame = self.viewer.dims.current_step[0]
        num_frames = self.octron.video_layer.metadata["num_frames"]
        end_frame = min(
            num_frames - 1,
            current_frame + self.octron.chunk_size * step_frames,
        )  # last case is for a batch of 15

        # Run prediction on frames
        counter = 0
        for (
            frame_idx,
            tracks_per_obj,
        ) in self.octron.predictor.propagate_in_video(
            current_frame,
            end_frame,
            step=step_frames,
        ):
            counter += 1
            # TODO NEXT
            # for obj_id, xyv in tracks_per_obj.items():

    def next_predict(self):
        "Single-frame variant (or"
        "possibly unnecessary if CoTracker always runs in "
        "batch windows)."
        pass
