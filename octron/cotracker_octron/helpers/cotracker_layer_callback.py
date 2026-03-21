from napari.utils.notifications import show_warning


class cotracker_octron_callbacks:
    def __init__(self, octron):
        self.octron = octron  # widget
        self.viewer = octron._viewer

        # keep track of added keypoints and zarr keypoint index
        # is a list where position i = "CoTracker's point i
        # corresponds to keypoint index skeleton_idx_per_cotracker_pt[obj_id][i]"
        # self.skeleton_idx_per_cotracker_pt = defaultdict(list)
        # self.skeleton = self.octron.skeleton_definition
        # self.object_organizer = self.octron.object_organizer # has obj_id
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
        and for each yielded (frame_idx, tracks_per_obj, obj_ids),
        calls write_frame_tracks() to persist to zarr, then yields
        results for napari display."""
        # starts by reading all points from the annotation layers,
        # registers them with the predictor, then runs propagation

        # register points in points layers for cotracker
        # add the point to the predictor register
        # NOTE: points are expected in xy coords
        # new_point_yx = points_layer.data[-1]  # last added point
        # self.octron.predictor.add_new_points(
        #     frame_idx,
        #     obj_id,
        #     new_point_yx[::-1],  # flip yx (napari) ->  xy (octron)
        # )

        # define skip frames


        # prefetch images


        # run predict



        pass

    def next_predict(self):
        "Single-frame variant (or"
        "possibly unnecessary if CoTracker always runs in "
        "batch windows)."
        pass
