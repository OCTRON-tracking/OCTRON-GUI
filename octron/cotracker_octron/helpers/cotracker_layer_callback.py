from collections import defaultdict

from napari.utils.notifications import show_warning


class cotracker_octron_callbacks:
    def __init__(self, octron):
        self.octron = octron  # widget
        self.viewer = octron._viewer

        # keep track of added keypoints and zarr keypoint index
        # is a list where position i = "CoTracker's point i
        # corresponds to keypoint index skeleton_idx_per_cotracker_pt[obj_id][i]"
        self.skeleton_idx_per_cotracker_pt = defaultdict(list)
        # self.skeleton = self.octron.skeleton_definition
        # self.object_organizer = self.octron.object_organizer # has obj_id
        # self.octron.gui — GUI elements like the keypoint combobox

    def on_points_changed(self, event):
        """When user clicks a point, read the current keypoint
        from the combobox, register in predictor and auto-advance the combobox."""
        action = event.action
        points_layer = event.source
        keypoint_combobox = self.octron.keypoint_combobox

        if not self.octron.predictor:
            show_warning("No model loaded.")
            return
        
        # If all points are removed
        if not len(points_layer.data):
            return

        if action in ["added", "removed", "changed"]:
            # read frame idx from slider
            frame_idx = self.viewer.dims.current_step[0]

            # read the index for the current keypoint
            # from the combobox
            keypoint_index = keypoint_combobox.currentIndex()

            # read object ID from the layer that triggers event
            # (each layer is a different object ID)
            obj_id = points_layer.metadata["_obj_id"]

            if action == "added":
                # add the point to the predictor register
                # NOTE: points are expected in xy coords
                new_point_yx = points_layer.data[-1]  # last added point
                self.octron.predictor.add_new_points(
                    frame_idx,
                    obj_id,
                    new_point_yx[::-1],  # flip yx (napari) ->  xy (octron)
                )

                # keep track of the skeleton index of points passed to cotracker
                # (this is to correctly save results to zarr later)
                self.skeleton_idx_per_cotracker_pt[obj_id].append(keypoint_index)

                # auto-advance combobox and cycle
                next_idx = keypoint_index + 1
                keypoint_combobox.setCurrentIndex(next_idx % keypoint_combobox.count())

            elif action == "removed":
                # remove
                pass
            elif action == "changed":
                # update changed point
                pass
            else:
                return  # ok?

        else:
            return  # ok?

    def batch_predict(self):
        """Generator that calls predictor.propagate_in_video(start, end),
        and for each yielded (frame_idx, tracks_per_obj, obj_ids),
        calls write_frame_tracks() to persist to zarr, then yields
        results for napari display."""
        # starts by reading all points from the annotation layers, 
        # registers them with the predictor, then runs propagation
        pass

    def next_predict(self):
        "Single-frame variant (or"
        "possibly unnecessary if CoTracker always runs in "
        "batch windows)."
        pass
