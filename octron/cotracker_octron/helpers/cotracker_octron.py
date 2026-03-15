from collections import OrderedDict

import torch


class CoTracker_octron:
    """
    OCTRON wrapper around CoTrackerOnlinePredictor for interactive
    point tracking in video.

    Provides a similar interface to SAM2_octron / SAM3_octron:
        - init_state(video_data, zarr_store)
        - add_new_points(frame_idx, obj_id, points)
        - propagate_in_video(start_frame, end_frame)
        - reset_state()
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

        # Cotracker does not require the input image to be a certain
        # size, but internally its working resolution is 384×512. It then
        # scales the predicted tracks back to the original video resolution.
        # For the zarr cache we set it to the actual video resolution to match
        # the napari videos and the cotracker results.
        self.image_size = None

        # step value (which is window_len // 2 = 8 by default)
        # tells you how many new frames to feed per chunk during
        # propagate_in_video:
        self.step = model.step

        # Set float16 on CUDA for faster inference
        if device.type == "cuda":
            self.model = self.model.half()

        # Initialise Per-object query points, populated by add_new_points().
        # {obj_id: {"frame_idx": int, "points": tensor (N, 2)}}
        self.point_inputs_per_obj = OrderedDict()

        # Octron initialisation flag, signals video data + zarr are set up,
        # set to True at the end of init_state()
        self.is_initialized = False

    @torch.inference_mode()
    def add_new_points(self, frame_idx, obj_id, points):
        """Register user-clicked query points.

        Logs clicked points for a given object on a given frame
        and stores in a dict `query_points_per_obj[obj_id][frame_idx]`
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)

        # If object ID is not in dict, initialise
        if obj_id not in self.point_inputs_per_obj:
            self.point_inputs_per_obj[obj_id] = {}

        # Store (or overwrite) points for this object on this frame.
        # Each entry keeps the raw pixel coords in napari axes -
        # we convert to cotracker's (t, x, y) format in propagate_in_video().
        self.point_inputs_per_obj[obj_id][frame_idx] = points

    @torch.inference_mode()
    def propagate_in_video(self, start_frame, end_frame):
        """Run CoTracker3 online tracking over a range of frames.

        Follows the CoTracker online API: feed overlapping video chunks of
        size ``step * 2``, advancing by ``step`` frames each iteration.
        The first chunk initialises the model with query points; subsequent
        chunks return cumulative tracks and visibilities.

        Yields
        ------
        frame_idx : int
            Absolute frame index in the video (0-based).
        obj_ids : list[int]
            Object IDs present in this result.
        tracks_per_obj : dict[int, torch.Tensor]
            Mapping obj_id → (N_points, 3) tensor of (x, y, visibility)
            for this frame.
        """
        # Prepare query tensor and log object-point pairs
        query_tensor, query_point_obj_ids = self._build_query_tensor(start_frame)
        if query_tensor is None:
            return  # nothing to track

        # Process each chunk of frames and yield results for new frames only
        list_obj_ids = list(self.point_inputs_per_obj.keys())
        yield from self._process_overlapping_chunks(
            start_frame,
            end_frame,
            query_tensor,
            query_point_obj_ids,
            list_obj_ids,
        )

        # Process any remaining frames that didn't fill a complete chunk / window
        yield from self._process_tail_chunk(
            query_tensor,
            query_point_obj_ids,
            list_obj_ids,
            start_frame,
        )

    def _build_query_tensor(self, start_frame):
        """Build the CoTracker query tensor from stored per-object points.

        CoTracker expects queries shaped (B, N_total, 3) with columns
        (t, x, y) where t is the frame index relative to start_frame.
        All objects' points are merged into one tensor; a parallel map
        records which query row belongs to which object.

        Returns
        -------
        query_tensor : torch.Tensor or None
            (1, N_total, 3) on self.device, or None if no points exist.
        """
        query_tensor_rows = []  # list of (t, x, y) tensors
        query_point_obj_ids = []  # obj_id for each query row

        list_obj_ids = list(self.point_inputs_per_obj.keys())
        for obj_id in list_obj_ids:
            # Get frames_dict for this object:
            #  {frame_idx: (N, 2) tensor of (y, x) points}
            frames_dict = self.point_inputs_per_obj[obj_id]
            for frame_idx, pts in frames_dict.items():
                # flip pts from napari (y, x) to CoTracker (x, y)
                pts_xy = pts[:, [1, 0]]  # (N, 2)

                # build relative frame column:
                # absolute frame index - start_frame
                relative_frame = torch.full(
                    (pts.shape[0], 1),
                    float(frame_idx - start_frame),
                )

                # concatenate to (t, x, y) format
                txy_rows = torch.cat([relative_frame, pts_xy], dim=1)  # (N, 3)
                query_tensor_rows.append(txy_rows)

                # record which obj_id each query row belongs to
                query_point_obj_ids.extend([obj_id] * pts.shape[0])

        if not query_tensor_rows:
            return None, []
        else:
            query_tensor = torch.cat(query_tensor_rows, dim=0).unsqueeze(0)
            # (1, N_total, 3)
            query_tensor = query_tensor.to(self.device)
            return query_tensor, query_point_obj_ids

    def _process_overlapping_chunks(
        self,
        start_frame,
        end_frame,
        query_tensor,
        query_point_obj_ids,
        list_obj_ids,
    ):
        """Collect frames and process overlapping chunks of size step*2.

        Accumulates frames from OctoZarr into a running window. Every
        ``step`` frames, builds a chunk and feeds it to the model.
        The first chunk initialises the model (is_first_step=True),
        subsequent chunks yield per-frame tracking results.

        Stores internal state (_window_frames, _is_first_step, _n_yielded)
        on self so that _process_tail can continue from where this left off.

        CoTracker uses overlapping windows of step*2 = 16 frames, advancing by step=8 each time:
        Chunk 1 (is_first_step=True):  frames [0..15]  → returns (None, None)
        Chunk 2:                       frames [8..23]  → returns tracks for frames [0..15]
        Chunk 3:                       frames [16..31] → returns tracks for frames [0..23]
        Chunk 4:                       frames [24..39] → returns tracks for frames [0..31]
        It returns cumulative tracks for ALL frames seen so far, not just the new chunk.
        """
        # State variables
        self._buffered_frames = []  # 2*step frames max
        self._is_first_step = True
        self._n_frames_to_yield = 0
        self._n_frames_seen = 0  # total frames seen (not len of window)

        # Loop thru frames for starting the chunks
        for abs_idx in range(start_frame, end_frame + 1):
            # Fetch frame from OctoZarr
            frame_tensor = self.images[abs_idx]  # (C, H, W)
            self._buffered_frames.append(frame_tensor)
            self._n_frames_seen += 1

            # Keep at most step*2 frames in memory to avoid
            # accumulating the entire video
            max_window = self.step * 2
            if len(self._buffered_frames) > max_window:
                self._buffered_frames = self._buffered_frames[-max_window:]

            # Process a chunk every ``step`` frames
            # (but not on the very first frame)
            if self._n_frames_seen % self.step == 0 and self._n_frames_seen != 0:
                # Build torch tensor from video chunk:
                # last step*2 frames -> (1, T, C, H, W)
                chunk = torch.stack(self._buffered_frames, dim=0).unsqueeze(0)

                # If first step, initialise online processing
                # with is_first_step=True and with queries. It does
                # not return any predictions.
                if self._is_first_step:
                    self.model(
                        video_chunk=chunk,
                        is_first_step=True,
                        queries=query_tensor,
                    )
                    # update for next iteration
                    self._is_first_step = False
                # else, just feed the chunk and
                # yield results for the new frames only
                else:
                    pred_tracks, pred_visibility = self.model(
                        video_chunk=chunk,
                    )  # (1, T_total, N, 2), (1, T_total, N)

                    # CoTracker returns cumulative tracks for ALL frames seen
                    # so far, not just the new chunk. Here we yield only the
                    # newly computed frames for this chunk.
                    n_total_frames = pred_tracks.shape[1]
                    for t in range(self._n_frames_to_yield, n_total_frames):
                        abs_frame = start_frame + t
                        tracks_per_obj = self._split_tracks_by_obj(
                            pred_tracks[0, t],  # (N, 2)
                            pred_visibility[0, t],  # (N,)
                            query_point_obj_ids,
                            list_obj_ids,
                        )
                        yield abs_frame, tracks_per_obj, list_obj_ids
                    # We have now yielded all frames up to n_total_frames,
                    # so update the counter for the next chunk
                    self._n_frames_to_yield = n_total_frames

    def _process_tail_chunk(
        self,
        query_tensor,
        query_point_obj_ids,
        list_obj_ids,
        start_frame,
    ):
        """Process remaining frames if video length is not a multiple of step.

        Uses _window_frames, _is_first_step and _n_yielded state left
        by _process_windowed_chunks.
        """
        # Compute remaining n of frames
        n_frames_remain = self._n_frames_seen % self.step
        if n_frames_remain == 0:
            return

        # The tail chunk needs to overlap with the previous window to
        # have context. So we take one step back. See
        # https://github.com/facebookresearch/co-tracker/blob/main/online_demo.py
        n_frames_tail_chunk = n_frames_remain + self.step + 1
        chunk = torch.stack(
            self._buffered_frames[-n_frames_tail_chunk:], dim=0
        ).unsqueeze(0)

        # If we haven't done the first step yet (e.g. if video is shorter than step*2),
        # we need to initialise the model with the queries.
        if self._is_first_step:
            self.model(
                video_chunk=chunk,
                is_first_step=True,
                queries=query_tensor,
            )
            self._is_first_step = False

        # Feed the tail chunk
        pred_tracks, pred_visibility = self.model(video_chunk=chunk)
        # (1, T_total, N, 2), (1, T_total, N)

        # Yield only the unpredicted frames
        n_total_frames = pred_tracks.shape[1]
        for t in range(self._n_frames_to_yield, n_total_frames):
            abs_frame = start_frame + t
            tracks_per_obj = self._split_tracks_by_obj(
                pred_tracks[0, t],  # (N, 2)
                pred_visibility[0, t],  # (N,)
                query_point_obj_ids,
                list_obj_ids,
            )
            yield abs_frame, tracks_per_obj, list_obj_ids

    def _split_tracks_by_obj(
        self,
        predicted_points,
        points_visibility,
        query_point_obj_ids,
        list_obj_ids,
    ):
        """Split flat per-query results back into per-object dicts.

        Parameters
        ----------
        predicted_points
            tensor (N_total, 2) — x, y per query point
        points_visibility
            tensor (N_total,) — visibility per query point
        query_point_obj_ids
            list of obj_id per query row (length N_total)
        list_obj_ids
            list of unique obj_ids

        Returns
        -------
        dict[int, tensor]
            Map from obj_id to (N_points, 3) array of predicted points for 
            this frame, with columns (x, y, visibility)
        """
        obj_id_to_xyv = {obj_id: [] for obj_id in list_obj_ids}

        # Loop thru predicted points in this frame
        for i, obj_id in enumerate(query_point_obj_ids):
            xy = predicted_points[i]  # (2,)
            vis = points_visibility[i].unsqueeze(0)  # (1,)
            obj_id_to_xyv[obj_id].append(torch.cat([xy, vis.float()]))  # (3,)

        # Fill in objects with no points with empty tensors
        return {
            obj_id: torch.stack(pts_xyv) if pts_xyv else torch.empty(0, 3)
            for obj_id, pts_xyv in obj_id_to_xyv.items()
        }

    @torch.inference_mode()
    def init_state():
        pass

    @torch.inference_mode()
    def reset_state(self):
        """clear all queries and trajectory data"""
        pass

    @torch.inference_mode()
    def remove_object():
        pass
