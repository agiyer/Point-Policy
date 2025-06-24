import numpy as np
import sys
import pickle
from PIL import Image
import torch
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from point_utils.correspondence import Correspondence
from point_utils.depth import Depth

from pathlib import Path
import json
from pydantic import BaseModel, ValidationError
from typing import List, Dict

# ---------- To validate the json annotationfile ----------
class Point(BaseModel):
    x: int
    y: int

class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int

class ObjectAnnotation(BaseModel):
    points: List[Point]
    bounding_box: BoundingBox

class PixelKeyData(BaseModel):
    image_path: str
    objects: Dict[str, ObjectAnnotation]

class AnnotationData(BaseModel):
    task_name: str
    pixel_keys: Dict[str, PixelKeyData]
# ------------------------------------------------------------

# ---------- To validate the vlm request ----------
class TaskRequest(BaseModel):
    task_name: str
    pixel_key: str
    objects: Dict[str, BoundingBox]
# ------------------------------------------------------------

class PointsClass:
    def __init__(
        self,
        root_dir,
        dift_path,
        cotracker_checkpoint,
        task_name,
        pixel_keys,
        device,
        width,
        height,
        image_size_multiplier,
        ensemble_size,
        dift_layer,
        dift_steps,
        num_points, # NOTE: never used but kept for backward compatibility
        vlm_request_path = None,
        use_gt_depth=False, # changed default to False
        **kwargs,
    ):
        """
        Initialize the Points Class for finding key points in the episode.

        Parameters:
        -----------
        root_dir : str
            The root directory for the github repository.

        task_name : str
            The name of the task done by the robot.

        device : str
            The device to use for computation, either 'cpu' or 'cuda' (for GPU acceleration).

        width : int
            The width that should be used in the correspondence model.

        height : int
            The height that should be used in the correspondence model.

        image_size_multiplier : int
            The size multiplier for the image in the correspondence model.

        ensemble_size : int
            The size of the ensemble for the DIFT model.

        dift_layer : int
            The specific layer of the DIFT model to use for feature extraction.

        dift_steps : int
            The number of steps or iterations for feature extraction in the DIFT model.
        """
        self.vlm_request = None
        if vlm_request_path is not None:
            with open(vlm_request_path, 'r') as f:
                request_json = json.load(f)
                self.vlm_request = TaskRequest(**request_json)

        self.pixel_keys = pixel_keys
        self.device = device

        self.tracks = {pixel_key: None for pixel_key in self.pixel_keys}
        
        # if "human_hand" in self.object_labels:
        #     # Do hand tracking with MediaPipe
        #     import mediapipe as mp

        #     # Initialize MediaPipe Hands
        #     mp_hands = mp.solutions.hands
        #     self.hands = mp_hands.Hands(
        #         static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
        #     )
        #     self.hand_tracks = {pixel_key: None for pixel_key in self.pixel_keys}

        #     # remove "human_hand" from object_labels
        #     self.object_labels.remove("human_hand")
        #     self.detect_hand = True
        #     self.num_hand_points = 9  # wrist + index finger + thumb
        # else:
        #     self.detect_hand = False

        
        # Set up the correspondence model and find the expert image features
        self.correspondence_model = Correspondence(
            device,
            dift_path,
            width,
            height,
            image_size_multiplier,
            ensemble_size,
            dift_layer,
            dift_steps,
        )

        json_path = Path(root_dir) / 'coordinates' / task_name / 'annotations.json'
        
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        try:
            self.annotations = AnnotationData.model_validate(raw_data)
        except ValidationError as e:
            print(f"Validation error: {e}")
            raise

        self.all_initial_coords = {}  # initial coords for entire img per pixel_key
        self.all_dift_features = {} # dift features for entire img per pixel_key
        
        for pixel_key in self.pixel_keys:

            expert_image = self.annotations.pixel_keys[pixel_key].image_path
            expert_image = Image.open(expert_image).convert("RGB")

            # Get DIFT features for the entire image
            self.all_dift_features[pixel_key] = self.correspondence_model.set_expert_correspondence(expert_image)
            
            all_points = []
            for obj_annotation in self.annotations.pixel_keys[pixel_key].objects.values():
                for point in obj_annotation.points:
                    all_points.append([0, point.x, point.y])   # NOTE: 0 is for backward compatibility with old code
            
            self.all_initial_coords[pixel_key] = np.array(all_points)
            self.num_points = len(all_points)

        # get dift features and points per object
        if self.vlm_request is not None:

            self.object_initial_coords = {} # initial coords per object per pixel_key
            self.object_dift_features = {} # dift features per object per pixel_key

            for pixel_key in self.pixel_keys:
                # Initialize nested dictionaries for each pixel_key
                self.object_initial_coords[pixel_key] = {}
                self.object_dift_features[pixel_key] = {}

                expert_image = self.annotations.pixel_keys[pixel_key].image_path
                expert_image = Image.open(expert_image).convert("RGB")
                
                for object_name, obj_annotation in self.annotations.pixel_keys[pixel_key].objects.items():

                    # Get DIFT features for each object by cropping based on bbox
                    bbox = obj_annotation.bounding_box
                    # Crop the image based on bounding box
                    cropped_img = expert_image.crop((bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))
                    
                    # Pass cropped image through correspondence model
                    self.object_dift_features[pixel_key][object_name] = self.correspondence_model.set_expert_correspondence(cropped_img)
                    
                    # Store object-specific initial coordinates (relative to cropped image)
                    object_points = []
                    for point in obj_annotation.points:

                        # Convert global coordinates to local coordinates relative to bbox
                        local_x = point.x - bbox.xmin
                        local_y = point.y - bbox.ymin
                        object_points.append([0, local_x, local_y])  # NOTE: 0 is for backward compatibility with old code
                    
                    self.object_initial_coords[pixel_key][object_name] = np.array(object_points)

        # Set up the depth model
        if use_gt_depth:
            self.depth_model = Depth("/path/to/Depth-Anything-V2/", device)

        # Set up cotracker
        sys.path.append(root_dir + "/co-tracker/")
        from cotracker.predictor import CoTrackerOnlinePredictor

        # self.cotracker = CoTrackerOnlinePredictor(checkpoint=root_dir + "/co-tracker/checkpoints/scaled_online.pth", window_len=16).to(device)
        self.cotracker = {}
        for pixel_key in self.pixel_keys:
            self.cotracker[pixel_key] = CoTrackerOnlinePredictor(
                checkpoint=cotracker_checkpoint,
                window_len=16,
            ).to(device)

        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.image_list = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.depth = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }

        # TODO: find out what this is for (in case we have to replicate per object)
        self.all_semantic_similar_points = {
            f"{pixel_key}": None
            for pixel_key in self.pixel_keys
        }

        self.device = device

        # in case image is cropped and resized
        self.original_image_size = None
        self.current_image_size = None
        self.crop_ratios = None

    # Image passed in here must be in RGB format
    def add_to_image_list(self, image, pixel_key):
        """
        Add an image to the image list for finding key points.

        Parameters:
        -----------
        image : np.ndarray
            The image to add to the image list. This image must be in RGB format.
        """

        key = f"{pixel_key}"

        transformed = (
            torch.from_numpy(image.astype(np.uint8)).permute(2, 0, 1).float() / 255
        )

        # We only want to track the last 16 images so pop the first one off if we have more than 16
        if self.image_list[key].shape[0] > 0 and self.image_list[key].shape[1] == 16:
            self.image_list[key] = self.image_list[key][:, 1:]

        # If it is the first image you want to repeat until the whole array is full
        # Otherwise it will just add the new image to the end of the array
        while self.image_list[key].shape[0] == 0 or self.image_list[key].shape[1] < 16:
            self.image_list[key] = torch.cat(
                (
                    self.image_list[key],
                    transformed.unsqueeze(0).unsqueeze(0).clone().to(self.device),
                ),
                dim=1,
            )

    def reset_episode(self):
        """
        Reset the image list for finding key points.
        """

        self.image_list = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.depth = {
            f"{pixel_key}": torch.tensor([]).to(self.device)
            for pixel_key in self.pixel_keys
        }
        self.tracks = {pixel_key: None for pixel_key in self.pixel_keys}
        self.hand_tracks = {pixel_key: None for pixel_key in self.pixel_keys}

    def find_semantic_similar_points(self, pixel_key):
        """
        Find the semantic similar points between the expert image and the current image.
        """
        # if vlm request path is not provided, use the entire image to find sem similar points
        if self.vlm_request is None:

            self.all_semantic_similar_points[pixel_key] = self.correspondence_model.find_correspondence(
                self.all_dift_features[pixel_key],
                self.image_list[pixel_key][0, -1],
                self.all_initial_coords[pixel_key],
            )

        # otherwise, find sem similar points for each object and then map back to the entire image
        else : 

            pixel_key = self.vlm_request.pixel_key

            if pixel_key not in self.pixel_keys:
                raise ValueError(f"Pixel key {self.vlm_request.pixel_key} not found in {self.pixel_keys}")

            # instead of finding sem similar points for the entire image, 
            # we need to find sem similar points for each object and then map back to the entire image
            # this should then be stored in self.all_semantic_similar_points[pixel_key]
            
            sem_similar_points = []

            for object_name, bbox in self.vlm_request.objects.items():                      

                if object_name in self.object_initial_coords[pixel_key]:

                    # TODO: replace image_list[0,-1] with the image the vlm saw
                    cropped_img = self.image_list[pixel_key][0, -1].crop((bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax))

                    obj_similar_points = self.correspondence_model.find_correspondence(
                        self.object_dift_features[pixel_key][object_name],
                        cropped_img,
                        self.object_initial_coords[pixel_key][object_name],
                    )

                    # Map coordinates back to the full image by adding bbox offset
                    obj_similar_points[:, 1] += bbox.xmin
                    obj_similar_points[:, 2] += bbox.ymin

                    sem_similar_points.append(obj_similar_points)
    
                else:
                    print(f'WARNING: object {object_name} not found in {self.annotations.pixel_keys[pixel_key].objects.keys()}. Skipping...')

            # Combine all object coordinates into a single numpy array
            if sem_similar_points:
                self.all_semantic_similar_points[pixel_key] = np.concatenate(sem_similar_points, axis=0)
            else:
                # If no objects found, create empty array with correct shape
                self.all_semantic_similar_points[pixel_key] = np.array([]).reshape(0, 3)


    def get_depth(self, pixel_key, last_n_frames=1):
        """
        Get the depth map for the current image using Depth Anything. Depth is height x width.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode
        """
        key = f"{pixel_key}"

        self.depth[key] = np.zeros(
            (
                last_n_frames,
                self.image_list[key].shape[3],
                self.image_list[key].shape[4],
            )
        )
        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            numpy_image = (
                self.image_list[key][0, frame_idx].cpu().numpy().transpose(1, 2, 0)
                * 255
            )
            depth = self.depth_model.get_depth(numpy_image)
            self.depth[key][frame_idx] = depth

    def set_depth(
        self,
        depth,
        pixel_key,
        original_image_size=None,
        current_image_size=None,
        crop_ratios=None,
    ):
        """
        If you are using ground truth depth, you can set the depth here.

        Parameters:
        -----------
        depth : np.ndarray
            The depth map for the current image. Depth is height x width.
        original_image_size : tuple
            The original size of the image before it was cropped and resize.
        current_image_size : tuple
            The current size of the image.
        crop_ratios : tuple -> ((float, float), (float, float)) - (crop_h, crop_w)
            The crop ratios used to crop the image.
        """
        key = f"{pixel_key}"
        self.original_image_size = original_image_size
        self.current_image_size = current_image_size
        self.crop_ratios = crop_ratios

        if self.depth[key].shape[0] == 8:
            self.depth[key] = self.depth[key][1:]

        while self.depth[key].shape[0] < 8:
            if self.depth[key].shape[0] == 0:
                self.depth[key] = depth[None, ...].copy()
            self.depth[key] = np.concatenate(
                (self.depth[key], depth[None, ...].copy()), axis=0
            )

    def track_points(
        self, pixel_key, last_n_frames=1, is_first_step=False, one_frame=True
    ):
        """
        Track the key points in the current image using the CoTracker model.

        Parameters:
        -----------
        is_first_step : bool
            Whether or not this is the first step in the episode.
        """

        if self.detect_hand:
            hand_tracks = self.track_points_hand(pixel_key)
            hand_tracks = torch.tensor(hand_tracks)

            if not is_first_step:
                if self.hand_tracks[pixel_key] is None:
                    self.hand_tracks[pixel_key] = hand_tracks[None]
                else:
                    hand_tracks = hand_tracks[-last_n_frames:]
                    self.hand_tracks[pixel_key] = torch.cat(
                        [
                            self.hand_tracks[pixel_key],
                            hand_tracks[None].to(self.hand_tracks[pixel_key].device),
                        ],
                        dim=1,
                    )

        if is_first_step:
            semantic_similar_points = []
            semantic_similar_points.append(
                self.all_semantic_similar_points[pixel_key]
            )
            semantic_similar_points = torch.cat(semantic_similar_points, dim=0)

            self.cotracker[pixel_key](
                video_chunk=self.image_list[pixel_key][0, 0]
                .unsqueeze(0)
                .unsqueeze(0),
                is_first_step=True,
                add_support_grid=True,
                queries=semantic_similar_points[None].to(self.device),
            )
            self.tracks[pixel_key] = semantic_similar_points
        else:
            tracks, _ = self.cotracker[pixel_key](
                self.image_list[pixel_key], one_frame=one_frame
            )
            # Remove the support points
            tracks = tracks[:, :, 0 : self.num_points, :]

            if self.detect_hand:
                self.hand_tracks[pixel_key] = self.hand_tracks[pixel_key].to(
                    tracks.device
                )
                self.tracks[pixel_key] = torch.cat(
                    [self.hand_tracks[pixel_key], tracks], dim=-2
                )
            else:
                self.tracks[pixel_key] = tracks.clone()


    def track_points_hand(self, pixel_key):
        frames = (
            self.image_list[pixel_key][0].cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        )
        frames = frames.astype(np.uint8)

        hand_tracks = []
        for frame in frames:
            results = self.hands.process(frame)
            if results.multi_hand_landmarks is not None:
                hand_track = []
                for hand_landmarks in results.multi_hand_landmarks:
                    # Wrist landmarks: 0
                    # Index finger landmarks: 5, 6, 7, 8
                    # Thumb landmarks: 1, 2, 3, 4
                    wrist_landmark = hand_landmarks.landmark[0]
                    index_finger_landmarks = [
                        hand_landmarks.landmark[i] for i in [5, 6, 7, 8]
                    ]
                    thumb_landmarks = [hand_landmarks.landmark[i] for i in [1, 2, 3, 4]]

                    # Draw wrist
                    x = int(wrist_landmark.x * frame.shape[1])
                    y = int(wrist_landmark.y * frame.shape[0])
                    hand_track.append([x, y])

                    # Draw index finger
                    for landmark in index_finger_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        hand_track.append([x, y])

                    # Draw thumb
                    for landmark in thumb_landmarks:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        hand_track.append([x, y])

                hand_track = np.array(hand_track)
                hand_tracks.append(hand_track)
            else:
                hand_tracks.append(
                    hand_tracks[-1]
                    if len(hand_tracks) > 0
                    else self.hand_tracks[pixel_key][0, -1].cpu()
                )
        return np.array(hand_tracks)

    def get_points(self, pixel_key, last_n_frames=1):
        """
        Get the list of points for the current frame.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        final_points : torch.Tensor
            The list of points for the current frame.
        """

        final_points = torch.zeros((last_n_frames, self.num_points, 3))

        for frame_num in range(last_n_frames):
            for point in range(self.num_points):
                frame_idx = -1 * (last_n_frames - frame_num)
                # try:
                if self.original_image_size is None:
                    depth = self.depth[pixel_key][
                        frame_idx,
                        int(self.tracks[pixel_key][0, frame_idx, point][1]),
                        int(self.tracks[pixel_key][0, frame_idx, point][0]),
                    ]
                else:
                    crop_h, crop_w = self.crop_ratios
                    w_orig, h_orig = self.original_image_size
                    w_curr, h_curr = self.current_image_size

                    # compute point_h in original image
                    point_h = self.tracks[pixel_key][0, frame_idx, point][1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = int(
                        (point_h / h_curr) * h_orig_cropped + h_orig * crop_h[0]
                    )

                    # compute point_w in original image
                    point_w = int(self.tracks[pixel_key][0, frame_idx, point][0])
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = int(
                        (point_w / w_curr) * w_orig_cropped + w_orig * crop_w[0]
                    )

                    depth = self.depth[pixel_key][frame_idx, point_h_orig, point_w_orig]

                x = self.tracks[pixel_key][0, frame_idx, point][0]
                y = self.tracks[pixel_key][0, frame_idx, point][1]

                final_points[frame_num, point] = torch.tensor([x, y, depth])

        return final_points

    def get_points_on_image(self, pixel_key, last_n_frames=1):
        """
        Get the list of points for the current frame in pixel space.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        final_points : torch.Tensor
            The list of points for the current frame.
        """

        final_points = torch.zeros((last_n_frames, self.num_points, 2))

        for frame_num in range(last_n_frames):
            for point in range(self.num_points):
                frame_idx = -1 * (last_n_frames - frame_num)

                x = self.tracks[pixel_key][0, frame_idx, point][0]
                y = self.tracks[pixel_key][0, frame_idx, point][1]
                final_points[frame_num, point] = torch.tensor([x, y])

        return final_points

    def plot_image(self, pixel_key, last_n_frames=1):
        """
        Plot the image with the key points overlaid on top of it. Running this will slow down your tracking, but it's good for debugging.

        Parameters:
        -----------
        last_n_frames : int
            The number of frames to look back in the episode.

        Returns:
        --------
        img_list : list
            A list of images with the key points overlaid on top of them.
        """

        img_list = []

        for frame_num in range(last_n_frames):
            frame_idx = -1 * (last_n_frames - frame_num)
            curr_image = (
                self.image_list[pixel_key][0, frame_idx]
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                * 255
            )

            fig, ax = plt.subplots(1)
            ax.imshow(curr_image.astype(np.uint8))

            rainbow = plt.get_cmap("rainbow")
            # Generate n evenly spaced colors from the colormap
            colors = [
                rainbow(i / self.tracks[pixel_key].shape[2])
                for i in range(self.tracks[pixel_key].shape[2])
            ]

            for idx, coord in enumerate(self.tracks[pixel_key][0, frame_idx]):
                ax.add_patch(
                    patches.Circle(
                        (coord[0].cpu(), coord[1].cpu()),
                        5,
                        facecolor=colors[idx],
                        edgecolor="black",
                    )
                )
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_list.append(img.copy())
            plt.close()

        return img_list
