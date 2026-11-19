# HUMAN-WRITTEN (core) — see CLAUDE.md. Do not AI-generate content in this file.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from splat.data.colmap import read_camera_data, read_image_data, read_point3D_data
from splat.data.geometry import quat_to_rotmat, camera_center

@dataclass
class View:
    name: str
    image: torch.Tensor # the image tensor, shape (Height, Width, 3), dtype torch.float32, values in [0, 1]
    width: int
    height: int
    params: tuple[float, float, float, float] # (fx, fy, cx, cy)
    R: torch.Tensor # rotation matrix, shape (3, 3) float32
    t: torch.Tensor # translation vector, shape (3,) float32
    C: torch.Tensor # camera center, shape (3,) float32


@dataclass
class Scene:
    views: list[View]
    test_views: list[View] # Views that are held out for testing, not used for training
    points3D: torch.Tensor # shape (N, 3) float32
    colors: torch.Tensor # shape (N, 3) float32; values are not yet divided by 255
    extent: float # the extent of the scene, computed as the maximum distance from the camera centers to the points3D


# folder_dir is type Path, not string
def load_scene(folder_dir: Path, downsample = 8, eval = True) -> Scene:
    """
    Load a scene from a COLMAP reconstruction.
    Args:
        folder_dir: Path to the directory containing the COLMAP reconstruction (cameras.bin, images.bin, points3D.bin).
        downsample: the ratio by which the image resolution should be downsampled. Default 8, the lowest resolution
        eval: boolean for whether the dataset should hold out images to use for evaluation
    Returns:
        A Scene object containing the views, points3D, colors, and extent.
    """
    # Downsample directory table
    SAMPLING_DIRS = {
        1: "images",
        2: "images_2",
        4: "images_4",
        8: "images_8",
    }

    if downsample not in SAMPLING_DIRS.keys():
        raise ValueError(f"{downsample} is not a valid downsampling value")

    image_dir = SAMPLING_DIRS.get(downsample)

    image_folder = folder_dir / image_dir

    # Load the COLMAP data
    cameras = read_camera_data(folder_dir / "sparse" / "0" / "cameras.bin")
    images = read_image_data(folder_dir / "sparse" / "0" / "images.bin")
    points3D = read_point3D_data(folder_dir / "sparse" / "0" / "points3D.bin")

    images = dict(sorted(images.items(), key=lambda item: item[1]["name"])) # sort images by name
    M = len(images) # number of images

    # Load the quaternions and translations for all images into tensors
    quats = torch.tensor([i["quaternion"] for i in images.values()], dtype=torch.float64) # shape (num_views, 4)
    translations = torch.tensor([i["translation"] for i in images.values()], dtype=torch.float64) # shape (num_views, 3)

    # Populate tensors for rotation matrices and camera centers
    R = quat_to_rotmat(quats) # shape (num_images, 3, 3)
    C = camera_center(R, translations) # shape (num_views, 3)

    # Extent finds the average position of the cameras; the centroid. It then finds distance of the camera farthest 
    # from that point and adds a 10% buffer. It acts as a baseline for the scale of the scene
    extent = 1.1 * ((C - C.mean(dim=0)).norm(dim=-1).max().item())

    # Create the views
    views = []

    # Hold out every 8th image for evaluation if eval is True, otherwise use all images for training
    if eval:
        test_views = []

    for i, entry in enumerate(images.values()):
        # Load the image
        image_path = image_folder / entry["name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB"))
        H, W = image.shape[:2]

        pixels = torch.from_numpy(image).float() / 255.0 # convert to float32 and normalize to [0, 1]

        # Get the camera parameters
        camera = cameras[entry["camera_id"]]

        # Normalize from simple_pinhole to (fx, fy, cx, cy)
        if camera["model_id"] == 0: # SIMPLE_PINHOLE
            fx = camera["params"][0] * (W/camera["width"])
            fy = camera["params"][0] * (H/camera["height"])
            cx = camera["params"][1] * (W/camera["width"])
            cy = camera["params"][2] * (H/camera["height"])
        elif camera["model_id"] == 1: # PINHOLE
            fx = camera["params"][0] * (W/camera["width"])
            fy = camera["params"][1] * (H/camera["height"])
            cx = camera["params"][2] * (W/camera["width"])
            cy = camera["params"][3] * (H/camera["height"])
        else:
            raise ValueError(f"Unsupported camera model: {camera['model_id']}")

        params = (fx, fy, cx, cy)

        # Get the rotation, translation, and camera center for this view
        rotation = R[i].float() # convert to float32
        translation = translations[i].float() # convert to float32
        center = C[i].float() # convert to float32

        view = View(
            name=entry["name"],
            image=pixels,
            width=W,
            height=H,
            params=params,
            R=rotation,
            t=translation,
            C=center
        )

        if eval and i % 8 == 0: # Hold out every 8th image for evaluation (if we are in eval mode)
            test_views.append(view)
        else:
            views.append(view)

    # convert position and color dicts to tensors
    points3D_tensor = torch.tensor([xyz["position"] for xyz in points3D.values()], dtype=torch.float32)
    rgb_tensor = torch.tensor([rgb["color"] for rgb in points3D.values()], dtype=torch.float32)

    return Scene(
        views=views,
        test_views=test_views if eval else [],
        points3D=points3D_tensor,
        colors=rgb_tensor,
        extent=extent
    )