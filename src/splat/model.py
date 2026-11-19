# HUMAN-WRITTEN (core) — see CLAUDE.md. Do not AI-generate content in this file.

from dataclasses import dataclass

import torch
import math

# Zeroth-order real spherical harmonic basis constant: Y_0^0 = 1 / (2 * sqrt(pi))
# Defined in the official 3DGS reference code (utils/sh_utils.py), adapted from Plenoxels (svox2).
# Paper: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
C0 = 0.28209479177387814
# I'm not well-versed in the exact math behind spherical harmonics so I won't claim to know exactly how
# this constant is derived, but I know it plays a role in normalizing the coefficients we find later,
# which prevents us from aquiring a value outside of the 0-255 range that rgb requires.


# names of the fields in the Gaussian class
GAUSSIAN_FIELDS = ("xyz",
                   "log_scales",
                   "quats_raw",
                   "opacity_logits",
                   "features_DC",
                   "features_rest",
                   "SHdegree_active",
                   "SHdegree_max"
                   )


@dataclass(eq=False, slots=True, kw_only=True)
class Gaussians:
    """
    A dataclass holding all the Gaussians in one scene and relevant methods for interacting with them. Gaussians are stored in
    6 separate tensors, where each one represents one aspect of a Gaussian. 
    Fields:
        xyz: raw location
        log_scales: raw scaling values; determines the stretch of a Gaussian along its own axes before rotation.
            The raw values stored here are passed into torch.exp() (which does e^scalar) before being applied 
            to the rendering process. This is done because Gaussains take on several scales in one scene,
            so the optimizer would have to learn several scales, but it is incapable of doing so efficiently.
            However, if the scalars are placed in log space, the optimizer updates Gaussian scaling by fractions
            of their size, making scale irrelevant to learning rate. Also this keeps the scale positive.
        quats_raw: rotation quaternion
        opacity_logits: a scalar determining opacity of a single Gaussian. The scalar is stored as a logit,
            meaning it is passed through the sigmoid function before rendering to map the value between 0 and 1
        features_DC: the 0th degree of Spherical Harmonics (SH). Determines the primary color of a Gaussian and trains
            at a different rate to the other bands, hence the separate field
        features_rest: the remaining bands of SH. Determines subtler coloring as the Gaussian is viewed from different 
            directions, i.e. brightening a Gaussian if the angle is reflecting sunlight to the viewer
        
        SHdegree_active: current number of SH bands in use. SH coefficients represent the majority of parameters
        SHdegree_max: the maximum number of SH bands allowed for use.
    """

    xyz: torch.Tensor # x, y, z; shape: N, 3
    log_scales: torch.Tensor # e^l per local axis x, y, z is the stretch along that axis; shape: N, 3
    quats_raw: torch.Tensor # quaternion w, x, y, z; shape: N, 4
    opacity_logits: torch.Tensor # single scalar, shape: N, 1
    features_DC: torch.Tensor # 0th degree of SH; shape: N, 1, 3
    features_rest: torch.Tensor # 1st-3rd degree of SH; shape: N, (SHdegree_max^2 - 1), 3
    SHdegree_active: int = 0 # current degree of SH
    SHdegree_max: int = 3 # maximum degree of SH


    # during training, the optimizer can return unconstrained values for certain fields
    # we don't bound them in the tensors because we don't want to constrain the optimizer
    @property
    def scale(self):
        return torch.exp(self.log_scales)

    @property
    def rotation(self):
        return torch.nn.functional.normalize(self.quats_raw, dim=-1)
    
    @property
    def opacity(self):
        return torch.sigmoid(self.opacity_logits)

    # also add this to count gaussians easily
    @property
    def N(self):
        return self.xyz.shape[0]



    # the @dataclass decorator compiles _init_ from the fields, calling this at the end
    # check model is not malformed at initialization
    def __post_init__(self):
        self.validate()

    # check that the gaussians are not malformed
    def validate(self):

        tensor_shapes = {
                "xyz": (3,),
                "log_scales": (3,),
                "quats_raw": (4,),
                "opacity_logits": (1,),
                "features_DC": (1, 3),
                "features_rest": ((((self.SHdegree_max + 1) ** 2) - 1), 3),
                }

        # xyz is a tensor
        if not isinstance(self.xyz, torch.Tensor):
            raise TypeError("xyz is not a tensor")

        # xyz is populated with floats
        if not self.xyz.is_floating_point():
            raise TypeError(f"xyz is {self.xyz.dtype} not a floating point value")
        
        N = self.xyz.shape[0]

        # verify tensor integrity
        for name in tensor_shapes:
            t = getattr(self, name)

            # is a tensor
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{name} is not a tensor")

            if not t.is_leaf:
                raise TypeError(f"{name} is not a leaf tensor")
            
            # same N
            if not (t.shape[0] == N):
                raise ValueError(f"{name} shape is {t.shape[0]} but N = {N}")

            # all same trailing dimensions
            if not (t.shape[1:] == tensor_shapes[name]):
                raise ValueError(f"{name} shape is {t.shape[1:]} but should be {tensor_shapes[name]}")

            # data type is consistent
            if not (t.dtype == self.xyz.dtype):
                raise TypeError(f"{name} is {t.dtype} but xyz is {self.xyz.dtype}")

            # device is not consistent
            if not (t.device == self.xyz.device):
                raise TypeError(f"{name} is on {t.device} but xyz is set to {self.xyz.device}")

        # active and max SH degree are valid values
        if not (0 <= self.SHdegree_active <= self.SHdegree_max <= 3):
            raise ValueError(f"Active SH degree = {self.SHdegree_active} and Max SH degree = {self.SHdegree_max} when they shouldn't")

    # initialize a Gaussian cloud from SFM points
    @classmethod
    def from_sfm(cls, xyz: torch.Tensor, rgb: torch.Tensor, max_sh_degree=3, device="cpu", dtype=torch.float32):
        """
        Initializes a Gaussian cloud from SFM data.
        Args:
            cls: implicit class argument
            xyz: position tensor. Shape of Px3
            rgb: color tensor. Shape of Px3, values are between 0-255
            max_sh_degree: the maximum degree of SH bands. Defaults to 3
            device: which device pytorch will run on. Defaults to cpu
            dtype: the type of data stored in each tensor. Defaults to float32
        Returns:
            a new Gaussians object populated as per the sfm data
        """
        # ensure all new tensors here are leaf tensors
        with torch.no_grad():
            # make a new copy of the positions tensor
            # this is intended to take in the tensor from the Scene class, so we don't want to modify it in place
            means = xyz.to(device=device, dtype=dtype, copy=True)
            P = means.shape[0]

            # convert the colors tensor from Scene to a DC tensor
            rgb = rgb.to(device=device, dtype=dtype)
            DC = ((rgb/255 - 0.5) / C0).unsqueeze(1)
            # initialize subsequent bands as 0s
            rest = torch.zeros((P,(((max_sh_degree + 1) ** 2) - 1), 3), device=device, dtype=dtype)

            # find the mean squared distance of each point's 3 nearest neighbors
            dist = knn_mean_squared_dist(means)
            # initialize the scaling as radius = root mean squared distance
            log_scales = 0.5 * torch.log(dist)
            log_scales = log_scales.unsqueeze(-1).repeat(1, 3)

            # initialize rotation quaternions as identity (no rotation) 1, 0, 0, 0
            quats = torch.zeros([P, 4], device=device, dtype=dtype)
            quats[:, 0] = 1

            # initialize opacity logits. Opacity is 0.1, stored as its logit roughly equaling -2.197
            opacity_logits = torch.full((P, 1), math.log(0.1 / (1 - 0.1)), device=device, dtype=dtype)

        # enable autograd now that everything is initialized
        means.requires_grad_()
        DC.requires_grad_()
        rest.requires_grad_()
        log_scales.requires_grad_()
        quats.requires_grad_()
        opacity_logits.requires_grad_()

        return cls(
            xyz=means,
            log_scales=log_scales,
            quats_raw=quats,
            opacity_logits=opacity_logits,
            features_DC=DC,
            features_rest=rest,
            SHdegree_max=max_sh_degree
            )



    # initialize a Gaussian cloud randomly
    @classmethod
    def from_random(cls, n, bbox_min, bbox_max, max_sh_degree=3, seed=42, device="cpu", dtype=torch.float32):
        """
        Initializes a random Gaussian cloud of count n within some bounding box. Can be seeded.
        Args:
            cls: implicit class argument
            n: number of Gaussians
            bbox_min: minimum corner of bounding box. Shape (3,) a tensor or sequence
            bbox_max: maximum corner of bounding box. Shape (3,) a tensor or sequence
            max_sh_degree: maximum degree of SH bands. Defaults to 3
            seed: seed for random generators. Defaults to 42
            device: which device pytorch will run on. Defaults to cpu
            dtype: the type of data stored in each tensor. Defaults to float32
        Returns:
            a new Gaussians object populated with random data
        """
        # generate random number on the cpu, avoiding mismatches across devices
        g = torch.Generator("cpu")
        g.manual_seed(seed)

        # generate random points with coordinates between 0, 1
        u = torch.rand((n, 3), generator=g, dtype=dtype)
        # ensure bounding box dimensions are tensors
        box_min = torch.as_tensor(bbox_min, dtype=dtype)
        box_max = torch.as_tensor(bbox_max, dtype=dtype)

        # map random points into the bounding box
        xyz = box_min + u * (box_max - box_min)

        # initialize random colors
        colors = torch.randint(0, 256, (n, 3), generator=g, device=device, dtype=dtype)

        return cls.from_sfm(
            xyz=xyz,
            rgb=colors,
            max_sh_degree=max_sh_degree,
            device=device,
            dtype=dtype
        )



# chunked k nearest neighbors function
def knn_mean_squared_dist(points, k=3, B=512):
    """
    Find the mean squared distance to the k nearest neighbors.
    Args:
        points: the points for which we want to know the k nearest neighbors
        k: how many neighbors to find
        B: the chunk size. Running kNN without chunking the points eats memory
    Returns:
        length-P tensor: mean squared distance to the k nearest neighbors. Minimum value
        is clamped at 1e-7. This is done to prevent the log-scaling process from producing
        a value of -infinity.
    """
    P = points.shape[0]
    if P < k+1:
        raise ValueError(f"there are {P} points but topk needs {k + 1} points to function")
    kmean = []
    with torch.no_grad():
        for i in range(0, P, B):
            chunk = points[i: i + B]
            distance = torch.cdist(chunk, points)
            knearest = torch.topk(distance, k + 1, dim=-1, largest=False).values
            knearest = knearest[:, 1:] ** 2
            mean_dist = knearest.mean(dim=-1)
            kmean.append(mean_dist)

        return torch.cat(kmean).clamp_min(1e-7)