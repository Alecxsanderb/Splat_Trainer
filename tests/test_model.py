import sys
from pathlib import Path

target_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(target_dir))
print(str(target_dir))

from splat.model import Gaussians

# super bare bones random initialization check
random_cloud1 = Gaussians.from_random(15, (-2, -4, -6), (2, 4, 6))
random_cloud2 = Gaussians.from_random(15, (-2, -4, -6), (2, 4, 6))

print(random_cloud1.xyz == random_cloud2.xyz)
print(random_cloud1.features_DC == random_cloud2.features_DC)