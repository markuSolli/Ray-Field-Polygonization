from ray_field.baseline import Baseline
from ray_field.baseline_device import BaselineDevice
from ray_field.baseline_cpu import BaselineCPU
from ray_field.prescan_cone import PrescanCone
from ray_field.candidate_sphere import CandidateSphere
from ray_field.ball_pivoting import BallPivoting
from ray_field.convex_hull import ConvexHull
from ray_field.angle_filter import AngleFilter

ALGORITHM_LIST = ['baseline', 'baseline_device', 'baseline_cpu', 'prescan_cone', 'candidate_sphere', 'ball_pivoting', 'convex_hull', 'angle_filter']
OBJECT_NAMES = ['armadillo', 'bunny', 'buddha', 'dragon'] # Runs out of memory when measuring Chamfer distance with 'lucy'
N_VALUES = list(range(50, 501, 50))

model_checkpoint_dict = {
    'armadillo': 'armadillo',
    'bunny': 'bunny',
    'dragon': 'dragon',
    'buddha': 'happy_buddha'
}

class_dict = {
    'baseline': Baseline,
    'baseline_device': BaselineDevice,
    'baseline_cpu': BaselineCPU,
    'prescan_cone': PrescanCone,
    'candidate_sphere': CandidateSphere,
    'ball_pivoting': BallPivoting,
    'convex_hull': ConvexHull,
    'angle_filter': AngleFilter
}

model_name_dict = {
    'armadillo': 'Armadillo',
    'bunny': 'Bunny',
    'dragon': 'Dragon',
    'buddha': 'Happy Buddha'
}