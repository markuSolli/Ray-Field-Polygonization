from ray_field.baseline import Baseline
from ray_field.baseline_device import BaselineDevice
from ray_field.prescan_cone import PrescanCone

ALGORITHM_LIST = ['baseline', 'baseline_device', 'prescan_cone', 'candidate_sphere']
OBJECT_NAMES = ['armadillo', 'bunny', 'happy_buddha', 'dragon'] # Runs out of memory when measuring Chamfer distance with 'lucy'
N_VALUES = list(range(50, 501, 50))

class_dict = {
    'baseline': Baseline,
    'baseline_device': BaselineDevice,
    'prescan_cone': PrescanCone
}