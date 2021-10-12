from nncore.core.registry import Registry
LEARNER_REGISTRY = Registry('LEARNER')

from .baselearner import BaseLearner
from .supervisedlearner import SupervisedLearner
