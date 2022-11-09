"""Registry of model and action dist names."""

from models.communication import (
    AttComModel,
    InvariantAttComModel,
    AttComActionMaskModel,
    InvariantAttComActionMaskModel,
    TorchHomogeneousMultiActionDistribution,
)
from models.torch.fcnet import ContextualFullyConnectedNetwork
from models.torch.action_mask_model import TorchActionMaskModel
from models.torch.invariant_action_mask_model import TorchInvariantActionMaskModel

MODELS = {
    "contextual_fcnet": ContextualFullyConnectedNetwork,
    "action_mask_model": TorchActionMaskModel,
    "invariant_action_mask_model": TorchInvariantActionMaskModel,
    "att_com_model": AttComModel,
    "invariant_att_com_model": InvariantAttComModel,
    "att_com_action_mask_model": AttComActionMaskModel,
    "invariant_att_com_action_mask_model": InvariantAttComActionMaskModel,
}


def get_model_class(model: str) -> type:
    """Returns the class of a known model given its name."""

    if model in MODELS:
        class_ = MODELS[model]
    else:
        raise Exception(f"Unknown model {model}.")

    return class_


ACTION_DISTS = {
    "hom_multi_action": TorchHomogeneousMultiActionDistribution,
}


def get_action_dist_class(action_dist: str) -> type:
    """Returns the class of a known model given its name."""

    if action_dist in ACTION_DISTS:
        class_ = ACTION_DISTS[action_dist]
    else:
        raise Exception(f"Unknown action distribution {action_dist}.")

    return class_
