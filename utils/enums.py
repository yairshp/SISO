from enum import Enum


class CriterionType(Enum):
    classifier_features = "classifier_features"
    features_multi_model = "features_multi_model"
    intermediate_features = "intermediate_features"
    ir_features = "ir_features"
    ir_dino_ensemble = "ir_dino_ensemble"
    alpha_clip = "alpha_clip"
