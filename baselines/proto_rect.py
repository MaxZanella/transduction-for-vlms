"""
Source : https://github.com/sicara/easy-few-shot-learning/blob/master/easyfsl/methods/bd_cspn.py
"""
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
from baselines.few_shot_classifier import FewShotClassifier

class BDCSPN(FewShotClassifier):
    """
    Jinlu Liu, Liang Song, Yongqiang Qin
    "Prototype Rectification for Few-Shot Learning" (ECCV 2020)
    https://arxiv.org/abs/1911.10713

    Rectify prototypes with label propagation and feature shifting.
    Classify queries based on their cosine distance to prototypes.
    This is a transductive method.
    """

    def rectify_prototypes(
        self, query_features: Tensor,
            Z=8,
            epsilon=10
    ):  # pylint: disable=not-callable
        """
        Updates prototypes with label propagation and feature shifting.
        Args:
            query_features: query features of shape (n_query, feature_dimension)
            Z: number of pseudo labels to extend the support set
            epsilon: temperature for prototype reweighting
        """
        n_classes = self.support_labels.unique().size(0)
        one_hot_support_labels = nn.functional.one_hot(self.support_labels, n_classes)

        average_support_query_shift = self.support_features.mean(
            0, keepdim=True
        ) - query_features.mean(0, keepdim=True)
        query_features = query_features + average_support_query_shift
        query_logits = self.cosine_distance_to_prototypes(query_features)


        pl_set_indexes = []
        pl_set_features = []
        pl_set_onehot = []
        for cls in range(n_classes):
            (_, pl_indexes) = torch.topk(query_logits[:, cls], k=Z, dim=0)
            pl_set_indexes.append(pl_indexes)
            pl_features = query_features[pl_indexes]
            pl_set_features.append(pl_features)
            pl_onehot = torch.zeros(Z, n_classes)
            pl_onehot[:, cls] = 1
            pl_set_onehot.append(pl_onehot)

        pl_set_features = torch.cat(pl_set_features, dim=0).to(self.support_features.device)
        pl_set_onehot = torch.cat(pl_set_onehot, dim=0).to(one_hot_support_labels.device)

        s_prime_features = torch.cat((self.support_features, pl_set_features), dim=0)
        s_prime_onehot = torch.cat((one_hot_support_labels, pl_set_onehot), dim=0)

        s_prime_logits = (self.cosine_distance_to_prototypes(s_prime_features).type(torch.FloatTensor) * epsilon).exp().to(self.support_features.device)



        normalization_vector = (
            (s_prime_onehot * s_prime_logits).sum(0)
        ).unsqueeze(
            0
        )  # [1, n_classes]
        s_prime_reweighting = (
            s_prime_onehot * s_prime_logits
        ) / normalization_vector  # [n_support_prime, n_classes]

        self.prototypes = (s_prime_reweighting * s_prime_onehot).type(s_prime_features.type()).t().matmul(s_prime_features)

    def forward(
        self,
        query_images: Tensor,
        Z=8,
        epsilon=10
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Update prototypes using query images, then classify query images based
        on their cosine distance to updated prototypes.
        """
        query_features = self.compute_features(query_images)

        self.rectify_prototypes(
            query_features=query_features,
            Z=Z,
            epsilon=epsilon
        )
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features)
        )

    @staticmethod
    def is_transductive() -> bool:
        return True