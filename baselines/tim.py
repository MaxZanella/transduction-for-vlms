import torch
from torch import Tensor, nn

from baselines.few_shot_classifier import FewShotClassifier

from baselines.tim_utils import get_mi, get_cond_entropy, get_entropy, get_one_hot
class TIM(FewShotClassifier):
    """
    Malik Boudiaf, Ziko Imtiaz Masud, Jérôme Rony, José Dolz, Pablo Piantanida, Ismail Ben Ayed.
    "Transductive Information Maximization For Few-Shot Learning" (NeurIPS 2020)
    https://arxiv.org/abs/2008.11297

    Fine-tune prototypes based on
        1) classification error on support images
        2) mutual information between query features and their label predictions
    Classify w.r.t. to euclidean distance to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    TIM is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 50,
        fine_tuning_lr: float = 1e-4,
        cross_entropy_weight: float = 0.1,
        marginal_entropy_weight: float = 1.0,
        conditional_entropy_weight: float = 0.1,
        temperature: float = 10.0,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            cross_entropy_weight: weight given to the cross-entropy term of the loss
            marginal_entropy_weight: weight given to the marginal entropy term of the loss
            conditional_entropy_weight: weight given to the conditional entropy term of the loss
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature

    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error and mutual information between
        query features and their label predictions.
        Then classify w.r.t. to euclidean distance to prototypes.
        """
        #self.prototypes.type(torch.DoubleTensor).to(self.support_features.device)
        query_features = self.compute_features(query_images)#.type(torch.DoubleTensor).to(self.support_features.device)

        num_classes = self.support_labels.unique().size(0)
        support_labels_one_hot = nn.functional.one_hot(  # pylint: disable=not-callable
            self.support_labels, num_classes
        )

        with torch.enable_grad():
            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.fine_tuning_lr)

            for _ in range(self.fine_tuning_steps):
                support_logits = self.temperature * self.cosine_distance_to_prototypes(
                    self.support_features
                )#.type(torch.DoubleTensor)
                query_logits = self.temperature * self.cosine_distance_to_prototypes(
                    query_features
                )#.type(torch.DoubleTensor)

                support_cross_entropy = (
                    -(support_labels_one_hot * support_logits.log_softmax(1))
                    .sum(1)
                    .mean(0)
                )

                query_soft_probs = query_logits.softmax(1)
                query_conditional_entropy = (
                    -(query_soft_probs * torch.log(query_soft_probs + 1e-12))
                    .sum(1)
                    .mean(0)
                )

                marginal_prediction = query_soft_probs.mean(0)
                marginal_entropy = -(
                    marginal_prediction * torch.log(marginal_prediction)
                ).sum(0)

                loss = self.cross_entropy_weight * support_cross_entropy - (
                    self.marginal_entropy_weight * marginal_entropy
                    - self.conditional_entropy_weight * query_conditional_entropy
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features),
            temperature=self.temperature,
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True


class TIM_ADM(FewShotClassifier):
    """
    Malik Boudiaf, Ziko Imtiaz Masud, Jérôme Rony, José Dolz, Pablo Piantanida, Ismail Ben Ayed.
    "Transductive Information Maximization For Few-Shot Learning" (NeurIPS 2020)
    https://arxiv.org/abs/2008.11297

    Fine-tune prototypes based on
        1) classification error on support images
        2) mutual information between query features and their label predictions
    Classify w.r.t. to euclidean distance to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    TIM is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 150,
        fine_tuning_lr: float = 1e-4,
        cross_entropy_weight: float = 0.1,
        marginal_entropy_weight: float = 1.0,
        conditional_entropy_weight: float = 0.1,
        temperature: float = 15.0,
        alpha: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            cross_entropy_weight: weight given to the cross-entropy term of the loss
            marginal_entropy_weight: weight given to the marginal entropy term of the loss
            conditional_entropy_weight: weight given to the conditional entropy term of the loss
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature

        self.alpha = alpha

        self.loss_weights = [self.cross_entropy_weight, self.marginal_entropy_weight, self.conditional_entropy_weight]


    def q_update(self, P):
        """
        inputs:
            P : torch.tensor of shape [q_shot, num_class]
                where P[j,k] = probability of point j belonging to class k
                (according to our L2 classifier)
        """
        l1, l2 = self.loss_weights[1], self.loss_weights[2]
        l3 = 1.0  # Corresponds to the weight of the KL penalty
        alpha = l2 / l3
        beta = l1 / (l1 + l3)

        Q = (P ** (1+alpha)) / ((P ** (1+alpha)).sum(dim=0, keepdim=True)) ** beta
        self.Q = (Q / Q.sum(dim=-1, keepdim=True)).float()

    def weights_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, num_classes]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        P_s = self.get_logits(support).softmax(-1)
        P_q = self.get_logits(query).softmax(-1)
        src_part = self.loss_weights[0] / (1 + self.loss_weights[2]) * y_s_one_hot.transpose(0, 1).matmul(support)
        src_part += self.loss_weights[0] / (1 + self.loss_weights[2]) * (self.weights * P_s.sum(0, keepdim=True).transpose(0, 1)\
                                                                         - P_s.transpose(0, 1).matmul(support))
        src_norm = self.loss_weights[0] / (1 + self.loss_weights[2]) * y_s_one_hot.sum(0).view(-1, 1)

        qry_part = self.N_s / self.N_q * self.Q.transpose(0, 1).matmul(query)
        qry_part += self.N_s / self.N_q * (self.weights * P_q.sum(0, keepdim=True).transpose(0, 1)\
                                           - P_q.transpose(0, 1).matmul(query))
        qry_norm = self.N_s / self.N_q * self.Q.sum(0).view(-1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.alpha * (new_weights - self.weights)

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [shot, num_class]
        """

        logits = self.temperature * self.cosine_distance_to_prototypes(
            samples
        )  # .type(torch.DoubleTensor)

        """
        logits = self.temperature * (samples.matmul(self.weights.transpose(0, 1)) \
                              - 1 / 2 * (self.weights ** 2).sum(1).view(1, -1) \
                              - 1 / 2 * (samples ** 2).sum(1).view(-1, 1))  #
        """

        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(-1)
        return preds

    def init_weights(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.weights = self.prototypes


    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error and mutual information between
        query features and their label predictions.
        Then classify w.r.t. to euclidean distance to prototypes.
        """
        #self.prototypes.type(torch.DoubleTensor).to(self.support_features.device)
        query_features = self.compute_features(query_images)#.type(torch.DoubleTensor).to(self.support_features.device)

        num_classes = self.support_labels.unique().size(0)
        support_labels_one_hot = nn.functional.one_hot(self.support_labels, num_classes) .type(torch.FloatTensor).to(self.support_features.device)
        #support_labels_one_hot = get_one_hot(self.support_labels).type(torch.DoubleTensor).to(self.support_features.device)
        #print(support_labels_one_hot.shape)
        self.N_s = support_labels_one_hot.shape[0]
        self.N_q = query_features.shape[0]

        self.init_weights(self.support_features, support_labels_one_hot)
        for i in range(self.fine_tuning_steps):
            P_q = self.get_logits(query_features).softmax(-1)
            self.q_update(P=P_q)
            self.weights_update(self.support_features, query_features, support_labels_one_hot)

        self.prototypes = self.weights
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features),
            #self.get_logits(query_features),
            temperature=self.temperature,
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True


class TIM_ADM_bis(FewShotClassifier):
    """
    Malik Boudiaf, Ziko Imtiaz Masud, Jérôme Rony, José Dolz, Pablo Piantanida, Ismail Ben Ayed.
    "Transductive Information Maximization For Few-Shot Learning" (NeurIPS 2020)
    https://arxiv.org/abs/2008.11297

    Fine-tune prototypes based on
        1) classification error on support images
        2) mutual information between query features and their label predictions
    Classify w.r.t. to euclidean distance to updated prototypes.
    As is, it is incompatible with episodic training because we freeze the backbone to perform
    fine-tuning.

    TIM is a transductive method.
    """

    def __init__(
        self,
        *args,
        fine_tuning_steps: int = 150,
        fine_tuning_lr: float = 1e-4,
        cross_entropy_weight: float = 0.1,
        marginal_entropy_weight: float = 1.0,
        conditional_entropy_weight: float = 0.1,
        temperature: float = 15.0,
        alpha: float = 1.0,
        **kwargs,
    ):
        """
        Args:
            fine_tuning_steps: number of fine-tuning steps
            fine_tuning_lr: learning rate for fine-tuning
            cross_entropy_weight: weight given to the cross-entropy term of the loss
            marginal_entropy_weight: weight given to the marginal entropy term of the loss
            conditional_entropy_weight: weight given to the conditional entropy term of the loss
            temperature: temperature applied to the logits before computing
                softmax or cross-entropy. Higher temperature means softer predictions.
        """
        super().__init__(*args, **kwargs)

        # Since we fine-tune the prototypes we need to make them leaf variables
        # i.e. we need to freeze the backbone.
        self.backbone.requires_grad_(False)

        self.fine_tuning_steps = fine_tuning_steps
        self.fine_tuning_lr = fine_tuning_lr
        self.cross_entropy_weight = cross_entropy_weight
        self.marginal_entropy_weight = marginal_entropy_weight
        self.conditional_entropy_weight = conditional_entropy_weight
        self.temperature = temperature

        self.alpha = alpha

        self.loss_weights = [self.cross_entropy_weight, self.marginal_entropy_weight, self.conditional_entropy_weight]


    def q_update(self, P):
        """
        inputs:
            P : torch.tensor of shape [n_tasks, q_shot, num_class]
                where P[i,j,k] = probability of point j in task i belonging to class k
                (according to our L2 classifier)
        """
        l1, l2 = self.loss_weights[1], self.loss_weights[2]
        l3 = 1.0  # Corresponds to the weight of the KL penalty
        alpha = l2 / l3
        beta = l1 / (l1 + l3)

        Q = (P ** (1+alpha)) / ((P ** (1+alpha)).sum(dim=1, keepdim=True)) ** beta
        self.Q = (Q / Q.sum(dim=2, keepdim=True)).float()

    def weights_update(self, support, query, y_s_one_hot):
        """
        Corresponds to w_k updates
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, s_shot, num_classes]


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        P_s = self.get_logits(support).softmax(2)
        P_q = self.get_logits(query).softmax(2)
        src_part = self.loss_weights[0] / (1 + self.loss_weights[2]) * y_s_one_hot.transpose(1, 2).matmul(support)
        src_part += self.loss_weights[0] / (1 + self.loss_weights[2]) * (self.weights * P_s.sum(1, keepdim=True).transpose(1, 2)\
                                                                         - P_s.transpose(1, 2).matmul(support))
        src_norm = self.loss_weights[0] / (1 + self.loss_weights[2]) * y_s_one_hot.sum(1).view(n_tasks, -1, 1)

        qry_part = self.N_s / self.N_q * self.Q.transpose(1, 2).matmul(query)
        qry_part += self.N_s / self.N_q * (self.weights * P_q.sum(1, keepdim=True).transpose(1, 2)\
                                           - P_q.transpose(1, 2).matmul(query))
        qry_norm = self.N_s / self.N_q * self.Q.sum(1).view(n_tasks, -1, 1)

        new_weights = (src_part + qry_part) / (src_norm + qry_norm)
        self.weights = self.weights + self.alpha * (new_weights - self.weights)

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = self.temperature * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1))  #
        return logits

    def get_preds(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, s_shot, feature_dim]

        returns :
            preds : torch.Tensor of shape [n_task, shot]
        """
        logits = self.get_logits(samples)
        preds = logits.argmax(2)
        return preds


    def init_weights(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = y_s  # get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1).type(torch.FloatTensor).to(self.support_features.device)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts


    def forward(
        self,
        query_images: Tensor,
    ) -> Tensor:
        """
        Overrides forward method of FewShotClassifier.
        Fine-tune prototypes based on support classification error and mutual information between
        query features and their label predictions.
        Then classify w.r.t. to euclidean distance to prototypes.
        """
        #self.prototypes.type(torch.DoubleTensor).to(self.support_features.device)
        query_features = self.compute_features(query_images).unsqueeze(0)   #.type(torch.DoubleTensor).to(self.support_features.device)

        num_classes = self.support_labels.unique().size(0)
        #support_labels_one_hot = nn.functional.one_hot(self.support_labels, num_classes) .type(torch.DoubleTensor).to(self.support_features.device).unsqueeze(0)
        support_labels_one_hot = get_one_hot(self.support_labels.unsqueeze(0)).type(torch.FloatTensor).to(self.support_features.device)
        #print(support_labels_one_hot.shape)
        self.N_s = support_labels_one_hot.shape[1]
        self.N_q = query_features.shape[1]

        self.init_weights(self.support_features.unsqueeze(0).type(torch.FloatTensor).cuda(), support_labels_one_hot)
        for i in range(self.fine_tuning_steps):
            P_q = self.get_logits(query_features).softmax(2)
            self.q_update(P=P_q)
            self.weights_update(self.support_features.unsqueeze(0), query_features, support_labels_one_hot)

        self.prototypes = self.weights[0]
        return self.softmax_if_specified(
            self.cosine_distance_to_prototypes(query_features[0]),
            #self.get_logits(query_features),
            temperature=self.temperature,
        ).detach()

    @staticmethod
    def is_transductive() -> bool:
        return True