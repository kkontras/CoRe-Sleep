import torch
import torch.nn as nn

class Consecutives_Predictor(nn.Module):
    def __init__(self, agent):
        super(Consecutives_Predictor,self).__init__()
        self.agent = agent

        if "training_type" not in self.agent.config.model or self.agent.config.model.training_type == "normal":
            self.predictor_func = "get_predictions_time_series"
        elif self.agent.config.model.training_type == "alignment":
            self.predictor_func = "get_predictions_time_series_alignment"
        elif self.agent.config.model.training_type == "alignment_order":
            self.predictor_func = "get_predictions_time_series_alignment_order"
        else:
            raise ValueError("Training type does not exist, check self.agent.config.model.training_type! Available ones are 'normal', 'alignment' and 'alignment_order' ")
        self.this_predictor_func = getattr(self, self.predictor_func)

    def forward(self, data, inits):
        pred = self.this_predictor_func(data, inits)
        return pred

    def get_predictions_time_series(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0 = self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1 = self.model(
                                [view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred = self.model(views)

        return pred
    def get_predictions_time_series_alignment(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            matches = torch.zeros(batch, outer, outer).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0, matches_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0, matches_0= self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1, matches_1 = self.model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1, matches_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        print(matches_0.shape)
                        print(matches_1.shape)
                        matches[idx * outer:(idx + 1) * outer] = torch.cat([matches_0, matches_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer], matches[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)], matches[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred, matches = self.model(views)

        return pred
    def get_predictions_time_series_alignment_order(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            matches = torch.zeros(batch, outer, outer).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0, matches_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0, matches_0= self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1, matches_1 = self.model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1, matches_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        print(matches_0.shape)
                        print(matches_1.shape)
                        matches[idx * outer:(idx + 1) * outer] = torch.cat([matches_0, matches_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer], matches[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)], matches[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred, matches = self.model(views)

        return pred
    def get_predictions_time_series_teacher(self, views, inits):
        """
        This is a function to exploit the fact that time series are not always continuous. We dont want to correlate signals from different patients/recordings just because the batch is not fully dividing the number of recording imgs.
        :param views: List of tensors, data views/modalities
        :param inits: Tensor indicating with value one, when there incontinuities.
        :return: predictions of the self.model on the batch
        """
        inits_sum = (inits.sum(dim=1) > 1).nonzero(as_tuple=True)[0]
        if len(inits_sum) > 0:
            batch = views[0].shape[0]
            outer = views[0].shape[1]
            batch_idx_checked = torch.ones(batch, dtype=torch.bool)
            pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            t_pred = torch.zeros(batch * outer, self.config.num_classes).cuda()
            for idx in inits_sum:
                if inits[idx].sum() > 1:
                    ones_idx = (inits[idx] > 0).nonzero(as_tuple=True)[0]
                    if (ones_idx[0] + 1 == ones_idx[1]  ): #and ones_idx[0]!=0 and ones_idx[1]!= len(inits[idx])
                        if ones_idx[0] == 0:
                            pred_split_0, t_pred_split_0 = self.model([view[idx, ones_idx[0]].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_0, t_pred_split_0 = self.model([view[idx, :ones_idx[0] + 1].unsqueeze(dim=0) for view in views])
                        if ones_idx[1] == len(inits[idx]):
                            pred_split_1, t_pred_split_1 = self.model([view[idx, -1].unsqueeze(dim=0).unsqueeze(dim=1) for view in views])
                        else:
                            pred_split_1, t_pred_split_1 = self.model([view[idx, ones_idx[1]:].unsqueeze(dim=0) for view in views])

                        pred[idx * outer:(idx + 1) * outer] = torch.cat([pred_split_0, pred_split_1], dim=0)
                        t_pred[idx * outer:(idx + 1) * outer] = torch.cat([t_pred_split_0, t_pred_split_1], dim=0)
                        batch_idx_checked[idx] = False
                    else:
                        pred[idx * outer:(idx + 1) * outer],t_pred[idx * outer:(idx + 1) * outer] = self.model([view[idx].unsqueeze(dim=0) for view in views])

            pred[batch_idx_checked.repeat_interleave(outer)], t_pred[batch_idx_checked.repeat_interleave(outer)] = self.model([view[batch_idx_checked] for view in views])

        else:
            pred, t_pred = self.agent.model(views)

        return pred, t_pred