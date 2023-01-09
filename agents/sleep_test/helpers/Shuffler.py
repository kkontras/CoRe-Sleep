import torch
import torch.nn as nn
import einops

class Shuffler(nn.Module):

    def __init__(self, mode=None):
        super(Shuffler, self).__init__()

        if mode == "random_shuffle_data":
            print("We are shuffling the outer data")
            self.func = "_shuffle_data_outer"
        elif mode == "random_shuffle_data_batch":
            print("We are shuffling the outer and batch data")
            self.func = "_shuffle_data_batch_outer"
        else:
            print("We are not shuffling the data")
            self.func ="_no_shuffle"

        self.this_shuffle_func = getattr(self, self.func)


    def forward(self, served_data):
        return self.this_shuffle_func(served_data)

    def _no_shuffle(self, served_data):
        perms = None
        return served_data, perms


    def _shuffle_data_outer(self, served_dict):
        perms = torch.randperm(served_dict["data"][list(served_dict["data"].keys())[0]].shape[1])
        served_dict["data"] = {view: served_dict["data"][view][:, perms] for view in served_dict["data"]}
        served_dict["label"] = served_dict["label"][list(served_dict["label"].keys())[0]][:, perms].flatten()
        served_dict["inits"] = {view: served_dict["inits"][view][:, perms] for view in served_dict["inits"]}
        return served_dict, perms


    def _shuffle_data_batch_outer(self, served_dict):
        d_shape = served_dict["data"][list(served_dict["data"].keys())[0]][0].shape
        perms = einops.rearrange(torch.randperm(d_shape[0] * d_shape[1]), " (batch seq) -> batch seq ")
        for view in served_dict["data"]:
            served_dict["data"][view] = einops.rearrange(served_dict["data"][view][perms],"(batch seq) b c d -> batch seq b c d", batch=d_shape[0], seq=d_shape[1])
            served_dict["inits"][view] = einops.rearrange(served_dict["inits"][view][perms],"(batch seq) b c d -> batch seq b c d", batch=d_shape[0], seq=d_shape[1])
        served_dict["label"] = einops.rearrange(served_dict["label"][perms],"(batch seq) b c d -> batch seq b c d", batch=d_shape[0], seq=d_shape[1])

        return served_dict, perms