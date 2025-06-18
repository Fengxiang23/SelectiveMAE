import torch
from torch import nn as nn


class TokenSelect_smooth(nn.Module):
    def __init__(
        self,
        expansion_step: list = [0, 200,220, 1180,1200],
        keep_rate: list = [0.25, 0.25, 0.25, 0.25, 0.25],
        initialization_keep_rate: float = 0.15,
        expansion_multiple_stage: int = 2,
        distance: str = "cosine",
        smooth_scale_epoch=20
    ):
        super().__init__()
        self.expansion_stage = 0
        self.sparse_inference = True

        self.expansion_step = expansion_step
        self.total_expansion_stage = len(expansion_step)
        self.initialization_keep_rate = initialization_keep_rate
        self.change_epoch_first=0
        self.change_epoch_second = 0
        self.smooth_scale_epoch=smooth_scale_epoch

        self.expansion_keep_rate = []
        self.keep_rate = keep_rate
        for i in range(len(keep_rate)):
            if i == 0:
                self.expansion_keep_rate.append(keep_rate[i] - initialization_keep_rate)
            else:
                self.expansion_keep_rate.append(keep_rate[i] - initialization_keep_rate)

        self.final_keep_rate = keep_rate[-1]
        self.expansion_multiple_stage = expansion_multiple_stage
        self.distance = distance

    def update_current_stage(self, epoch: int):
        import bisect
        expansion_stage = bisect.bisect_right(self.expansion_step, epoch)

        if epoch == self.expansion_step[1]:
            self.expansion_stage=1
        elif epoch == self.expansion_step[3]:
            self.expansion_stage = 3
        else:
            self.expansion_stage = expansion_stage
        if expansion_stage==2:
            self.change_epoch_first =epoch-self.expansion_step[1]
        elif expansion_stage==4:
            self.change_epoch_second =epoch-self.expansion_step[3]




    def get_score(self, a: torch.Tensor, b: torch.Tensor):
        if self.distance == "cosine":
            dist = a @ b.transpose(-1, -2)
        elif self.distance == "manhattan":
            dist = torch.sum(
                torch.abs(a.unsqueeze(2) - b.unsqueeze(1)),
                dim=-1,
            )
        elif self.distance == "euclidean":
            dist = torch.sqrt(torch.sum((a.unsqueeze(2) - b.unsqueeze(1)) ** 2, dim=-1))
        else:
            raise Exception("Wrong distance!", self.distance)
        return dist

    def token_initialization(self, token: torch.Tensor):
        x = int((self.token_num - 1) * self.initialization_keep_rate)
        step = int(1 // self.initialization_keep_rate)
        with torch.no_grad():
            select_index = []
            unselect_index = []
            for i in range(self.token_num - 1):
                if i % step == 0 and len(select_index) < x:
                    select_index.append(i)
                else:
                    unselect_index.append(i)
            select_index = (
                torch.tensor(select_index)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device=token.device)
            ).expand(
                token.shape[0],
                x,
                token.shape[2],
            )
            unselect_index = (
                torch.tensor(unselect_index)
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(device=token.device)
            ).expand(
                token.shape[0],
                token.shape[1] - x,
                token.shape[2],
            )

        select_token = token.gather(dim=1, index=select_index)
        unselect_token = token.gather(dim=1, index=unselect_index)

        assert select_token.shape[1] + unselect_token.shape[1] == (
            self.token_num - 1
        ), "Wrong shape!"
        assert select_index.shape[1] + unselect_index.shape[1] == (
            self.token_num - 1
        ), "Wrong shape!"

        return (select_token, select_index), (unselect_token, unselect_index)

    def token_expansion(
        self,
        select_token: torch.Tensor,
        select_index: torch.Tensor,
        unselect_token: torch.Tensor,
        unselect_index: torch.Tensor,x
    ):
        self.token_num = x.shape[1]

        for stage in range(1, 2):
            expansion_token_num = int(self.token_num  * self.keep_rate[stage - 1])-int(
                    self.token_num  * self.initialization_keep_rate)

            for k in range(1, self.expansion_multiple_stage + 1):
                if k == self.expansion_multiple_stage:
                    multiple_expansion_token_num = expansion_token_num - (
                        self.expansion_multiple_stage - 1
                    ) * (expansion_token_num // self.expansion_multiple_stage)
                else:
                    multiple_expansion_token_num = (
                            expansion_token_num // self.expansion_multiple_stage
                    )
                with torch.no_grad():
                    select_token_norm = select_token / select_token.norm(
                        dim=-1, keepdim=True
                    )
                    unselect_token_norm = unselect_token / unselect_token.norm(
                        dim=-1, keepdim=True
                    )
                    scores = self.get_score(unselect_token_norm, select_token_norm)
                    node_max, node_idx = scores.max(dim=-1)
                    edge_idx = node_max.argsort(dim=-1, descending=False)


                    if self.expansion_stage==1:
                        add_node_index = edge_idx[..., -multiple_expansion_token_num:]
                        unadd_node_index = edge_idx[..., :-multiple_expansion_token_num]
                    elif self.expansion_stage==2:
                        near_ratio = 1 - (self.change_epoch_first / (self.smooth_scale_epoch))
                        far_ratio = (self.change_epoch_first / (self.smooth_scale_epoch))

                        near_node_num = int(near_ratio * multiple_expansion_token_num)
                        far_node_num = multiple_expansion_token_num - near_node_num

                        if near_node_num > 0:
                            near_node_indices = edge_idx[..., -near_node_num:]
                        else:
                            near_node_indices = torch.tensor([], dtype=torch.long, device=edge_idx.device)

                        if far_node_num > 0:
                            far_node_indices = edge_idx[..., :far_node_num]
                        else:
                            far_node_indices = torch.tensor([], dtype=torch.long, device=edge_idx.device)


                        add_node_index = torch.cat((near_node_indices, far_node_indices), dim=-1)
                        mask = torch.isin(edge_idx[1], add_node_index[1])
                        mask = ~mask
                        unadd_node_index = torch.masked_select(edge_idx, mask).view(edge_idx.shape[0], -1)

                    elif self.expansion_stage==3:
                        add_node_index = edge_idx[..., :multiple_expansion_token_num]
                        unadd_node_index = edge_idx[..., multiple_expansion_token_num:]
                    elif self.expansion_stage==4:
                        far_ratio = 1-(self.change_epoch_second / (self.smooth_scale_epoch))
                        far_node_num = int(far_ratio * multiple_expansion_token_num)
                        random_node_num=multiple_expansion_token_num-far_node_num
                        far_node_indices = edge_idx[..., :far_node_num]
                        unadd_node_index = edge_idx[..., far_node_num:]
                        total_indices = unadd_node_index.shape[-1]  # 这通常是等于172，根据初始设置
                        random_indices = torch.randperm(total_indices, device=edge_idx.device)
                        random_add_node_index = random_indices[:random_node_num]
                        random_node_index = unadd_node_index[..., random_add_node_index]
                        add_node_index = torch.cat(( far_node_indices,random_node_index), dim=-1)
                        mask = torch.isin(edge_idx[-1], add_node_index[-1])
                        mask = ~mask
                        unadd_node_index = torch.masked_select(edge_idx, mask).view(edge_idx.shape[0], -1)

                    else:
                        total_indices = edge_idx.shape[-1]
                        random_indices = torch.randperm(total_indices, device=edge_idx.device)
                        random_add_node_index = random_indices[:multiple_expansion_token_num]
                        add_node_index = edge_idx[..., random_add_node_index]
                        random_unadd_node_index = random_indices[multiple_expansion_token_num:]
                        unadd_node_index = edge_idx[..., random_unadd_node_index]


                add_index = unselect_index.gather(
                    dim=1,
                    index=add_node_index.expand(
                        unselect_token.shape[0],
                        multiple_expansion_token_num
                    ),
                )
                add_token = unselect_token.gather(
                    dim=1,
                    index=add_node_index[..., None].expand(
                        unselect_token.shape[0],
                        multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )

                select_index = torch.cat([select_index, add_index], dim=1)
                select_token = torch.cat([select_token, add_token], dim=1)

                unselect_index = unselect_index.gather(
                    dim=1,
                    index=unadd_node_index.expand(
                        unselect_token.shape[0],
                        unselect_token.shape[1] - multiple_expansion_token_num,
                    ),
                )
                unselect_token = unselect_token.gather(
                    dim=1,
                    index=unadd_node_index[..., None].expand(
                        unselect_token.shape[0],
                        unselect_token.shape[1] - multiple_expansion_token_num,
                        unselect_token.shape[2],
                    ),
                )

        return (select_token, select_index), (unselect_token, unselect_index)

    def token_merge(
        self,
        select_token: torch.Tensor,
        select_index: torch.Tensor,
        unselect_token: torch.Tensor,
        unselect_index: torch.Tensor,
        mode="mean",
    ):
        rest_token_num = unselect_token.shape[1]

        with torch.no_grad():
            select_token_norm = select_token / select_token.norm(dim=-1, keepdim=True)
            unselect_token_norm = unselect_token / unselect_token.norm(
                dim=-1, keepdim=True
            )
            scores = self.get_score(unselect_token_norm, select_token_norm)

            node_max, node_idx = scores.max(dim=-1)
            merge_unselect_node_index = node_idx[..., None]

        select_token = select_token.scatter_reduce(
            dim=1,
            index=merge_unselect_node_index.expand(
                unselect_token.shape[0],
                rest_token_num,
                unselect_token.shape[2],
            ),
            src=unselect_token,
            reduce=mode,
        )

        return (select_token, select_index)
