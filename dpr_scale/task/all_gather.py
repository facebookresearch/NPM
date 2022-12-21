# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch import Tensor

# taken from https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/distributed.html#all_gather_ddp_if_available
class AllGatherGrad(torch.autograd.Function):

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        tensor: Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = torch.distributed.group.WORLD,
    ) -> Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None]:
        grad_output = torch.cat(grad_output)
        torch.distributed.all_reduce(grad_output,
                                     op=torch.distributed.ReduceOp.SUM,
                                     async_op=False,
                                     group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None

# from https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py
class AllGatherGrad2(torch.autograd.Function):
    """
    all_gather with gradient back-propagation
    """
    @staticmethod
    def forward(ctx, tensor_list, tensor):
        torch.distributed.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = torch.distributed.get_rank()

        #print ("all gather 1")
        dist_ops = [
            torch.distributed.reduce(grad_list[i], i, async_op=True) for i in range(torch.distributed.get_world_size())
        ]

        #print ("all gather 2")
        for op in dist_ops:
            op.wait()

        #print ("all gather 3")
        return None, grad_list[rank]

my_all_gather = AllGatherGrad.apply
my_all_gather2 = AllGatherGrad2.apply


