# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/12
from components.constants import *
import torch


def pad_snt(snt_ids_trunc, max_len):
    snt_ids_trunc_pad = snt_ids_trunc + [PAD_ID] * (max_len - len(snt_ids_trunc))
    return snt_ids_trunc_pad


use_cuda = torch.cuda.is_available()

def cuda_if_gpu(T):
    """
    Move tensor to GPU if it's available
    :param T:
    :return:
    """
    return T.cuda() if use_cuda else T


if __name__ == '__main__':
    pass

