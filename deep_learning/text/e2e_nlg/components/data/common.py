# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/12
import functools
import copy

import torch

from components.constants import *


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


def cudify(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return cuda_if_gpu(result)
    return wrapper


@cudify
def ids2var(snt_ids, *dims, addEOS=True):
    snt_ids_copy = copy.deepcopy(snt_ids)
    if addEOS: snt_ids_copy.append(EOS_ID)
    result = torch.autograd.Variable(torch.LongTensor(snt_ids_copy).view(dims))
    return result


if __name__ == '__main__':
    pass


