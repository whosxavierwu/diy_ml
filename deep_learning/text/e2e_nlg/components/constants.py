# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/12

PAD_TOKEN = '<blank>'
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'
# we simply ignore 'name' & 'near' field in data set
NAME_TOKEN = '<name>'
NEAR_TOKEN = '<near>'

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

START_VOCAB = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN
]

MR_FIELDS = [
    "name",
    "familyFriendly",
    "eatType",
    "food",
    "priceRange",
    "near",
    "area",
    "customer rating"
]

MR_KEYMAP = {
    field: idx
    for idx, field in enumerate(MR_FIELDS)
}

MR_KEY_NUM = len(MR_FIELDS)


if __name__ == '__main__':
    pass

