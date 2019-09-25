# -*- coding: utf8 -*-
# Created by: wuzewei
# Created by: 2019/8/26
import progressbar


def create_progress_bar(dynamic_msg=None):
    widgets = [
        ' [batch ', progressbar.SimpleProgress(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') '
    ]
    if dynamic_msg is not None:
        widgets.append(progressbar.DynamicMessage(dynamic_msg))
    return progressbar.ProgressBar(widgets=widgets)


if __name__ == '__main__':
    pass


