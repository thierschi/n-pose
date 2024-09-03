import torch.nn as nn


class CombinationLoss(nn.Module):
    def __init__(self, l1, l2, w1=1, w2=1, added_direction=False):
        super(CombinationLoss, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.w1 = w1
        self.w2 = w2
        self.added_direction = added_direction

    def forward(self, output, target):
        pos_output = output[:, :3]
        dir_output = output[:, 3:]

        pos_target = target[:, :3]
        dir_target = target[:, 3:]

        if self.added_direction:
            point2_output = pos_output + dir_output
            point2_target = pos_target + dir_target
            dir_output = point2_output - pos_output
            dir_target = point2_target - pos_target

        loss1 = self.l1(pos_output, pos_target)
        loss2 = self.l2(dir_output, dir_target)

        return self.w1 * loss1 + self.w2 * loss2
