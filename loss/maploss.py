import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()

    def single_image_loss(self, pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        internel = batch_size
        for i in range(batch_size):
            average_number = 0
            loss = torch.mean(pre_loss.view(-1)) * 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= 0.1)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < 0.1)])
                    average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                    average_number += 3*positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                average_number += 500
                sum_loss += nega_loss

        return sum_loss


    def forward(self, gh_label, p_gh, mask):
        gh_label = gh_label
        p_gh = p_gh
        loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

        assert p_gh.size() == gh_label.size()
        loss1 = loss_fn(p_gh, gh_label)
        loss_g = torch.mul(loss1, mask)

        word_loss = self.single_image_loss(loss_g, gh_label)
        return word_loss/loss_g.shape[0]