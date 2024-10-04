import numpy as np
import torch
import torch.nn as nn


class OHEMloss(nn.Module):
    def __init__(self, use_gpu = True):

        super(OHEMloss,self).__init__()

    def hard_negative_mining(self, pred, target):

   
    # Online hard mining on the entire batch
    # :param pred: predicted character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    # :param target: target character or affinity heat map, torch.cuda.FloatTensor, shape = [num_pixels]
    # :return: Online Hard Negative Mining loss
    

        cpu_target = target.data.cpu().numpy()

        all_loss = torch.nn.functional.mse_loss(pred, target, reduction='none')

        positive = np.where(cpu_target >= 0.1)[0]
        negative = np.where(cpu_target <= 0.0)[0]

        positive_loss = all_loss[positive]
        negative_loss = all_loss[negative]

        negative_loss_cpu = np.argsort(
            -negative_loss.data.cpu().numpy())[0:min(max(1000, 3 * positive_loss.shape[0]), negative_loss.shape[0])]

        return (positive_loss.sum() + negative_loss[negative_loss_cpu].sum()) / (
                    positive_loss.shape[0] + negative_loss_cpu.shape[0])


    def forward(self, output, character_map, affinity_map):

        batch_size, height, width, channels = output.shape

        output = output.contiguous().view([batch_size * height * width, channels])

        character = output[:, 0]
        affinity = output[:, 1]

        affinity_map = affinity_map.view([batch_size * height * width])
        character_map = character_map.view([batch_size * height * width])

        loss_character = self.hard_negative_mining(character, character_map)
        loss_affinity = self.hard_negative_mining(affinity, affinity_map)

        # weight character twice then affinity
        all_loss = loss_character * 2 + loss_affinity

        return all_loss