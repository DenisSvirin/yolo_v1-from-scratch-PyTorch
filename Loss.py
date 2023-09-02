from metrics import IoU
import torch.nn as nn
import torch


class Yolo_loss(nn.Module):
    def __init__(self, c=20):
        super().__init__()
        self.c = c
        self.mse = nn.MSELoss(reduction="sum")
        self.l_coord = 5
        self.l_noobj = 0.5

    def forward(self, predictions, labels):
        predictions = predictions.reshape(-1, 7, 7, 5 * 2 + 20)

        obj_exist = labels[..., 0].unsqueeze(3)

        # Box predicion loss

        true_bb = labels[2:6]
        bb1 = predictions[..., 2:6]
        bb2 = predictions[..., 7:12]

        IoU_for_bb1 = IoU(bb1, true_bb)
        IoU_for_bb2 = IoU(bb2, true_bb)

        IoUs = torch.cat([IoU_for_bb1.unsqueeze(0), IoU_for_bb2.unsqueeze(0)], dim=0)
        best_bb, best_bb_ind = torch.max(IoUs, dim=0)

        box_pred = obj_exist * ((1 - best_bb_ind) * bb1 + best_bb_ind * bb2)

        box_center_loss = nn.mse(box_pred[..., 0:2], labels[..., 1:3])

        bb_shape_pred = torch.sign(box_pred[..., 2:4]) * torch.sqrt(
            torch.abs(box_pred[..., 2:4] + 1e-5)
        )
        box_shape_loss = nn.mse(bb_shape_pred, torch.sqrt(labels[..., 3:5]))

        # class loss
        pred_classes = predictions[..., 10:31]
        true_classes = labels[..., 10:31]

        class_loss = self.mse(obj_exist * pred_classes, obj_prob * true_classes)
