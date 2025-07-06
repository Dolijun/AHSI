import tensorboardX
from utils.metric_seg import runningScore
import os
from utils.averager import averager
import time
import torch
import numpy as np
from utils import visual



class Recorder:
    def __init__(self, args):
        # tensorboard
        self.writer = tensorboardX.SummaryWriter(os.path.join(args.save_dir, "logfile"))
        # segmentation metric: cityscapes: 19, sbd: 20 + 1
        if args.dataset in ["sbd", "nyudv2"]:
            self.seg_metric = runningScore(args.nclasses + 1)
        else:
            self.seg_metric = runningScore(args.nclasses)
        # 时间平均
        self.time_rec = averager(args.iters_average)
        # 训练损失平均
        self.iters_average = args.iters_average
        self.loss_avger = averager(self.iters_average)
        self.loss_sed_avger = averager(self.iters_average)
        self.loss_seg_avger = averager(self.iters_average)
        self.loss_edge_avger = averager(self.iters_average)

        self.loss_val_avger = averager(-1)
        self.loss_val_sed_avger = averager(-1)
        self.loss_val_seg_avger = averager(-1)
        self.loss_val_edge_avger = averager(-1)

        self.max_iter = args.num_iters


    def update(self, loss_all, loss_items, start_time=0, is_training=True):
        if is_training:
            self.time_rec.append(time.time() - start_time)
            self.loss_avger.append(loss_all.item())
            self.loss_sed_avger.append(loss_items[0].item())
            self.loss_seg_avger.append(loss_items[1].item())
            self.loss_edge_avger.append(loss_items[2].item())
        else:
            self.loss_val_avger.append(loss_all.item())
            self.loss_val_sed_avger.append(loss_items[0].item())
            self.loss_val_seg_avger.append(loss_items[1].item())
            self.loss_val_edge_avger.append(loss_items[2].item())


    def seg_metric_update(self, results, gt):
        pred = results[1]
        pred = pred.data.max(1)[1].cpu().numpy()
        gt = gt.cpu().numpy()
        self.seg_metric.update(gt, pred)

    def reset_val_recs(self):
        self.seg_metric.reset()
        self.loss_val_avger.clear()
        self.loss_val_sed_avger.clear()
        self.loss_val_seg_avger.clear()
        self.loss_val_edge_avger.clear()

    def logout(self, cur_iter, cur_lr, is_training=True):
        if is_training:
            average_loss = self.loss_avger.get_avg()
            average_loss_sed = self.loss_sed_avger.get_avg()
            average_loss_seg = self.loss_seg_avger.get_avg()
            average_loss_edge = self.loss_edge_avger.get_avg()
            average_time = self.time_rec.get_avg()
            eta = seconds_to_hms(int(average_time * (self.max_iter - cur_iter)))

            print(
                "train_loss: %.4f train_loss_sed: %.4f train_loss_seg: %.4f train_loss_edge: %.4f cur_iter: %d cur_lr: %.4e" % (
                    average_loss, average_loss_sed, average_loss_seg, average_loss_edge, cur_iter, cur_lr),
                "// Avg time/iter: %.4f s" % (average_time), " eta: ", eta)
            self.writer.add_scalar(f'train/{self.iters_average}_all', average_loss, cur_iter)
            self.writer.add_scalar(f'train/{self.iters_average}_sedge', average_loss_sed, cur_iter)
            self.writer.add_scalar(f'train/{self.iters_average}_seg', average_loss_seg, cur_iter)
            self.writer.add_scalar(f'train/{self.iters_average}_edge', average_loss_edge, cur_iter)
            self.writer.add_scalar('train/lr', cur_lr, cur_iter)
        else:
            average_val_loss = self.loss_val_avger.get_avg()
            average_val_loss_sed = self.loss_val_sed_avger.get_avg()
            average_val_loss_seg = self.loss_val_seg_avger.get_avg()
            average_val_loss_edge = self.loss_val_edge_avger.get_avg()
            seg_score, class_iou = self.seg_metric.get_scores()
            print(
                "val_loss: %.4f val_loss_sed: %.4f val_loss_seg: %.4f val_loss_edge: %.4f val_miou: %.4f cur_iter: %d cur_lr: %.4e" % (
                    average_val_loss, average_val_loss_sed, average_val_loss_seg, average_val_loss_edge, seg_score["Mean IoU : \t"],
                    cur_iter, cur_lr))
            self.writer.add_scalar('val/all', average_val_loss, cur_iter)
            self.writer.add_scalar('val/sed', average_val_loss_sed, cur_iter)
            self.writer.add_scalar('val/seg', average_val_loss_seg, cur_iter)
            self.writer.add_scalar('val/edge', average_val_loss_edge, cur_iter)
            self.writer.add_scalar('val/miou', seg_score["Mean IoU : \t"], cur_iter)


def visualize(args, images, labels, cur_iter, results):
    score, pred_seg, pred_edge = results
    image = images[0].cpu().detach().numpy()
    label = labels[0].cpu().detach().numpy()
    prob = torch.sigmoid(score[0]).cpu().detach().numpy()
    prob_edge = torch.sigmoid(pred_edge[0, 0, :, :]).cpu().detach().numpy()
    seg = pred_seg[0].data.cpu().numpy()
    prob_seg = np.argmax(seg, axis=0)
    prob_seg = prob_seg.astype(np.uint8)
    visual.visual_data3(args.dataset, image, label, prob, prob_seg, prob_edge,
                        image_name='visual_map_' + str(cur_iter) + '_次.png',
                        dir=os.path.join(args.save_dir, "visual"), is_show=False)


def visualize_only_sed(args, images, labels, cur_iter, results):
    if args.aux_head is not None:
        score, score_aux = results
    else:
        score_aux = None
        score = results
    image = images[0].cpu().detach().numpy()
    label = labels[0].cpu().detach().numpy()
    prob = torch.sigmoid(score[0]).cpu().detach().numpy()
    prob_aux = prob if score_aux is None else torch.sigmoid(score_aux[0]).cpu().detach().numpy()
    visual.visual_only_sed(args.dataset, image, label, prob, prob_aux=prob_aux,
                        image_name='/visual_map_' + str(cur_iter) + '_次.png',
                        dir=os.path.join(args.save_dir, "visual"), is_show=False)


def seconds_to_hms(n):
    hours = n // 3600
    n %= 3600
    minutes = n // 60
    seconds = n % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


