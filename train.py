import os
from functools import partial
import datetime
from random import random
import torch
import scipy.signal
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import numpy as np
from yolo import Yolo,EvalCallback
from dataloader import YoloDataset, yolo_dataset

# 设置种子
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# loader的种子
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


#获得类
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

# 获得先验框
def get_anchors(anchors_path):
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def get_lr_scheduler(lr, min_lr, total_iters, step_num = 10):
    def step_lr(lr, decay_rate, step_size, iters):
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
    step_size = total_iters / step_num
    func = partial(step_lr, lr, decay_rate, step_size)
    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param in optimizer.param_groups:
        param['lr'] = lr

# 获得学习率
def getlr(optimizer):
    for param in optimizer.param_groups:
        return param['lr']

def fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda,  save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            images = images.cuda(local_rank)
            targets = [ann.cuda(local_rank) for ann in targets]

        optimizer.zero_grad()
        outputs = model_train(images)

        loss_value_all = 0
        # 计算损失
        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], targets)
            loss_value_all += loss_item
        loss_value = loss_value_all
        # 反向传播
        loss_value.backward()
        optimizer.step()

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': getlr(optimizer)})
            pbar.update(1)

    #模型评估
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            optimizer.zero_grad()
            outputs = model_train(images)

            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)

        # 保存权值
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "epoch_weights.pth"))


class Loss():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
        self.writer.add_graph(model, dummy_input)

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        if len(self.losses) < 25:
            num = 5
        else:
            num = 15

        plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                 label='smooth train loss')
        plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
                 label='smooth val loss')

        # 设定参数
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        # 保存
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


#设置参数
#cpu or gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    Cuda=True
else:
    device = torch.device("cpu")
    Cuda=False
print(device)

#设置种子
seed = 11
fp16 = False

classes_path = 'classes.txt'

# 先验框
anchors_path = 'model_data\\yolo_anchors.txt'
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

#输入的形状
input_shape = [416, 416]

# 冻结阶段
Init_Epoch = 0
Freeze_Epoch = 50
Freeze_batch_size = 8

# 解冻阶段
UnFreeze_Epoch = 100
Unfreeze_batch_size = 4

# 进行冻结训练
Freeze_Train = True

# 最大 最小学习率
Init_lr = 1e-2
Min_lr = Init_lr * 0.01

# 优化器和参数
optimizer_type = "adam"
momentum = 0.937
weight_decay = 0

# 学习率下降方式
lr_decay_type = "step"

# 保存的次数
save_period = 5

# 保存路径
save_dir = 'save'

# 对模型评估
eval_flag = True
eval_period = 5

num_workers = 0

eval_callback = None

# 图片的路径和标签
train_annotation_path = '2012_train.txt'
val_annotation_path = '2012_val.txt'

seed_everything(seed)

local_rank = 0
rank = 0

# 获取classes和anchor
class_names, num_classes = get_classes(classes_path)
anchors, num_anchors = get_anchors(anchors_path)


#创建模型
model = Yolo(anchors_mask, num_classes)

#损失函数 作图
yolo_loss = Loss(anchors, num_classes, input_shape, Cuda, anchors_mask)
if local_rank == 0:
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = Loss(log_dir, model, input_shape=input_shape)
else:
    loss_history = None

#模型训练
model_train = model.train()
model_train = torch.nn.DataParallel(model)
cudnn.benchmark = True
model_train = model_train.cuda()

#读取txt
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()
num_train = len(train_lines)
num_val = len(val_lines)

if local_rank == 0:
    wanted_step = 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

UnFreeze_flag = False

#冻结训练
for param in model.backbone.parameters():
    param.requires_grad = False

batch_size = Freeze_batch_size
nbs = 64
lr_limit_max = 1e-3
lr_limit_min = 3e-4
Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

# 选择优化器
pg0, pg1, pg2 = [], [], []
for k, v in model.named_modules():
    if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        pg2.append(v.bias)
    if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        pg0.append(v.weight)
    elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        pg1.append(v.weight)
optimizer = {
    'adam' : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
}[optimizer_type]
optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
optimizer.add_param_group({"params": pg2})

# 学习率下降公式
lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

# 判断世代长度
epoch_step = num_train // batch_size
epoch_step_val = num_val // batch_size

# 数据loader
train_dataset = YoloDataset(train_lines, input_shape, num_classes, train = True)
val_dataset = YoloDataset(val_lines, input_shape, num_classes, train = False)

train_sampler = None
val_sampler = None
shuffle = True

train_loader = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset, sampler=train_sampler,
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
val_loader = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset, sampler=val_sampler,
                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

if local_rank == 0:
    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines,
                                 log_dir, Cuda, \
                                 eval_flag=eval_flag, period=eval_period)
else:
    eval_callback = None


# 模型训练
for epoch in range(Init_Epoch, UnFreeze_Epoch):

    # 解冻并设置参数
    if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
        batch_size = Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 1e-3
        lr_limit_min = 3e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        train_loader = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset, sampler=train_sampler,worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        val_loader = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,drop_last=True, collate_fn=yolo_dataset, sampler=val_sampler,worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        UnFreeze_flag = True

    set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    # 计算损失
    fit_one_epoch(model_train, model, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_loader, val_loader, UnFreeze_Epoch, Cuda, save_period, save_dir, local_rank)

if local_rank == 0:
    loss_history.writer.close()