import os
import sys
import cv2
import math
import torch
import wandb
import optuna
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MultiVideoSlidingDataset(Dataset):
    def __init__(self, data_root, seq_length=44, input_size=(224,224), use_cache=False):
        super().__init__()
        self.data_root = data_root
        self.seq_length = seq_length
        self.input_h, self.input_w = input_size
        self.use_cache = use_cache
        self.video_list = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root,d))])
        self.all_sequences = []
        self.video_infos = []
        for vid_idx, vdir in enumerate(self.video_list):
            vp = os.path.join(data_root, vdir)
            fd = os.path.join(vp, 'frames')
            af = os.path.join(vp, 'annotations.txt')
            cd = os.path.join(vp, 'cache')
            frs = sorted([f for f in os.listdir(fd) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]))
            ann = {}
            with open(af, 'r') as f:
                for line in f:
                    i, x, y, w, h = line.strip().split()
                    ann[int(i)] = (float(x), float(y))
            if len(frs) == 0:
                continue
            if self.use_cache:
                os.makedirs(cd, exist_ok=True)
            img0 = cv2.imread(os.path.join(fd, frs[0]))
            oh, ow = img0.shape[:2]
            self.video_infos.append({'fd': fd, 'frs': frs, 'ann': ann, 'cd': cd, 'oh': oh, 'ow': ow})
            F = len(frs)
            L = seq_length
            num_seq = F - (L - 1)
            if num_seq < 1:
                continue
            for start_idx in range(num_seq):
                self.all_sequences.append((vid_idx, start_idx))
        if len(self.video_infos) == 0:
            self.orig_h, self.orig_w = 0, 0
        else:
            self.orig_h, self.orig_w = self.video_infos[0]['oh'], self.video_infos[0]['ow']
    def __len__(self):
        return len(self.all_sequences)
    def _load_one_frame(self, vid_idx, frame_idx):
        info = self.video_infos[vid_idx]
        fd, frs, ann, cd = info['fd'], info['frs'], info['ann'], info['cd']
        fname = frs[frame_idx]
        nidx = int(os.path.splitext(fname)[0])
        path = os.path.join(fd, fname)
        cp = os.path.join(cd, f"{nidx}.pt")
        if self.use_cache and os.path.exists(cp):
            return torch.load(cp)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        ir = cv2.resize(img, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA).astype(np.float32)
        ir = ir / 255.0
        sx = self.input_w / float(W)
        sy = self.input_h / float(H)
        gx, gy = ann.get(nidx, (0.0, 0.0))
        gx2 = gx * sx
        gy2 = gy * sy
        timg = torch.from_numpy(ir).float().unsqueeze(0)
        if self.use_cache:
            torch.save((timg, (gx2, gy2)), cp)
        return timg, (gx2, gy2)
    def __getitem__(self, idx):
        vid_idx, st = self.all_sequences[idx]
        fs = []
        ps = []
        for i in range(self.seq_length):
            frame_idx = st + i
            timg, (gx, gy) = self._load_one_frame(vid_idx, frame_idx)
            fs.append(timg)
            ps.append([gx, gy])
        fs = torch.stack(fs, dim=0)
        ps = torch.tensor(ps, dtype=torch.float32)
        return fs, ps

class MobileNetV3LargeMultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            b = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            b = mobilenet_v3_large(weights=None)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.stage1 = nn.Sequential(*b.features[0:3])
        self.stage2 = nn.Sequential(*b.features[3:7])
        self.stage3 = nn.Sequential(*b.features[7:11])
        self.stage4 = nn.Sequential(*b.features[11:14])
        self.stage5 = nn.Sequential(*b.features[14:17])
    def forward(self, x):
        x = (x - self.mean) / self.std
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return f1, f2, f3, f4, f5

class CrossScaleAttentionFusion(nn.Module):
    def __init__(self, in_channels_list, common_size=(28,28), hidden_dim=128):
        super().__init__()
        self.common_size = common_size
        self.unify = nn.ModuleList()
        for c in in_channels_list:
            self.unify.append(nn.Sequential(nn.AdaptiveAvgPool2d(common_size), nn.Conv2d(c, hidden_dim, 1), nn.ReLU()))
        num_features = len(in_channels_list)
        self.weight_conv = nn.Conv2d(hidden_dim * num_features, num_features, 3, padding=1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, features):
        u = []
        for i, feat in enumerate(features):
            x = self.unify[i](feat)
            u.append(x)
        c = torch.cat(u, dim=1)
        w = self.weight_conv(c)
        w = self.softmax(w)
        s = []
        num_features = len(u)
        for i in range(num_features):
            h = u[i] * w[:, i:i+1]
            s.append(h)
        f = sum(s)
        return f

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)
        self.hidden_dim = hidden_dim
    def forward(self, x, h_cur, c_cur):
        c = torch.cat([x, h_cur], dim=1)
        o = self.conv(c)
        i, f, o2, g = torch.split(o, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o2 = torch.sigmoid(o2)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o2 * torch.tanh(c_next)
        return h_next, c_next

class TemporalConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            d = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(d, hidden_dim, kernel_size))
    def forward(self, x):
        B, T, C, H, W = x.shape
        hs = []
        cs = []
        for i in range(self.num_layers):
            hs.append(torch.zeros(B, self.cells[i].hidden_dim, H, W, device=x.device))
            cs.append(torch.zeros(B, self.cells[i].hidden_dim, H, W, device=x.device))
        out = []
        for t in range(T):
            z = x[:, t]
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(z, hs[i], cs[i])
                z = hs[i]
            out.append(z.unsqueeze(1))
        out = torch.cat(out, dim=1)
        return out

class RegressionHead(nn.Module):
    def __init__(self, input_dim, output_dim=2, heatmap_size=(28,28), final_scale=224):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.final_scale = final_scale
        self.conv = nn.Conv2d(input_dim, 1, 1)
    def forward(self, x):
        b, c, h, w = x.size()
        heatmap = self.conv(x)
        heatmap = heatmap.view(b, h*w)
        softmax_heatmap = torch.softmax(heatmap, dim=1)
        softmax_heatmap = softmax_heatmap.view(b, h, w)
        device = x.device
        grid_y = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(b, h, w)
        grid_x = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(b, h, w)
        exp_x = (softmax_heatmap * grid_x).sum(dim=(1,2))
        exp_y = (softmax_heatmap * grid_y).sum(dim=(1,2))
        coords = torch.stack([exp_x, exp_y], dim=1) * self.final_scale
        return coords

class EndToEndMultiScaleModelV2(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, convlstm_kernel=3, pretrained=True, fusion_common_size=(28,28)):
        super().__init__()
        self.backbone = MobileNetV3LargeMultiScale(pretrained=pretrained)
        in_channels_list = [24, 40, 80, 160, 960]
        self.fusion = CrossScaleAttentionFusion(in_channels_list, common_size=fusion_common_size, hidden_dim=hidden_dim)
        self.temporal = TemporalConvLSTM(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=convlstm_kernel, num_layers=num_layers)
        self.reg_head = RegressionHead(input_dim=hidden_dim, output_dim=2, heatmap_size=fusion_common_size, final_scale=224)
    def forward(self, seq):
        B, T, C, H, W = seq.shape
        s = []
        for t in range(T):
            x = seq[:, t]
            if C == 1:
                x = x.repeat(1, 3, 1, 1)
            f1, f2, f3, f4, f5 = self.backbone(x)
            f = self.fusion([f1, f2, f3, f4, f5])
            s.append(f.unsqueeze(1))
        z = torch.cat(s, dim=1)
        o = self.temporal(z)
        p = []
        for t in range(T):
            r = self.reg_head(o[:, t])
            p.append(r.unsqueeze(1))
        p = torch.cat(p, dim=1)
        return p

class GradientClipCallback:
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm
    def __call__(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

class LRWarmupCallback:
    def __init__(self, optimizer, warmup_epochs=5, target_lr=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr or optimizer.param_groups[0]['lr']
        self.initial_lr = self.target_lr / self.warmup_epochs
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            for p in self.optimizer.param_groups:
                p['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class SmoothL1LossNoInertia(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.lossfn = nn.SmoothL1Loss(beta=beta, reduction='mean')
    def forward(self, pred, gt):
        return self.lossfn(pred, gt)

def calc_distance_accuracy(pred, gt, threshold=5.0):
    with torch.no_grad():
        dist = (pred - gt).pow(2).sum(dim=2).sqrt()
        mask = (dist <= threshold).float()
        return mask.mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device, callbacks=None, epoch_idx=0, max_epoch=100):
    model.train()
    metrics = defaultdict(float)
    pbar = tqdm(loader, desc=f"Train Epoch {epoch_idx}/{max_epoch}", leave=False)
    for batch_idx, (fs, ps) in enumerate(pbar):
        fs = fs.to(device)
        ps = ps.to(device)
        optimizer.zero_grad()
        pr = model(fs)
        loss = criterion(pr, ps)
        loss.backward()
        if callbacks and 'grad_clip' in callbacks:
            callbacks['grad_clip'](model)
        optimizer.step()
        metrics['train_loss'] += loss.item()
        acc = calc_distance_accuracy(pr, ps)
        metrics['train_acc'] += acc
        rmse = torch.sqrt(((pr - ps)**2).mean()).item()
        metrics['train_rmse'] += rmse
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}", "rmse": f"{rmse:.4f}"})
    n = len(loader)
    for k in metrics:
        metrics[k] /= n
    return dict(metrics)

@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_idx=0, max_epoch=100):
    model.eval()
    metrics = defaultdict(float)
    pbar = tqdm(loader, desc=f"Val Epoch {epoch_idx}/{max_epoch}", leave=False)
    for batch_idx, (fs, ps) in enumerate(pbar):
        fs = fs.to(device)
        ps = ps.to(device)
        pr = model(fs)
        loss = criterion(pr, ps).item()
        metrics['val_loss'] += loss
        acc = calc_distance_accuracy(pr, ps)
        metrics['val_acc'] += acc
        rmse = torch.sqrt(((pr - ps)**2).mean()).item()
        metrics['val_rmse'] += rmse
        pbar.set_postfix({"val_loss": f"{loss:.4f}", "val_acc": f"{acc:.4f}", "val_rmse": f"{rmse:.4f}"})
    n = len(loader)
    for k in metrics:
        metrics[k] /= n
    return dict(metrics)

def objective(trial, train_dataset, val_dataset, device, base_config):
    run = wandb.init(project=base_config['wandb_project'], name=f"trial_{trial.number}_h100", reinit=True)
    c = {}
    c['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    c['hidden_dim'] = trial.suggest_int('hidden_dim', 192, 384)
    c['num_layers'] = trial.suggest_int('num_layers', 2, 3)
    wandb.config.update(c)
    model = EndToEndMultiScaleModelV2(hidden_dim=c['hidden_dim'], num_layers=c['num_layers'], pretrained=True).to(device)
    full_train_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    if len(full_train_dataset.video_infos) > 0:
        oh = full_train_dataset.video_infos[0]['oh']
        ow = full_train_dataset.video_infos[0]['ow']
    optimizer = optim.AdamW(model.parameters(), lr=c['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = SmoothL1LossNoInertia(beta=1.0)
    grad_clip = GradientClipCallback(max_norm=1.0)
    warmup = LRWarmupCallback(optimizer, warmup_epochs=5)
    es = EarlyStopping(patience=10)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    best = {'val_loss': float('inf'), 'val_acc': 0.0, 'val_rmse': float('inf')}
    e = 0
    max_epoch = 200
    while not es.early_stop and e < max_epoch:
        warmup.step(e)
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, callbacks={'grad_clip': grad_clip}, epoch_idx=e, max_epoch=max_epoch)
        va = evaluate(model, val_loader, criterion, device, epoch_idx=e, max_epoch=max_epoch)
        scheduler.step(va['val_loss'])
        es(va['val_loss'])
        if va['val_loss'] < best['val_loss']:
            best = va
            torch.save({'epoch': e, 'model': model.state_dict(), 'opt': optimizer.state_dict(), 'best': best, 'conf': c}, f"trial_{trial.number}_best.pth")
        wandb.log({'epoch': e, 'train_loss': tr['train_loss'], 'train_acc': tr['train_acc'], 'train_rmse': tr['train_rmse'], 'val_loss': va['val_loss'], 'val_acc': va['val_acc'], 'val_rmse': va['val_rmse'], 'lr': optimizer.param_groups[0]['lr']})
        trial.report(va['val_loss'], e)
        if trial.should_prune():
            run.finish()
            raise optuna.TrialPruned()
        e += 1
    run.finish()
    return best['val_loss']

if __name__ == "__main__":
    config = {'data_root': './data', 'seq_length': 132, 'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'n_trials': 10, 'wandb_project': 'JustTracker', 'val_ratio': 0.2, 'use_cache': False}
    ds = MultiVideoSlidingDataset(data_root=config['data_root'], seq_length=config['seq_length'], input_size=(224,224), use_cache=config['use_cache'])
    idxs = list(range(len(ds)))
    tr_idx, val_idx = train_test_split(idxs, test_size=config['val_ratio'], shuffle=True)
    tr_ds = Subset(ds, tr_idx)
    val_ds = Subset(ds, val_idx)
    device = torch.device(config['device'])
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda t: objective(t, tr_ds, val_ds, device, config), n_trials=config['n_trials'])
    best_trial = study.best_trial
    for k, v in best_trial.params.items():
        print(k, v)
    wandb.finish()
