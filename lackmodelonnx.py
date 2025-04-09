import os
import torch
import onnx
import onnxoptimizer
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3LargeMultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            b = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            b = mobilenet_v3_large(weights=None)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
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
    def __init__(self, in_channels_list, common_size=(7, 7), hidden_dim=128):
        super().__init__()
        self.common_size = common_size
        self.unify = nn.ModuleList()
        for c in in_channels_list:
            self.unify.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(common_size),
                nn.Conv2d(c, hidden_dim, 1),
                nn.ReLU()
            ))
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
    def __init__(self, input_dim, output_dim=2, heatmap_size=(7, 7), final_scale=224):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.final_scale = final_scale
        self.conv = nn.Conv2d(input_dim, 1, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        heatmap = self.conv(x)
        heatmap = heatmap.view(b, h * w)
        softmax_heatmap = torch.softmax(heatmap, dim=1)
        softmax_heatmap = softmax_heatmap.view(b, h, w)
        device = x.device
        grid_y = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(b, h, w)
        grid_x = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(b, h, w)
        exp_x = (softmax_heatmap * grid_x).sum(dim=(1, 2))
        exp_y = (softmax_heatmap * grid_y).sum(dim=(1, 2))
        coords = torch.stack([exp_x, exp_y], dim=1) * self.final_scale
        return coords

class EndToEndMultiScaleModelV2(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2, convlstm_kernel=3, pretrained=True, fusion_common_size=(7, 7)):
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

def main():
    CHECKPOINT_PATH = "./onnxmodel.pth"
    ONNX_MODEL_PATH = "./model.onnx"
    OPT_ONNX_MODEL_PATH = "./optimized_model.onnx"
    
    device = torch.device("cpu")
    
    cp = torch.load(CHECKPOINT_PATH, map_location=device)
    model = EndToEndMultiScaleModelV2(
        hidden_dim=192,
        num_layers=2,
        pretrained=True
    ).to(device)
    model.load_state_dict(cp["model"])
    model.eval()

    dummy_input = torch.randn(1, 132, 1, 224, 224, device=device)
    
    torch.onnx.export(
        model, dummy_input, ONNX_MODEL_PATH,
        input_names=["input"], output_names=["output"],
        opset_version=12
    )
    print(f"ONNX 모델이 {ONNX_MODEL_PATH}로 저장되었습니다.")

    onnx_model = onnx.load(ONNX_MODEL_PATH)
    passes = [
        "eliminate_deadend",
        "eliminate_nop_dropout",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "eliminate_nop_transpose"
    ]
    optimized_model = onnxoptimizer.optimize(onnx_model, passes)
    onnx.save(optimized_model, OPT_ONNX_MODEL_PATH)
    print(f"최적화된 ONNX 모델이 {OPT_ONNX_MODEL_PATH}로 저장되었습니다.")

if __name__ == "__main__":
    main()
