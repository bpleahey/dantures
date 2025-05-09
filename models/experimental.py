import numpy as np
import random
import torch
import torch.nn as nn
import cv2
import os
from utils.general import xywh2xyxy, xyxy2xywh
from utils.plots import plot_one_box
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

from models.common import Conv, DWConv
from utils.google_utils import attempt_download

class DualLayer(nn.Module):
    def __init__(self, rgb_branch, lwir_branch):
        super().__init__()
        self.rgb_branch = rgb_branch
        self.lwir_branch = lwir_branch

    def forward(self, x):
        # print(f"[DualLayer] Received input: len={len(x)}, types={[type(e) for e in x]}") debug line

        if not isinstance(x, (list, tuple)):
            raise ValueError(f"DualLayer expected tuple/list input, got {type(x)}")

        # Case 1: (rgb, lwir, time_idx)
        if len(x) == 3 and all(isinstance(i, torch.Tensor) for i in x):
            rgb_x, lwir_x, time_idx = x

        # Case 2: ([rgb, time], [lwir, time])
        elif len(x) == 2 and all(isinstance(i, (list, tuple)) and len(i) == 2 for i in x):
            (rgb_x, rgb_time), (lwir_x, lwir_time) = x
            if not torch.equal(rgb_time, lwir_time):
                raise ValueError("Mismatched time_idx in DualLayer inputs.")
            time_idx = rgb_time

        # Case 3: N * [(feat, time_idx)]
        elif all(isinstance(i, (list, tuple)) and len(i) == 2 for i in x):
            feats, times = zip(*x)
            if not all(torch.equal(t, times[0]) for t in times):
                raise ValueError("Mismatched time_idx across inputs to DualLayer.")
            time_idx = times[0]
            n = len(feats) // 2
            rgb_x = torch.cat(feats[:n], dim=1)
            lwir_x = torch.cat(feats[n:], dim=1)

        # Case 4: List of (rgb_feat, lwir_feat, time_idx) tuples
        elif all(isinstance(e, (tuple, list)) and len(e) == 3 for e in x):
            rgb_feats, lwir_feats, times = zip(*x)
            time_idx = times[0]
            if not all(torch.equal(t, time_idx) for t in times):
                raise ValueError(f"Time indices are not all equal: {[t.item() for t in times]}")
            rgb_x = torch.cat(rgb_feats, dim=1)
            lwir_x = torch.cat(lwir_feats, dim=1)

        else:
            raise ValueError(
                f"DualLayer expected (rgb, lwir, time_idx), ([rgb, time], [lwir, time]), or N*[(feat, time)], got: {[type(e) for e in x]}"
            )

        # Process each branch separately
        rgb_out = self.rgb_branch(rgb_x)
        lwir_out = self.lwir_branch(lwir_x)

        # No fusion: return the tuple for downstream use
        return (rgb_out, lwir_out, time_idx)


#TODO: complete and test symmetriccrossattention block
class SymmetricCrossAttention(nn.Module):
    def __init__(self, channels, time_dim=3, mode="manual", downsample=True, ds_factor=4):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.downsample = downsample
        self.ds_factor = ds_factor

        self.query_rgb = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_ir = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_ir = nn.Conv2d(channels, channels, kernel_size=1)

        self.query_ir = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_rgb = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_rgb = nn.Conv2d(channels, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        if mode == "learned":
            self.time_proj = nn.Linear(time_dim, 2 * channels)
            self.gate_proj = nn.Sequential(
                nn.Conv2d(2 * channels, 1, kernel_size=1),
            )
        elif mode == "manual":
            self.register_buffer("alpha_table", torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

    def forward(self, x, targets=None):
        if targets is not None:
            z_rgb, z_ir, time_idx = x[0]
        else:
            z_rgb, z_ir, time_idx = x

        B, C, H, W = z_rgb.shape

        # === Optional Spatial Downsampling ===
        if self.downsample:
            z_rgb_ds = F.interpolate(z_rgb, scale_factor=1/self.ds_factor, mode='bilinear', align_corners=False)
            z_ir_ds = F.interpolate(z_ir, scale_factor=1/self.ds_factor, mode='bilinear', align_corners=False)
            H_ds, W_ds = z_rgb_ds.shape[2:]
        else:
            z_rgb_ds, z_ir_ds = z_rgb, z_ir
            H_ds, W_ds = H, W

        # === 1. Cross Attention ===
        q_rgb = self.query_rgb(z_rgb_ds).view(B, C, -1).permute(0, 2, 1)
        k_ir = self.key_ir(z_ir_ds).view(B, C, -1).permute(0, 2, 1)
        v_ir = self.value_ir(z_ir_ds).view(B, C, -1).permute(0, 2, 1)

        q_ir = self.query_ir(z_ir_ds).view(B, C, -1).permute(0, 2, 1)
        k_rgb = self.key_rgb(z_rgb_ds).view(B, C, -1).permute(0, 2, 1)
        v_rgb = self.value_rgb(z_rgb_ds).view(B, C, -1).permute(0, 2, 1)

        attn_rgb_to_ir = self.softmax(q_rgb @ k_ir.transpose(-2, -1) / (C ** 0.5))
        attn_ir_to_rgb = self.softmax(q_ir @ k_rgb.transpose(-2, -1) / (C ** 0.5))

        z_rgb_prime = attn_rgb_to_ir @ v_ir
        z_ir_prime = attn_ir_to_rgb @ v_rgb

        z_rgb_prime = z_rgb_prime.permute(0, 2, 1).view(B, C, H_ds, W_ds)
        z_ir_prime = z_ir_prime.permute(0, 2, 1).view(B, C, H_ds, W_ds)

        # === 2. Fusion ===
        if self.mode == "manual":
            alpha = self.alpha_table.to(dtype=z_rgb.dtype, device=z_rgb.device)[time_idx].view(-1, 1, 1, 1)
            fused = alpha * z_rgb_prime + (1 - alpha) * z_ir_prime
        else:
            if time_idx.ndim == 1:
                time_onehot = F.one_hot(time_idx, num_classes=3).float()
            else:
                time_onehot = time_idx.float()
            t_proj = self.time_proj(time_onehot).unsqueeze(-1).unsqueeze(-1)

            z_fusion = torch.cat([z_rgb_prime, z_ir_prime], dim=1)
            gate_logits = self.gate_proj(z_fusion + t_proj.expand(-1, -1, H_ds, W_ds))
            gate = torch.sigmoid(gate_logits)
            fused = gate * z_rgb_prime + (1 - gate) * z_ir_prime

        # === Optional Upsampling ===
        if self.downsample and (H_ds != H or W_ds != W):
            fused = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=False)

        if not self.training and random.random() < 0.01:
            if targets is not None:
                for i in range(z_rgb.shape[0]):
                    matching_targets = targets[targets[:, 0] == i]
                    self.save_fused_tensor(fused[i], targets=matching_targets)
            else:
                for i in range(z_rgb.shape[0]):
                    self.save_fused_tensor(fused[i])

        return fused.to(dtype=z_rgb.dtype, device=z_rgb.device)
    
    @property
    def out_channels(self):
        return self.channels

    def save_fused_tensor(self, x, targets=None):
        # TODO: add in tensor saving logic
        pass

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super(CrossConv, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super(Sum, self).__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1., n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depthwise Conv https://arxiv.org/abs/1907.09595
    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, groups - 1E-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(groups)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output





class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)


class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=1,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None, n_classes=80):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh # if max_wh != 0 : non-agnostic else : agnostic
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)
        self.n_classes=n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat([X, selected_boxes, selected_categories, selected_scores], 1)

class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes

    def forward(self, x):
        boxes = x[:, :, :4]
        conf = x[:, :, 4:5]
        scores = x[:, :, 5:]
        if self.n_classes == 1:
            scores = conf # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            scores *= conf  # conf = obj_conf * cls_conf
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(boxes, scores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None, n_classes=80):
        super().__init__()
        device = device if device else torch.device('cpu')
        assert isinstance(max_wh,(int)) or max_wh is None
        self.model = model.to(device)
        self.model.model[-1].end2end = True
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device, n_classes)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, mode="manual"):
        super().__init__()
        self.mode = mode
        if mode == "learned":
            # self.time_embedding = nn.Embedding(3, 8)  # 3 time categories
            # self.predictor = nn.Sequential(
            #     nn.Linear(8, 16),
            #     nn.ReLU(),
            #     nn.Linear(16, 1),
            #     nn.Sigmoid()
            # )
            self.alpha_table = nn.Parameter(torch.tensor([0.7, 0.5, 0.3],dtype=torch.float32))
        elif mode == "manual":
            #TODO: toggle these ratios and see what works best
            self.register_buffer("alpha_table", torch.tensor([0.7, 0.5, 0.3], dtype=torch.float32))  # noon, post_sunrise_or_pre_sunset, pre_sunrise_or_post_sunset
   
    def forward(self, x, targets=None):
        # if self.mode == "learned":
        #     time_embed = self.time_embedding(time_idx)
        #     alpha = self.predictor(time_embed).view(-1, 1, 1, 1)
        # elif self.mode == "manual":
        #     alpha = self.alpha_table[time_idx].view(-1, 1, 1, 1).to(rgb.device)

        if targets is not None:
            (rgb, lwir, time_idx) = x[0]
        else:
            (rgb, lwir, time_idx) = x

        alpha = self.alpha_table.to(dtype=rgb.dtype, device=rgb.device)[time_idx].view(-1, 1, 1, 1)

        #TODO: decide if i want to be saving images or not, change threshold
        # if self.training:  # Only save during training
        # # Save a few examples
        
        fused = alpha * rgb + (1 - alpha) * lwir

        if not self.training and random.random() < .01:  # only 1% of batches
            # save_fused_tensor(alpha * rgb + (1 - alpha) * lwir)
            if targets is not None:
                for i in range(rgb.shape[0]):  # for each image in batch
                    matching_targets = targets[targets[:, 0] == i]
                    self.save_fused_tensor(fused[i], targets=matching_targets)
            else:
                for i in range(rgb.shape[0]):
                    self.save_fused_tensor(fused[i])

        # print(f"[FusionLayer] output shape: {out.shape}")  # Debug line
        
        return fused.to(dtype=rgb.dtype, device=rgb.device)
    @staticmethod
    def save_fused_tensor(x_fused, targets=None, idx=None):
        """
        Saves a fused tensor as a .jpg image with optional bounding boxes.
        For mid-fusion features, applies random visualization strategy.

        Args:
            x_fused (Tensor or ndarray): (C, H, W)
            targets (Tensor or ndarray): (num_boxes, 6) in normalized xywh format
            idx (int): Optional index for filename uniqueness
        """
        if isinstance(x_fused, torch.Tensor):
            x_fused = x_fused.detach().cpu().numpy()
        assert x_fused.ndim == 3, "Expected 3D tensor (C, H, W)"
        C, H, W = x_fused.shape

        os.makedirs('fused_samples', exist_ok=True)
        if idx is None:
            idx = random.randint(0, 999999)

        if C == 3:
            # Treat as RGB image
            img = np.transpose(x_fused, (1, 2, 0))
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Draw targets
            if targets is not None and len(targets):
                if isinstance(targets, torch.Tensor):
                    targets = targets.cpu().numpy()
                boxes = xywh2xyxy(targets[:, 2:6])
                boxes[:, [0, 2]] *= W
                boxes[:, [1, 3]] *= H
                for i, box in enumerate(boxes):
                    cls = int(targets[i][1])
                    plot_one_box(box, img, label=str(cls), color=[0, 255, 0])

            cv2.imwrite(f'fused_samples/fused_{idx}.jpg', img)

        else:
            # Treat as mid-fusion feature map
            choice = random.choice(["mean_heatmap", "pca_rgb", "channel_grid"])

            if choice == "mean_heatmap":
                fmap = x_fused.mean(axis=0)
                fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
                fmap = (fmap * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
                cv2.imwrite(f"fused_samples/fused_heatmap_{idx}.jpg", heatmap)

            elif choice == "pca_rgb":
                flat = x_fused.reshape(C, -1).T  # shape (H*W, C)
                pca = PCA(n_components=3)
                rgb = pca.fit_transform(flat).reshape(H, W, 3)
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                rgb = (rgb * 255).astype(np.uint8)
                cv2.imwrite(f"fused_samples/fused_pca_rgb_{idx}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            elif choice == "channel_grid":
                ncols = min(8, C)
                nrows = (C + ncols - 1) // ncols
                fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
                for i in range(C):
                    ax = axs[i // ncols, i % ncols]
                    ax.imshow(x_fused[i], cmap='viridis')
                    ax.axis('off')
                for j in range(C, nrows * ncols):
                    axs[j // ncols, j % ncols].axis('off')
                plt.tight_layout()
                plt.savefig(f"fused_samples/fused_grid_{idx}.png")
                plt.close()


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location, weights_only=False)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
    
    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    
    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble