import numpy as np
import random
import torch
import torch.nn as nn
import cv2
import os

from models.common import Conv, DWConv
from utils.google_utils import attempt_download

#TODO: complete and test symmetriccrossattention block
class SymmetricCrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query_rgb = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_ir = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_ir = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.query_ir = nn.Conv2d(channels, channels, kernel_size=1)
        self.key_rgb = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_rgb = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.gate = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: tuple (z_rgb, z_ir)
        Each of shape [batch, channels, height, width]
        """
        z_rgb, z_ir = x
        B, C, H, W = z_rgb.shape
        
        # Flatten spatially
        # z_rgb_flat = z_rgb.view(B, C, -1).permute(0, 2, 1)
        # z_ir_flat = z_ir.view(B, C, -1).permute(0, 2, 1) 

        # Project features
        q_rgb = self.query_rgb(z_rgb).view(B, C, -1).permute(0, 2, 1)
        k_ir = self.key_ir(z_ir).view(B, C, -1).permute(0, 2, 1)
        v_ir = self.value_ir(z_ir).view(B, C, -1).permute(0, 2, 1)
        
        q_ir = self.query_ir(z_ir).view(B, C, -1).permute(0, 2, 1)
        k_rgb = self.key_rgb(z_rgb).view(B, C, -1).permute(0, 2, 1)
        v_rgb = self.value_rgb(z_rgb).view(B, C, -1).permute(0, 2, 1)

        # Cross Attention: RGB attends to IR
        attn_rgb_to_ir = self.softmax(q_rgb @ k_ir.transpose(-2, -1) / (C ** 0.5))  # [B, HW, HW]
        z_rgb_prime = attn_rgb_to_ir @ v_ir 

        # Cross Attention: IR attends to RGB
        attn_ir_to_rgb = self.softmax(q_ir @ k_rgb.transpose(-2, -1) / (C ** 0.5))
        z_ir_prime = attn_ir_to_rgb @ v_rgb

        # Reshape back
        z_rgb_prime = z_rgb_prime.permute(0, 2, 1).view(B, C, H, W)
        z_ir_prime = z_ir_prime.permute(0, 2, 1).view(B, C, H, W)

        # Adaptive gating (Equation 3 in your description)
        fused = torch.sigmoid(self.gate) * z_rgb_prime + (1 - torch.sigmoid(self.gate)) * z_ir_prime
        
        return fused

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
            self.time_embedding = nn.Embedding(3, 8)  # 3 time categories
            self.predictor = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        elif mode == "manual":
            #TODO: toggle these ratios and see what works best
            self.alpha_table = torch.tensor([0.7, 0.5, 0.3])  # noon, post_sunrise_or_pre_sunset, pre_sunrise_or_post_sunset

    def forward(self, rgb, lwir, time_idx):
        lwir = lwir.repeat(1, 3, 1, 1)  # Convert grayscale LWIR to 3-channel
        if self.mode == "learned":
            time_embed = self.time_embedding(time_idx)
            alpha = self.predictor(time_embed).view(-1, 1, 1, 1)
        elif self.mode == "manual":
            alpha = self.alpha_table[time_idx].view(-1, 1, 1, 1).to(rgb.device)

        #TODO: decide if i want to be saving images or not, change threshold
        if self.training:  # Only save during training
        # Save a few examples
          if random.random() < .25:  # only 1% of batches
            save_fused_tensor(alpha * rgb + (1 - alpha) * lwir)  # your save function

        return alpha * rgb + (1 - alpha) * lwir

def save_fused_tensor(x_fused, idx=None):
    """
    Saves a fused tensor as a .jpg image.
    
    Args:
        x_fused (Tensor): Tensor of shape (C, H, W) or (B, C, H, W)
        idx (Optional[int]): An optional index for filename uniqueness
    """
    if isinstance(x_fused, torch.Tensor):
        x_fused = x_fused.detach().cpu().numpy()

    if x_fused.ndim == 4:
        x_fused = x_fused[0]  # Take first image if batch

    # (C, H, W) -> (H, W, C)
    x_fused = np.transpose(x_fused, (1, 2, 0))
    if x_fused.max() <= 1.0: # scale and normalize
        x_fused = x_fused * 255.0
    x_fused = np.clip(x_fused, 0, 255).astype(np.uint8)
    x_fused = cv2.cvtColor(x_fused, cv2.COLOR_RGB2BGR)

    # Generate a random name if no idx is given
    if idx is None:
        idx = random.randint(0, 999999)

    save_path = os.path.join('fused_samples', f'fused_{idx}.jpg')
    cv2.imwrite(save_path, x_fused)

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