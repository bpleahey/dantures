import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, weighted_average_boxes, \
    weighted_nms_with_time
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False,
         enable_label_remap=False,
         ir_weights=None,
         late_fusion_type='NMS',
         ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        model_ir = None
        if ir_weights:
            model_ir = attempt_load(ir_weights, map_location=device)
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if hasattr(model, 'fusion_type'):
            tracing_safe = model.fusion_type not in ['early', 'mid', 'late']
        else:
            tracing_safe =   True
        if trace and tracing_safe:
            model = TracedModel(model, device, imgsz)
            if model_ir:
                model_ir = TracedModel(model_ir, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
        if model_ir:
            model_ir.half()

    # Configure
    model.eval()

    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check

    fusion_type = data.get('fusion_type', 'none')
    is_fusion = fusion_type in ['early', 'mid', 'late']

    if is_fusion:
        for m in model.modules():
            if hasattr(m, "eval_mode_fast"):
                m.eval_mode_fast = True
    
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            if is_fusion:
                if fusion_type in ['early', 'mid']:
                    dummy_rgb = torch.zeros(1, 3, imgsz, imgsz).to(device)
                    dummy_ir = torch.zeros(1, 3, imgsz, imgsz).to(device)
                    dummy_time = torch.zeros(1, dtype=torch.long).to(device)

                    # Match model's precision
                    dummy_rgb = dummy_rgb.type_as(next(model.parameters()))
                    dummy_ir = dummy_ir.type_as(next(model.parameters()))

                    model((dummy_rgb, dummy_ir, dummy_time))
                elif fusion_type == 'late':
                    dummy_rgb = torch.zeros(1, 3, imgsz, imgsz).to(device)
                    dummy_ir = torch.zeros(1, 3, imgsz, imgsz).to(device)
                    dummy_time = torch.zeros(1, dtype=torch.long).to(device)

                    # Match model's precision
                    dummy_rgb = dummy_rgb.type_as(next(model.parameters()))
                    dummy_ir = dummy_ir.type_as(next(model.parameters()))

                    model(dummy_rgb)
                    model_ir(dummy_ir)
                else:
                    raise ValueError(f"Unknown fusion type: {fusion_type}")
            else:
                dummy = torch.zeros(1, 3, imgsz, imgsz).to(device)
                # Match model's precision
                dummy = dummy.type_as(next(model.parameters()))
                model(dummy)
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '), fusion_type=fusion_type)[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        if is_fusion:
            rgb_img, ir_img, time_idx = img
            rgb_img = rgb_img.to(device, non_blocking=True)
            ir_img = ir_img.to(device, non_blocking=True)
            time_idx = time_idx.to(device, non_blocking=True) if isinstance(time_idx, torch.Tensor) else time_idx

            rgb_img = rgb_img.half() if half else rgb_img.float()  # uint8 to fp16/32
            ir_img = ir_img.half() if half else ir_img.float()  # uint8 to fp16/32
            rgb_img /= 255.0  # 0 - 255 to 0.0 - 1.0
            ir_img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = (rgb_img, ir_img, time_idx)
        else:
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

        
        targets = targets.to(device)

        if enable_label_remap and targets.shape[0]:
            your_to_coco_list = [2, 7, 0]
            your_to_coco_tensor = torch.tensor(your_to_coco_list, device=targets.device)
            targets[:, 1] = your_to_coco_tensor[targets[:, 1].long()]
        
        if is_fusion:
            nb = img[0].shape[0]  # batch size
            _, _, height, width = img[0].shape  # batch size, channels, height, width
        else:
             nb, _, height, width = img.shape  # batch size, channels, height, width


        with torch.no_grad():
            out_ir = None
            time_idx = None
            # Run model
            t = time_synchronized()

            if isinstance(img, (tuple, list)):
                if is_fusion:
                    if fusion_type in ['early', 'mid']:
                        (rgb_img, ir_img, time_idx) = img
                        out, train_out = model((rgb_img, ir_img, time_idx), targets=targets)  # inference and training outputs
                    elif fusion_type == 'late':
                        (rgb_img, ir_img, time_idx) = img
                        out, _ = model(rgb_img, augment=augment)  # RGB inference outputs (late fusion never needs training metrics)
                        out_ir, _ = model_ir(ir_img, augment=augment)  # IR inference outputs (late fusion never needs training metrics)
                    else:
                        raise ValueError(f"Unknown fusion type: {fusion_type}")
                else:
                    raise ValueError("Model missing fusion_type attribute but input is a tuple.")
            else:
                out, train_out = model(img, augment=augment)
            
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            if out_ir is not None:
                if late_fusion_type == 'NMS':
                    out = torch.cat((out, out_ir), dim=1) if out is not None else out_ir
                    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                elif late_fusion_type == 'NMS+Weight':
                # TODO: implement modified NMS that takes in time idxs and operates on final boxes rather than raw boxes
                    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                    out_ir = non_max_suppression(out_ir, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

                    out_combined = []
                    if time_idx is None:
                        raise ValueError("time_idx is None, but late fusion type is NMS+Weight")
                    
                    for det_vis, det_ir, ti in zip(out, out_ir, time_idx):
                        fused = weighted_nms_with_time(det_vis, det_ir, ti.item() if isinstance(ti, torch.Tensor) else ti, iou_thres=iou_thres)
                        out_combined.append(fused)

                    out = out_combined
                elif late_fusion_type == 'AVG':
                    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                    out_ir = non_max_suppression(out_ir, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                    out_combined = []
                    for det_vis, det_ir in zip(out, out_ir):
                        fused = weighted_average_boxes(det_vis, det_ir, iou_threshold=iou_thres)
                        out_combined.append(fused)
                    
                    out = out_combined
                elif late_fusion_type == 'AVG+Weight':
                    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                    out_ir = non_max_suppression(out_ir, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
                    out_combined = []
                    if time_idx is None:
                        raise ValueError("time_idx is None, but late fusion type is AVG+Weight")
                    for det_vis, det_ir, ti in zip(out, out_ir, time_idx):
                        # Example fixed weights
                        if ti == 0:  # noon
                            w_rgb, w_ir = 0.8, 0.2
                        elif ti == 1:  # twilight
                            w_rgb, w_ir = 0.5, 0.5
                        elif ti == 2:  # night
                            w_rgb, w_ir = 0.2, 0.8
                        else:
                            w_rgb, w_ir = 0.5, 0.5  # fallback
                        
                        fused = weighted_average_boxes(det_vis, det_ir, weight_rgb=w_rgb, weight_ir=w_ir, iou_threshold=iou_thres)
                        out_combined.append(fused)
                    
                    out = out_combined
                else:
                    out = torch.cat((out, out_ir), dim=1) if out is not None else out_ir
                    print("unknown late fusion type, using NMS")
                    out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            else: 
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

            t1 += time_synchronized() - t

            if enable_label_remap:
                # wanted_coco_classes = [2, 7, 0]  # car, truck, person
                wanted_coco_classes = [2] # suppress to only car with bad dataset

                for i in range(len(out)):
                    if out[i] is not None and len(out[i]):
                        out[i] = out[i][torch.isin(out[i][:, 5].long(), torch.tensor(wanted_coco_classes, device=out[i].device))]

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            if is_fusion:
                scale_coords(rgb_img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            else:
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

            # Append to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging - Media Panel Plots
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                    if is_fusion:
                        wandb_images.append(wandb_logger.wandb.Image(rgb_img[si], boxes=boxes, caption=path.name))
                    else:
                        wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                if is_fusion:
                    scale_coords(rgb_img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                else:
                    scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            # predictions:
            if is_fusion:
                f_rgb = save_dir / f'test_batch{batch_i}_rgb_labels.jpg'
                f_ir = save_dir / f'test_batch{batch_i}_lwir_labels.jpg'
                Thread(target=plot_images, args=(rgb_img, targets.cpu(), paths, f_rgb, names), daemon=True).start()
                Thread(target=plot_images, args=(ir_img, targets.cpu(), paths, f_ir, names), daemon=True).start()

                f_rgb = save_dir / f'test_batch{batch_i}_rgb_pred.jpg'
                f_ir = save_dir / f'test_batch{batch_i}_lwir_pred.jpg'
                Thread(target=plot_images, args=(rgb_img, output_to_target(out), paths, f_rgb, names), daemon=True).start()
                Thread(target=plot_images, args=(ir_img, output_to_target(out), paths, f_ir, names), daemon=True).start()
            else:
                print("not reading as fusion in this thread")
                f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
                Thread(target=plot_images, args=(img, targets.cpu(), paths, f, names), daemon=True).start()
                f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
                Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--enable-label-remap', action='store_true', help='enable label remapping for using default coco labels with your dataset')
    parser.add_argument('--ir-weights', type=str, help='path to IR model weights for late fusion')
    parser.add_argument('--late-fusion-method', type=str, default='NMS', help='late fusion type (NMS, NMS+Weight, AVG, AVG+Weight)')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             enable_label_remap=opt.enable_label_remap,
             ir_weights=opt.ir_weights,
             late_fusion_type=opt.late_fusion_method,
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
