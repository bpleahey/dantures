# DANTURES

Implementation of Honors Thesis - [DANTURES](https://drive.google.com/file/d/149nFAIvZFRVQ295swum8E7VPv2zPeIgZ/view?usp=drive_link)

Datasets: https://drive.google.com/drive/folders/1CIn_wIYH15_BhSsn-yjwXq3Fb9sqYGjo

Training Runs (with model weights): https://drive.google.com/drive/folders/1Hike1cY2mF2tC7Wl7zxgiahxMxVL3sXB

## Training

Training may be carried out through directly calling the train.py file, defining a Model instance and running pytorch code, or through the `train_val` notebook we have implemented. The structure for a training run in the command line may be found below:

```
!python train.py --img 640 \
--batch 16 \ # batch size
--epochs 50 \ # num epochs
--data <your_data_paths>.yaml \ # data yaml. if no other directory is specified it will start from ./data/
--cfg <model>.yaml \ # model config yaml. if not other directory is specified it will start from ./cfg/
--weights <weights>.pt \ # load weights if desired
--hyp hyp.scratch.custom.yaml \ # hyperparameter yaml. if no other directory is specified it will start from ./data/
--name <your_name> # run name for logging purposes

```

### Suggested Preprocessing

Current preprocessing may be found in our jupyter notebooks. We are currently migrating this to helper files and will update this description accordingly.

## Testing

Similar to training, testing may be carried out through directly calling the test.py file, using a PyTorch model you have trained yourself, or through the `evaluate` notebook we have implemented. The structure for a evaluating a model in the command line may be found below:

```
!python test.py --img 640 \ # image size
    --batch 16 \ # batch size
    --data <eval_x>.yaml \ # data yaml. if no other directory is specified it will start from ./data/
    --device 0 \ # specify GPU (default is 0)
    --weights <weights>.pt \  # load weights if desired
    --conf 0.01 \ # bounding box confidence threshold
    --iou 0.65 \ # iou threshold for considering overlapping prediction/ground truth boxes the same
    --name <your_name> \ # run name for logging purposes
    # --ir-weights <ir_best>.pt \ # if testing for late fusion, specify your ir model weights
    # --late-fusion-method <method> # optionally for late fusion, specify which method you are using
```

### Elevation Ablation Testing

Our test dataset contains elevation information in addition to time-of-day information. Within `evaluate.ipynb`, we include code to split the dataset on each time of day. By modifying your data yaml, you may then assess how a model performs at each elevation.

## Inference

Inference does not yet support late or mid fusion but will be implemented shortly.

## Sample Fusion Outputs:

(more may be found at https://drive.google.com/drive/folders/1cHKFAZkAxPivKjn2tGZdk5yV4OmkACOg?usp=drive_link)

Early Fusion Images:

![fused_421673](https://github.com/user-attachments/assets/012a1ddd-8808-4c67-b8ab-67cce5405dad)
![fused_441312](https://github.com/user-attachments/assets/26e1d8a7-63e0-45c6-b57e-8ae59582ede7)
![fused_613786](https://github.com/user-attachments/assets/0d180fd1-ac8e-4bad-8711-1bfe0fdae621)
![fused_661374](https://github.com/user-attachments/assets/5b1a9e9d-ac27-4fd1-82ee-dcfd432bd81e)

Middle Fusion Heatmaps:

![fused_heatmap_405785](https://github.com/user-attachments/assets/a4fdfd4c-9329-4b82-a922-5922a4d1c18a)
![fused_heatmap_80971](https://github.com/user-attachments/assets/d2309637-2d65-481d-9ad7-6b6ad3f31a44)
![fused_heatmap_247110](https://github.com/user-attachments/assets/b3114d30-1496-4c6c-ae47-3b88820c1f0d)
![fused_heatmap_168766](https://github.com/user-attachments/assets/c76843ec-ddd3-4341-b5fb-2c7b07f10863)

Middle Fusion Layer Grids:

![fused_grid_139671](https://github.com/user-attachments/assets/d96b8ad7-a2e2-4ed0-ba64-f00efc5172cf)
![fused_grid_174795](https://github.com/user-attachments/assets/7b4f1655-b105-4c03-8e50-4c3a0682d6ca)
![fused_grid_47164](https://github.com/user-attachments/assets/76ad4165-d3c1-4bad-81b6-f74471f13542)
![fused_grid_124878](https://github.com/user-attachments/assets/8ea8d83d-f1ae-4d8c-b74c-bf0cdce583d4)

Middle Fusion PCA-reduced Features:

![fused_pca_rgb_741893](https://github.com/user-attachments/assets/63b74b50-27a6-40ca-bfc9-a7095cfd2538)
![fused_pca_rgb_21863](https://github.com/user-attachments/assets/1e5685fa-cd48-436a-ba3b-ffd5c8c2e1c2)
![fused_pca_rgb_758054](https://github.com/user-attachments/assets/ebc8124c-87b4-4dbf-97b0-f9275fcb24d8)
![fused_pca_rgb_672594](https://github.com/user-attachments/assets/4af29f73-2f61-485c-8d02-aa4988201efc)

