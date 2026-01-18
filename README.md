# Final_Project

## Introduction

This paper develops a visual explanation system using CAM and Grad-CAM to interpret convolutional neural network decisions. We implement these methods to reveal the specific image regions driving model predictions, starting with image classification and extending the application to object detection.

## Guidance

To run the project, please first clone the project

```bash
git clone [https://github.com/Babu-luo/Understanding_Deep_Convolutional_Nerual_Networks.git](https://github.com/Babu-luo/Understanding_Deep_Convolutional_Nerual_Networks.git)
cd Understanding_Deep_Convolutional_Nerual_Networks
```

then install the required packages

```bash
pip install -r requirements.txt
```

The checkpoints of the resnet-18 model are provided in the `resnet_best.pth` file, and that of the faster-rcnn model are not provided in this repo because it's too large! You can  download it from [https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth), and put it in the default path of torch models 'C:/Users/YourName/.cache/torch/hub/checkpoints'.
To get heatmaps, please run your command with the following arguments:

```bash
--mode test --save_path ./checkpoints/resnet_best.pth --vis cam --vis_num 10
```

*vis* refer to the visualization method, which can be either *cam* or *gradcam*. Your can also choose *none* to simply get the accuracy of the model in test set.
*vis_num* refer to the number of heatmaps you want to obtain.

To run the deletion and insertion evaluation, please run your command with the following arguments:

```bash
--mode test --save_path ./checkpoints/resnet_best.pth --vis cam --hidden_strategy zero
```

After running this command, you will get the deletion and insertion curves of the model which uses *cam* to get heatmaps and *zero* strategy to hide the image regions.
*vis* can determine the visualization method, which can be either *cam* or *gradcam*.
*hidden_strategy* refer to the strategy to hide the image regions, which can be either *zero*, *gray* or *mean*.

For Object Detection task using faster-rcnn model, please run your command with the following arguments:

```bash
--mode test --det_vis --det_img test.png
```

*det_vis* refer to the flag to visualize the detection results.
*det_img* refer to the image to be visualized. If the det_img is not provided, the model will get its data from PASCAL VOC 2007 test set.

Or you can run Faster-RCNN on the PASCAL VOC dataset via setting  det_img as blank.

```bash
--mode test --det_vis
```

Please download PASCAL VOC testset and put it to ./data before running this command. You can download it via [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar).
