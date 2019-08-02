# ICE VISION CHALLANGE
# Traffic Sign Detection and Classification

## Evaluation

### what is the metric? 
Based on metric type, the training strategy needs adaptation.
1.  >Bounding boxes with an area smaller than 100 pixels are ignored during evaluation. 
    
2.  >Detection is considered successful if IoU is bigger or equal to 0.5 and bounding box has a correct class code. 

3.  >If sign is detected twice, then detection with a smallest IoU will be counted as a false positive. 

4.  >**Each false positive or incorrect detection results in penalty equal to 2 points**.

5.  >If IoU of a true positive detection is bigger than 0.85, it results in adding 1 point to final result. Otherwise points are calculated as `((IoU - 0.5)/0.35)^0.25`.

6.  >The final score is computed as sum of all true positive points minus all penalties.

    The metric mainly focus on accuracy. Dataset is uneven, we need metrics like AUC(PR), ROC, F1 and mAP.
    

### Metrics
1. Precision and Recall: 
    [Article](https://towardsdatascience.com/precision-vs-recall-386cf9f89488) 
    and 
    [Wiki Page](https://en.wikipedia.org/wiki/Precision_and_recall).
2. ROC: [Paper](https://www.biostat.wisc.edu/~page/rocpr.pdf) and [Discussion](https://stats.stackexchange.com/q/7210s)
3. AUC: [Disscusion1](https://www.kaggle.com/general/7517) and [Disscusion2](https://stats.stackexchange.com/q/123370). 
4. mAP: [Article](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173).

# Point Cloud Detection and Classification

## Insight
* Sensor: Velodyne **64E** S3 LiDAR
* Data Structure: A sequence containing **10-20** seconds of data
* Classes: **Car**, **Bus**, **Truck**, **Pedestrian** and Other

## Datasets

### The [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset

A widely used computer vision benchmark which was released in 2012. A Volkswagen station was fitted with grayscale and color cameras, a Velodyne 3D Laser Scanner and a GPS/IMU system. They have datasets for various scenarios like urban, [residential](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential), [highway](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=road), and [campus](http://www.cvlibs.net/datasets/kitti/raw_data.php?type=campus). This dataset has been used in papers for object detection such as [VoxelNet](https://arxiv.org/abs/1711.06396), [MV3D](https://arxiv.org/abs/1611.07759), and [Vote3Deep](https://arxiv.org/abs/1609.06666). \
Data Formats of different sensors:
* 3D Point Cloud Data (Velodyne HDL-64E @ 10Hz spin)
* .bin files (in the velodyne_points folder) – [Code](https://gist.github.com/prerakmody/110f80c3e1d99ac100c481d6428e3c75#file-visualize_lidar-py) to parse and visualize data
* 2D Image Data (2 × PointGray Flea2 grayscale cameras+ 2 × PointGray Flea2 color cameras)
* .png (in the image_00/01/02/03 folder)
* One can use [this](https://github.com/utiasSTARS/pykitti) repo to browse through the data.

* **Label Description** from NVIDIA [DIGITS](https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md#label-format).

### Honda 3D Dataset
The [H3D](https://usa.honda-ri.com/H3D) is a large scale full-surround 3D multi-object detection and tracking dataset. It is gathered from HDD dataset, a large scale naturalistic driving dataset collected in San Francisco Bay Area. H3D consists of following features:
* Full 360 degree LiDAR dataset (dense pointcloud from Velodyne-64)
* 160 crowded and highly interactive traffic scenes.
* 1,071,302 3D bounding box labels.
* 8 common classes of traffic participants (Manually annotated every 2Hz and linearly propagated for 10 Hz data).
* Benchmarked on state-of-the art algorithms for 3D only detection and tracking algorithms.

### Sydney Urban Objects Dataset
This [DATASET](http://www.acfr.usyd.edu.au/papers/SydneyUrbanObjectsDataset.shtml) contains a variety of common urban road objects scanned with a Velodyne HDL-64E LIDAR, collected in the CBD of Sydney, Australia. There are 631 individual scans of objects across classes of vehicles (including 88 instances of cars, 16 buses and 12 trucks), pedestrians, signs and trees.
## Benchmarks
* [KITTI Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
* TBD.

## Giants
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN).
* [Multi-task Multi-Sensor Fusion for 3D Object Detection](https://eng.uber.com/research/multi-task-multi-%20sensor-fusion-for-3d-object-detection/).
* [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion
Forecasting with a Single Convolutional Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf)