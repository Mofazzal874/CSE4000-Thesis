Thesis Title : Hierarchical Body-Part Attention Enhanced YOLO for Human Detection in Aerial Imagery

- I am trying to improve the current work by testing and trying different things with the YOLO model. 
I have tried to replace the c2psa block with CBAM and ECA attention blocks.Also tried to replace the P2Head and other things.(you can see the 01-02-2026- ablation study folder for more information) I have also tried to replace the neck and backbone with Mamba SSM blocks.

- The results are not very significant, but I am trying to find ways to improve the performance of the model. I am also trying to find ways to make the model faster so that it can work with real-time applications. 


I am running the everything on kaggle free tier. 
it has these specs : 
Core Free Tier Specifications:
CPU: 2x Intel Xeon @ 2.0GHz.
RAM: 32 GB.
GPU: NVIDIA Tesla P100 or T4 (16GB VRAM).
TPU: TPU v3-8 (128GB high-speed memory).
Storage: 20 GB persistent storage (/kaggle/working), up to 100GB+ non-persistent temp storage.
Session Time: Up to 12 hours (session terminates after 20 mins inactivity). But after 12 hours, the session will automatically terminate, and you will need to restart it.

Weekly Limit: 30 hours for GPU, 20 hours for TPU. 
LinkedIn
LinkedIn
 +7
Additional Features:
Environment: Pre-installed libraries (TensorFlow, PyTorch, Scikit-learn, etc.). 





C2A Dataset Specifications: 
- The C2A dataset consists of 10,215 images containing over 360,000 annotated human instances in various disaster scenarios. It includes diverse human poses (bent, kneeling, lying, sitting, upright) and disaster contexts (traffic incidents, fire, flood, collapsed buildings).

Dataset Structure
The C2A dataset is organized into training, validation, and test sets, with multiple annotation formats for flexibility in use. Below is the structure of the dataset:

new_dataset3/
│
├── All labels with Pose info/
│   └── [YOLO format labels with pose information for all images]
│
├── test/
│   ├── images/
│   │   └── [Test image files]
│   ├── labels/
│   │   └── [YOLO format label files for test images]
│   └── test_annotations.json  [COCO format annotations for test set]
│
├── train/
│   ├── images/
│   │   └── [Training image files]
│   ├── labels/
│   │   └── [YOLO format label files for training images]
│   └── train_annotations.json  [COCO format annotations for training set]
│
└── val/
    ├── images/
    │   └── [Validation image files]
    ├── labels/
    │   └── [YOLO format label files for validation images]
    └── val_annotations.json  [COCO format annotations for validation set]

File Descriptions
Image Files:

Located in the images subfolder of each set (train, val, test)
Format: [Specify image format, e.g., JPG, PNG]
YOLO Format Annotations:

Located in the labels subfolder of each set
Format: Text files (.txt) with one line per object
Each line contains: class x_center y_center width height
Coordinates are normalized to [0, 1]
COCO Format Annotations:

JSON files: train_annotations.json, val_annotations.json, test_annotations.json
Contains detailed information about images and annotations in COCO format
Pose Information Labels:

Located in All labels with Pose info folder
YOLO format with an additional value for pose
Format: class x_center y_center width height pose
Pose values:
0: Bent
1: Kneeling
2: Lying
3: Sitting
4: Upright
Usage Notes
The dataset is split into training, validation, and test sets.
Two annotation formats are provided for flexibility: YOLO and COCO.
Pose information is available in a separate folder for all images.
Users can choose between standard object detection (YOLO/COCO) or pose-aware detection (All labels with Pose info).
Key Features
Synthetic dataset combining real disaster scenes with human poses
Over 360,000 annotated human instances
5 human pose categories
4 disaster scenario types
Image resolutions ranging from 123x152 to 5184x3456 pixels
Designed to improve human detection in complex disaster environments




- I have previously trained a model on this dataset using the YOLO format labels.

This is the discussion : https://claude.ai/share/ebf752e4-fc1c-48bf-9a24-fd71bca1b26a

