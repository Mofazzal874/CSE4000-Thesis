Paper Name:
Deep Learning for Human Detection from Aerial Images in Search and Rescue Missions

---
Overview:
This paper addresses the critical challenge of detecting humans from aerial imagery during Search and Rescue (SAR) operations in the context of natural disasters. The research focuses on improving detection performance under varying lighting conditions—a significant limitation in existing models and datasets. The authors augmented the SARD (SAR Dataset) to include diverse lighting scenarios and evaluated six YOLO model variations to identify the most effective solution for real-world SAR missions.
---

**What They've Worked On:**
The research team worked on several key areas:

**Dataset Enhancement:** Augmented the original SARD dataset (1,981 images) to create a new dataset of 4,000 images with varied lighting conditions through brightness and contrast adjustments
**Model Comparison:** Fine-tuned and evaluated six YOLO model variations:

YOLOv8: nano (v8n), small (v8s), medium (v8m)
YOLOv10: nano (v10n), small (v10s), medium (v10m)

**Robustness Testing:** Specifically addressed the challenge of human detection under different lighting scenarios (daytime, low light, varying weather conditions)
**Performance Benchmarking:** Compared their results against previous research using YOLOv4 and YOLOv5 models on the original SARD dataset

Methodology:
Data Collection:

Base dataset: SARD with 1,981 manually annotated images from video frames
Images depicted people in various distress states (exhaustion, injury) and activities (sitting, walking)
Captured in wilderness contexts (forests, grassy areas)

Data Augmentation:

Brightness Adjustment: Simulated different times of day and weather conditions
Contrast Adjustment: Enhanced/diminished differences between objects and backgrounds
Final Dataset: 4,000 images maintaining original diversity plus lighting variations

Dataset Partitioning:

Training: 3,000 images (75%)
Validation: 600 images (15%)
Testing: 400 images (10%)

Experimental Setup:

Hardware: 12th Gen Intel Core i7-12700F processor, NVIDIA GeForce RTX 3060
Software: Python 3.12.6, Windows 11 Enterprise
Framework: YOLO models from Ultralytics GitHub repository
Training: Tested with 100, 125, and 150 epochs using default hyperparameters

Performance Metrics:

Accuracy, Precision, Recall, F1 Score, mAP (mean Average Precision)
Confusion Matrix analysis: True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)


Results:
YOLOv8 Models Performance:

V8n: 80.6% accuracy, 86.0% recall, 92.8% precision, 89.3% F1, 90.8% mAP
V8s: 84.9% accuracy, 91.1% recall, 92.6% precision, 91.9% F1, 94.7% mAP
V8m: 88.5% accuracy, 91.6% recall, 96.3% precision, 93.9% F1, 95.2% mAP

YOLOv10 Models Performance:

V10n: 81.8% accuracy, 88.3% recall, 91.7% precision, 90.0% F1, 94.1% mAP
V10s: 88.7% accuracy, 92.9% recall, 95.2% precision, 94.0% F1, 96.4% mAP
V10m: 90.7% accuracy, 93.4% recall, 96.9% precision, 95.1% F1, 97.2% mAP (Best Overall)


Key Findings:

Best Model: YOLOv10m demonstrated superior performance across all metrics, offering the best balance for SAR missions
Comparison with Previous Research:

YOLOv4 on original SARD: 96.0% mAP
YOLOv5s on original SARD: 93.3% mAP
YOLOv5m on original SARD: 76.2% mAP
Their YOLOv10m on original SARD: 96.9% mAP
Their YOLOv10m on augmented SARD: 97.2% mAP


Lighting Robustness: The augmented dataset significantly improved model generalization across different lighting conditions
Practical Implications:

High recall (93.4%) minimizes missed detections (critical for saving lives)
High precision (96.9%) optimizes resource deployment by reducing false alarms
Superior adaptability to unpredictable real-world lighting scenarios


Training Validation: Loss functions (box_loss, cls_loss, dfl_loss) showed consistent decrease during training, confirming model convergence and reliability


Recommendation: The authors recommend deploying YOLOv10m for SAR missions due to its exceptional performance metrics and robustness under varying lighting conditions.