#!/usr/bin/env python
# coding: utf-8

# # Deep Learning for Brain-Signal Decoding from Electroencephalography (EEG)

# Machine learning, particularly deep learning, has the potential to improve medical applications by processing large amounts of data and extracting information that may not be easy to extract for medical doctors. In particular, brain-signal decoding from electroencephalographic (EEG) recordings is a promising area for machine learning due to the large amount of information contained in these signals and the difficulty of interpreting them manually. Deep neural networks (DNNs) have been successful at a variety of tasks, including object detection and speech recognition, and may be well-suited for decoding EEG signals due to their ability to handle high-dimensional hierarchical natural signals. However, prior to the work in this thesis, it was still unclear how well DNNs perform on EEG decoding compared to hand-engineered, feature-based approaches, and more research was needed to determine the optimal approaches for using deep learning in this context. This thesis describes constructing and training EEG-decoding deep learning networks that perform as well as feature-based approaches and developing visualizations that suggest they extract physiologically meaningful features.

# | Chapter    | Summary    |
# | :--- | ---: |
# | **Introduction**|
# | [Introduction](introduction)   |  Deep learning on EEG is a very promising approach for brain-signal-based medical applications like automatic diagnosis  |
# | [Prior Work](prior-work)   | Prior to 2017, research did not clearly show how competitive deep learning is compared with well-optimized feature baselines   |
# | **Methods**|
# | [Neural Network Architectures for EEG-Decoding](network-architectures)   | Starting from a network mimicking an established EEG decoding pipeline, progressively more generic neural network architectures for EEG decoding |
# | [Cropped Training](cropped-training)   | A training strategy to use many sliding windows in a computationally efficient manner |
# | [Perturbation Visualization](perturbation-visualization)   | A visualization of how frequency features affect the trained network and its limitations |
# | **Results**|
# | [Movement-Related Decoding](movement-related)   | Deep learning can be at least as good as feature-based baselines for movement-related decoding; deep networks also learn to extract known hand-engineered features|
# | [Task-Related Decoding](task-related)   | Deep learning also performs well at other task-related decoding|
# | [Decoding Pathology](pathology)  | Deep networks designed for task-related decoding can also decode pathology well  |
# | [Invertible Networks](invertible-networks) | Better Understanding and larger data through invertible networks |
# | [Future Work](future-work)   | Newer DL architectures such as transformers or invertible networks may allow better performance and better interpretability   |

# Possible:
# 
# | Chapter    | Summary    |
# | :--- | ---: |
# | [Back to Features - a Comparison](feature-comparison)   | Well-optimized feature baselines remain competitive with deep learning    |
# | [Further Deep Learning Improvements](dl-improvements)   | Deep learning improvements from other domains partially carry over to EEG decoding    ?|
# 
# * [Introduction](introduction)
# * [Prior Work](prior-work)
# * [Neural Network Architectures for EEG-Decoding](network-architectures)
# * [Cropped Training](cropped-training)
# * [Movement-Related Decoding](movement-related)
# * [Decoding Pathology](pathology)
# * [Further Deep Learning Improvements](dl-improvements)
# * [Back to Features - a Comparison](feature-comparison)
# * [Invertible Networks](invertible-networks)
# * [Further Interpretability](deep-interpretability)
# * [Future Work](future-work)
