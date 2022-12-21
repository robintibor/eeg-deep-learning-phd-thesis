#!/usr/bin/env python
# coding: utf-8

# # Deep Learning for Brain-Signal Decoding from Electroencephalography

# ## Abstract

# Machine learning has the potential to improve medical applications by processing larger amounts of data and extracting different information from it than medical doctors. In particular, brain-signal decoding from electroencephalographic (EEG) recordings is a promising area for machine learning due to the relative ease of acquiring large amounts of EEG recordings and the difficulty of interpreting them manually. Deep neural networks (DNNs) have been successful at a variety of natural-signal decoding tasks like object recognition from images or speech recognition from audio and thus may be well-suited for decoding EEG signals. However, prior to the work in this thesis, it was still unclear how well DNNs perform on EEG decoding compared to hand-engineered, feature-based approaches, and more research was needed to determine the optimal approaches for using deep learning in this context. This thesis describes constructing and training EEG-decoding deep learning networks that perform as well as feature-based approaches and developing visualizations that suggest they extract physiologically meaningful features.

# ## Contents

# | Chapter    | Summary    |
# | :--- | ---: |
# | **Introduction and Background**|
# | [Introduction](introduction)   |  Deep learning on EEG is a very promising approach for brain-signal-based medical applications  |
# | [Prior Work](prior-work)   | Prior to 2017, research did not clearly show how competitive deep learning is compared with well-optimized feature baselines   |
# | [Filterbank Common Spatial Patterns and Filterbank Network](fbscp-and-filterbank-net)   | Filter Bank Common Spatial Patterns was as an inspiration for initial network architectures|
# | **Methods**|
# | [Neural Network Architectures for EEG-Decoding](network-architectures)   | Progressively more generic neural network architectures for EEG decoding were created |
# | [Cropped Training](cropped-training)   | A training strategy to use many sliding windows was implemented in a computationally efficient manner |
# | [Perturbation Visualization](perturbation-visualization)   | A visualization of how frequency features affect the trained network was developed |
# | **Applications and Results**|
# | [Movement-Related Decoding](movement-related)   | Deep learning can be at least as good as feature-based baselines for movement-related decoding; deep networks also learn to extract known hand-engineered features|
# | [Generalization to Other Tasks](task-related)   | Our deep networks generalize well to decoding other decoding tasks|
# | [Decoding Pathology](pathology)  | Deep networks designed for task-related decoding can also decode pathology well  |
# | [Invertible Networks](invertible-networks) | Better Understanding and larger data through invertible networks |
# | [Future Work](future-work)   | Newer DL architectures such as transformers may allow better performance and better interpretability   |

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
