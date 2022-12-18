#!/usr/bin/env python
# coding: utf-8

# (cropped-training)=
# # Cropped Training

# In this chapter, we describe a training strategy called "cropped training" for regularizing deep networks on EEG data. The goal of this strategy is to improve the performance of deep networks on the often relatively small EEG datasets by training them on many sliding temporal windows within the data. This approach had been similarly used as spatial cropping in computer vision, where networks are trained on multiple cropped versions of images. We first describe the concept of regular, non-cropped training and then introduce cropped training on a conceptual level. Finally, we discuss how to implement this approach efficiently. Our aim is to demonstrate the effectiveness and computational efficiency of cropped training as a regularization technique for deep networks on EEG data.

# ## Non-Cropped/Trialwise Training

# ![title](images/trialwise_explanation.png)

# ```{figure} images/trialwise_explanation.png
# ---
# name: trialwise-figure
# ---
# Trialwise training example. An entire single trial is fed through the network and the network's prediction is compared to the trial target to train the network.
# ```

# In the trialwise training of deep networks on EEG data, each example consists of the EEG signals from a single trial and its corresponding label. Due to the typically small size of EEG datasets, networks trained in this way may only be trained on a few hundred to a few thousand examples per subject. This is significantly fewer examples than those used to train networks in computer vision, where tens of thousands or even millions of images are commonly used.
# 

# ## Cropped Training

# ![title](images/cropped_explanation.png)

# ```{figure} images/cropped_explanation.png
# ---
# name: cropped-figure
# ---
# Cropped training example. A compute window contains many temporal windows (crops) inside that are used as individual examples to train the network.
# ```

# Cropped training increases the number of training examples by training on many crops, i.e., temporal windows, within the trial.  For example, in a 4-second trial, all possible 2-second windows within the trial could be used as "independent" examples. This  approach drastically increases the number of training examples, although many of the examples are highly overlapping. This can be seen as an extreme version of the method to use random crops of images that is used to train deep networks in computer vision. However, a naive implementation of cropped training would greatly increase the computational cost per epoch due to the highly increased number of examples. Thankfully, the high overlap between neighbouring crops can be exploited for a more efficient implementation.

# ## Computationally Faster Cropped Training

# ![title](images/Multiple_Prediction_Matplotlib_Graphics.ipynb.2.png)

# ```{figure} images/Multiple_Prediction_Matplotlib_Graphics.ipynb.2.png
# ---
# name: cropped-naive-computation-figure
# ---
# Naive cropped training toy example. Each possible length-5 crop is taken from the original length-7 trial and independently processed by the Conv-Conv-Linear projection network. All filter values of the network are assumed to be ones. Each crop is processed independently. The values in red are identical and unnecessarily computed independently for each crop.
# ```

# ![title](images/Multiple_Prediction_Matplotlib_Graphics.ipynb.3.png)

# ```{figure} images/Multiple_Prediction_Matplotlib_Graphics.ipynb.3.png
# ---
# name: cropped-efficient-computation-figure
# width: 50%
# ---
# Efficient cropped training. 
# ```

# Cropped training can be implemented with substantially less computations by exploiting that highly overlapping crops result in highly overlapping intermediate network activations. By passing a group of neighbouring crops together to the network, we can reuse intermediate computations. See {numref}`cropped-naive-computation-figure` and  {numref}`cropped-efficient-computation-figure` for a concrete example of this speedup method. This idea had been used in the same way for dense predictions on images, e.g. for segmentation {cite}`giusti_fast_2013,nasse_face_2009,sermanet_overfeat:_2013,shelhamer_fully_2016`.

# Efficient cropped training then results in the exact same predictions and training as if the neighbouring crops were passed separately through the network. This is only true for networks that either use left-padding or no padding at all to the input and the intermediate activations. In the deep and shallow network described here, we do not use any padding. In the residual network, we use padding, hence the training is not exactly identical to passing neighbouring crops separately, but we still found it to improve over trial-wise training.

# The more efficient way to do cropped training introduces a new hyperparameter, the number of neighbouring crops that are decoded together. The larger this hyperparameter, the more computations are saved and the more speedup one gets (see {cite}`giusti_fast_2013` for a more detailed speedup analysis on images). Larger numbers of neighbouring crops that are trained on simultanaeously require more memory and may also affect the training dynamics due to more neighbouring crops being in the same mini-batch. However, we did not find negative effects on the training dynamics from larger number of simultaneously decoded neighbouring crops, consistent with prior work in computer vision {cite}`shelhamer_fully_2016`.
