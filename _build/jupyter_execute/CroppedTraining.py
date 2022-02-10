#!/usr/bin/env python
# coding: utf-8

# (cropped-training)=
# # Cropped Training

# In this chapter, we describe a specific "cropped" training strategy that regularizes the networks by training on many sliding temporal windows within the data. This is meant to squeeze out more performance from deep networks on EEG, as the performance of deep networks often scales well with more training data [ref] and EEG datasets are often rather small. We show how to use a cropped training strategy, similarly used in computer vision by training on crops of the images, on EEG data. First, we will describe regular non-cropped training, then cropped training on a conceptual level and finally how to make cropped training computationally faster. 

# ## Non-Cropped/Trialwise Training

# In trialwise EEG training, deep networks are trained using EEG signals of entire trials and their corresponding labels as examples. With typical sizes of EEG datasets, networks may therefore be trained on ~100-1000 examples per subject. This is much less in computer vision, where networks are typically trained on tens of thousands or even millions of images. 
# 

# ![title](images/trialwise_explanation.png)

# ```{figure} images/trialwise_explanation.png
# ---
# name: trialwise-figure
# ---
# Trialwise training example. An entire single trial is fed through the network and the network's prediction is compared to the target to train the network.
# ```

# ## Cropped Training
# 
# Cropped training increases the number of training examples by training on many crops, i.e., temporal windows, within the trial. For example, for a 4-second trial, one may create all possible 2-second windows inside the trial and use these as "independent" examples. This drastically increases the number of training examples, albeit many of the examples are highly overlapping. This is an exteme version of the method to use random crops of images that is used to train deep entworks in computer vision. A naive implementation here would increase the computational cost per training epoch a lot as now there are much more examples. Thankfully, the high overlap between neighbouring crops can be exploited for a more efficient implementation.

# ![title](images/cropped_explanation.png)

# ```{figure} images/cropped_explanation.png
# ---
# name: cropped-figure
# ---
# Cropped training example. A compute window contains many temporal windows (crops) inside that are used as individual examples to train the network.
# ```

# ## Computationally Faster Cropped Training

# Cropped training can be implemented with substantially less computations by exploiting that highly overlapping crops result in highly overlapping intermediate representations. By passing a group of neighbouring crops together, we can reuse intermediate computations. See Figures XX and YY for a concrete example of this speedup method. This idea had been used in the same way for dense predictions on images, e.g. for segmentation [Giusti et al., 2013; Nasse et al., 2009; Sermanet et al., 2014; Shelhamer et al., 2016]. 

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

# Efficient cropped training then results in the exact same predictions and training as if the neighbouring crops were passed separately through the network. This is only true for networks that either use left-padding or no padding at all. In the deep and shallow network described here, we do not use any padding. In the residual network, we use padding, hence the training is not exactly identical to passing neighbouring crops separately, but we found it still improves over trial-wise training.

# The more efficient way to do cropped training introduces a new hyperparameter, the number of neighbouring crops that are decoded together. The larger this hyperparameter, the more computations are saved and the more speedup one gets (see Giusti et al. [2013] for a more detailed speedup analysis on images). Larger numbers of neighbouring crops that are trained on simultanaeously require more memory and may also affect the training dynamics due to more neighbouring crops being in the same mini-batch. However, we did not find negative effects on the training dynamics from larger number of simultaneously decoded neighbouring crops, consistent with prior work in computer vision ([Shelhamer et al., 2016]).

# In[ ]:




