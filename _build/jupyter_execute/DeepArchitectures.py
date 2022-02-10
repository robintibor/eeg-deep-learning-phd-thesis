#!/usr/bin/env python
# coding: utf-8

# (network-architectures)=
# # Neural Network Architectures for EEG-Decoding

# We developed neural network architectures for EEG decoding with a EEG-specific development strategy. We started from smaller architectures that closely mimic a feature-based EEG-decoding algorithm and later progressed to a more generic architecture. This development strategy ensured that we could be fairly confident that the initial network architectures should perform as well as the feature-based algorithm. That also allowed us to use these smaller architectures to create a robust data preprocessing pipeline. After validating that the smaller architectures perform well with this pipeline, we could proceed to develop and evaluate more generic architectures.
# 
# 
# 

# ## Filter Bank Common Spatial Patterns as a Starting Point

# We selected filter bank common spatial patterns (FBSCP) as the feature-based EEG-decoding algorithm to  mimic using neural network architectures. FBCSP is an EEG-decoding algorithm that has been successfully used in task-related EEG-decoding competitions [refs]. FBCSP aims to decode changes in the amplitude of different frequencies. These amplitude changes often happen in the EEG signal during certain tasks. The basic building block of FBCSP is the Common Spatial Patterns (CSP) algorithm. CSP aims to find a spatial filter over the EEG electrodes, such that the variance of the spatially filtered EEG signal allows distinguish two conditions. More specifically, the spatially filtered signal maximizes the ratio of the signal variance between the two conditions, e.g. of the signal during two different movements. For example, the signal of a spatial filter computed by CSP may have a very large variance during movements of the left hand and a very small variance during movements of the right hand.
# 
# 
# 

# ![title](images/Methods_Common_Spatial_Patterns_18_0.png)

# ```{figure} images/Methods_Common_Spatial_Patterns_18_0.png
# ---
# name: csp-figure
# ---
# Common Spatial Patterns example taken from a master thesis [ref]. Top parts show EEG signals for three electrodes during a left hand and  a right hand movement. Bottom parts show spatially filtered signals of two CSP filters. Green parts have lower variance and red parts have higher variance. Note that this difference is strongly amplified after CSP filtering.
# ```

# ### Common Spatial Patterns

# In EEG Decoding, Common Spatial Patterns (CSP) [ref] is used to decode brain signals that lead to a change in the amplitudes of the EEG signal with a specific spatial topography. To do that, CSP aims to maximize the ratio of the signal variance between signals of two classes. Concretely, we are given signals $X_{1}, X_{2} \in \mathbb{R}^{n x k x t}$ from $n$ EEG trials (can be different for $X_1, X_2$), $k$ EEG electrodes and $t$ timepoints within each trial. CSP then finds a spatial filter $w$ that maximize the ratio of the variances of the spatially filtered $X_1,X_2$:
# 
# $w=\arg\!\max_w\frac{Var(w^T X_1)}{Var(w^T X_2)}= \arg\!\max_w\frac{||w^T X_1||^2}{||w^T X_2||^2}=\arg\!\max_w\frac{w^T X_1  X_1^T w}{w^T X_2  X_2^T w}$
# 

# Rather than just finding a single spatial filter $w$, CSP is typically used to find a whole matrix of spatial filters $W^{kxk}$, with spatial filters ordered by the above variance ratio and orthogonal to each other. So the first filter $w_1$ results in the largest variance ratio and the last filter $w_k$ results in the smallest variance ratio. Different algorithms can then be used to subselect some set of filters to filter signals for a subsequent decoding algorithm.
# 
# The CSP-filtered signals can be used to construct features to train a classifier. Since the CSP-filtered signals should have very different variances for the different classes, the natural choice is to use the per-trial variances of the CSP-filtered signals as features. This results in as many features per trial as the number of CSP filters that were selected for decoding. Typically, one applies the logarithm to the variances to get more standard-normally distributed features.

# ## Filterbank

# CSP is typically applied to an EEG signal that has been bandpass filtered to a specific frequency range. The filtering to a frequency range is useful as brain signals cause EEG signal amplitude changes that are temporally and spatially different for different frequencies [refs]. For example, during movement the alpha rhythm may be suppressed for multiple electrodes covering a fairly large region on the scalp while the high gamma rhythm would be amplified for a few electrodes covering a smaller region.

# Filterbank Common Spatial Patterns applies CSP separately on signals bandpass-filtered to different frequency ranges [ref]. This allows to capture multiple frequency-specific changes in the EEG signal and can also make the decoding more robust to subject-specific signal characteristics, i.e., which frequency range is most informative for a given subject. The trial-log-variance features of each frequencyband and each CSP filter are then concatenated to form the entire trial feature vector. Typically, a feature selection procedure will select a subset of these features to train the final classifier.

# The overall FBCSP pipeline hence looks like this (from [ref]):
# 
# 1. **Bandpass filtering**: Different bandpass filters are applied to separate the raw EEG signal into different frequency bands.
# 2. **Epoching**: The continuous EEG signal is cut into trials as explained in the section “Input and labels.”
# 3. **CSP computation**: Per frequency band, the common spatial patterns (CSP) algorithm is applied to extract spatial filters. CSP aims to extract spatial filters that make the trials discriminable by the power of the spatially filtered trial signal (see Koles et al. [1990], Ramoser et al. [2000], and Blankertz et al. [2008] for more details on the computation). 
# 4. **Spatial filtering**: The spatial filters computed in Step 2 are applied to the EEG signal.
# 5. **Feature construction**: Feature vectors are constructed from the filtered signals: Specifically, feature vectors are the log-variance of the spatially filtered trial signal for each frequency band and for each spatial filter.
# 6. **Classification**: A classifier is trained to predict per-trial labels based on the feature vectors.
# 

# ## Filterbank network architecture
# 
# The first neural network architecture was developed by us in a prior master thesis [ref] to jointly learn the same steps that are learned separately by FBCSP. Concretely, the network simultaenously learn the spatial filters across many frequency bands and the classification weights for the trial variances of all resulting spatially filtered signals. To be able to do that, the network is fed with several signals that were bandpass-filtered to different frequency ranges. The network then performs the following steps:
# 
# 1. Apply learnable spatial filter weights, resulting in spatially filtered signals
# 2. Square the resulting signals
# 3. Sum the squared signals across the trial
# 4. Take the logarithm of the summed values
# 5. Apply learnable classification weights on these "log-variance" features
# 6. Take the softmax to produce per-class predictions.
# 
# The spatial filter weights and the classification weights are trained jointly.

# ![title](images/csp_as_a_net_explanation.png)

# ```{figure} images/csp_as_a_net_explanation.png
# ---
# name: filterbank-net-figure
# ---
# Filterbank network architecture overview.  Input signals were bandpass filtered to different frequency ranges. Signals are first transformed by learned spatial filters, then squared, summed and the log-transformed. The resulting features are transformed into class probabilities by a classification weights followed by the softmax function.
# ```

# ## Shallow Network Architecture

# Next, we developed the shallow network architecture, a more flexible architecture that also learns temporal filters on the input signal and on the later representation. Instead of bandpass-filtered signals, it is fed the raw signals as input. The first step are learnable temporal filters that are indepedently convolved with the signals of each EEG electrode. Afterwards, the channel dimension of the network representation    contains $\mathrm{electrodes} \cdot \mathrm{temporal~ filters}$ channels. In the next step that combines spatial filtering with mixing the outputs of the temporal filters, this network-channel dimension is linearly transformed by learned weights to a smaller dimensionality for further preprocessing. The resulting feature timeseries are then squared, average-pooled and log-transformed, which allows the network to more easily learn log-variance-based features. Unlike the filterbank network, the average pooling does not collapse the feature timeseries into one value per trial. So after these processing steps, still some temporal information about the timecourse of the variance throughout the trial can be preserved. Then, the final classification layer transforms these feature timecourses into class probabilities using a linear transformation and a softmax function.
# 
# 
# 
# 
# 

# ![title](images/3D_Diagram_MatplotLib.ipynb.0.png)

# ```{figure} images/3D_Diagram_MatplotLib.ipynb.0.png
# ---
# name: shallow-net-figure
# width: 50%
# ---
# Shallow network architecture, figure from [ref].  EEG input (at the top) is progressively transformed toward the bottom, until the final classifier output. Black cuboids: inputs/feature maps; brown cuboids: convolution/pooling kernels. The corresponding sizes are indicated in black and brown, respectively.
# ```

# ## Deep Network Architecture

# The deep architecture is a more generic architecture, closer to network architectures used in computer vision. The first two temporal convolution and spatial filtering layers are the same in the shallow network, which is followed by a ELU nonlinearity [ref] and max pooling. The following three blocks simply consist of a convolution, a ELU nonlinearity and a max pooling. In the end, there is again a final linear classification layer with a softmax function. Due to its less specific and more generic computational steps, the deep architecture should be able to capture a large variety of features. Hence, the learned features may also be less biased towards the amplitude features commonly used in task-related EEG decoding. 

# ![title](images/3D_Diagram_MatplotLib.ipynb.1.png)

# ```{figure} images/3D_Diagram_MatplotLib.ipynb.1.png
# ---
# name: deep-net-figure
# width: 75%
# ---
# Deep network architecture, figure from [ref].
# ```

# ## Residual Network

# We also developed a residual network (ResNet) for EEG decoding. We use the same residual blocks as the original paper, described in Figure {numref}`residual-net-figure`. Our ResNet used exponential linear unit activation functions [Clevert et al., 2016] throughout the network (same as the deep ConvNet) and also starts with a splitted temporal and spatial convolution (same as the deep and shallow ConvNets), followed by 14 residual blocks, mean pooling and a final softmax dense classification layer (for further details, see Supporting Information, Section A.3 in [ref]). 

# ![title](images/residual_block.png)

# ```{figure} images/residual_block.png
# ---
# name: residual-net-figure
# ---
# 
# Residual block, Figure from [ref]. "Residual block used in the ResNet architecture and as described in original paper (He et al. [2015]; see Fig. 2) with identity shortcut option A, except using ELU instead of ReLU nonlinearities."
# ```
# 

# ## To remove? : Temporal Convolutional Network

# In another master thesis [ref], Patrick Ch... developed the Temporal Convolutional Network for EEG decoding using automatic hyperparameter optimization. Temporal Convolutional Networks use residual blocks and dilated convolutions and had originally been introduced as an alternative to recurrent neural networks. 
