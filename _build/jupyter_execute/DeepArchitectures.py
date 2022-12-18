#!/usr/bin/env python
# coding: utf-8

# (network-architectures)=
# # Neural Network Architectures for EEG-Decoding

# We continued developing our neural network architectures with our EEG-specific development strategy of starting with networks that resemble feature-based algorithms. After the filterbank network from the master thesis, we adapted the so-called shallow network, initally also developed in the same master thesis. The shallow network still resembles filter bank common spatial patterns, but less closely than the filterbank network. After validating that these initial network architectures perform as well as filter bank common spatial patterns, we progressed to developing and evaluating more generic architectures.
# 
# In this section, I describe the architectures presented in our first publication on EEG deep learning decoding {cite}`schirrmeisterdeephbm2017`. This part uses text and figures from {cite}`schirrmeisterdeephbm2017` adapted for readibility in this thesis.

# ## Shallow Network Architecture

# We developed the shallow network architecture, a more flexible architecture than the filterbank network that also learns temporal filters on the input signal and on the later representation. Instead of bandpass-filtered signals, it is fed the raw signals as input. The first step are learnable temporal filters that are indepedently convolved with the signals of each EEG electrode. Afterwards, the channel dimension of the network representation    contains $\mathrm{electrodes} \cdot \mathrm{temporal~ filters}$ channels. In the next step that combines spatial filtering with mixing the outputs of the temporal filters, this network-channel dimension is linearly transformed by learned weights to a smaller dimensionality for further preprocessing. The resulting feature timeseries are then squared, average-pooled and log-transformed, which allows the network to more easily learn log-variance-based features. Unlike the filterbank network, the average pooling does not collapse the feature timeseries into one value per trial. So after these processing steps, still some temporal information about the timecourse of the variance throughout the trial can be preserved. Then, the final classification layer transforms these feature timecourses into class probabilities using a linear transformation and a softmax function.

# ![title](images/3D_Diagram_MatplotLib.ipynb.0.png)

# ```{figure} images/3D_Diagram_MatplotLib.ipynb.0.png
# ---
# name: shallow-net-figure
# width: 50%
# ---
# Shallow network architecture, figure from [ref].  EEG input (at the top) is progressively transformed toward the bottom, until the final classifier output. Black cuboids: inputs/feature maps; brown cuboids: convolution/pooling kernels. The corresponding sizes are indicated in black and brown, respectively. Note that the final dense layer operates only on a small remaining temporal dimension, making it similar to a regular convolutional layer.
# ```

# ## Deep Network Architecture

# The deep architecture is a more generic architecture, closer to network architectures used in computer vision. The first two temporal convolution and spatial filtering layers are the same in the shallow network, which is followed by a ELU nonlinearity (ELUs, $f(x)=x$ for $x > 0$ and $f(x) = e^x-1$ for $x <= 0$ {cite}`clevert_fast_2016`) and max pooling. The following three blocks simply consist of a convolution, a ELU nonlinearity and a max pooling. In the end, there is again a final linear classification layer with a softmax function. Due to its less specific and more generic computational steps, the deep architecture should be able to capture a large variety of features. Hence, the learned features may also be less biased towards the amplitude features commonly used in task-related EEG decoding. 

# ![title](images/3D_Diagram_MatplotLib.ipynb.1.png)

# ```{figure} images/3D_Diagram_MatplotLib.ipynb.1.png
# ---
# name: deep-net-figure
# width: 75%
# ---
# Deep network architecture, figure from {cite}`schirrmeisterdeephbm2017`. Conventions as in {numref}`shallow-net-figure`.
# ```

# ## Residual Network

# We also developed a residual network (ResNet {cite}`he_deep_2015`) for EEG decoding. We use the same residual blocks as the original paper, described in Figure {numref}`residual-net-figure`. Our ResNet used ELU activation functions throughout the network (same as the deep ConvNet) and also starts with a splitted temporal and spatial convolution (same as the deep and shallow ConvNets), followed by 14 residual blocks, mean pooling and a final softmax dense classification layer (for further details, see Supporting Information, Section A.3 in {cite}`schirrmeisterdeephbm2017`). 

# ![title](images/residual_block.png)

# ```{figure} images/residual_block.png
# ---
# name: residual-net-figure
# ---
# 
# Residual block, Figure from {cite}`schirrmeisterdeephbm2017`. "Residual block used in the ResNet architecture and as described in original paper ({cite}`he_deep_2015`; see Fig. 2) with identity shortcut option A, except using ELU instead of ReLU nonlinearities."
# ```
# 
