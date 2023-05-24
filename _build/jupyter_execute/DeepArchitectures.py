#!/usr/bin/env python
# coding: utf-8

# (network-architectures)=
# # Neural Network Architectures for EEG-Decoding

# ```{admonition} Three progressively more generic architectures
# * Shallow network learns temporal filters and later average-pools over large timeregions
# * Deep network uses smaller temporal filters and max-pooling over small timeregions
# * Residual network uses many layers with even smaller temporal filters
# ```

# We continued developing our neural network architectures with our EEG-specific development strategy of starting with networks that resemble feature-based algorithms. After the filterbank network from the master thesis, we adapted the so-called shallow network, initally also developed in the same master thesis {cite:p}`schirrmeister_msc_thesis_2015`. The shallow network still resembles filter bank common spatial patterns, but less closely than the filterbank network. After validating that these initial network architectures perform as well as filter bank common spatial patterns, we progressed to developing and evaluating more generic architectures.
# 
# In this section, I describe the architectures presented in our first publication on EEG deep learning decoding {cite:p}`schirrmeisterdeephbm2017`. This part uses text and figures from {cite:p}`schirrmeisterdeephbm2017` adapted for readibility in this thesis.

# ## Shallow Network Architecture

# ![title](images/3D_Diagram_MatplotLib.ipynb.0.png)

# ```{figure} images/3D_Diagram_MatplotLib.ipynb.0.png
# ---
# name: shallow-net-figure
# width: 50%
# ---
# Shallow network architecture, figure from {cite}`schirrmeisterdeephbm2017`.  EEG input (at the top) is progressively transformed toward the bottom, until the final classifier output. Black cuboids: inputs/feature maps; brown cuboids: convolution/pooling kernels. The corresponding sizes are indicated in black and brown, respectively. Note that the final dense layer operates only on a small remaining temporal dimension, making it similar to a regular convolutional layer.
# ```

# We developed the shallow network architecture, a more flexible architecture than the filterbank network that also learns temporal filters on the input signal and on the later representation. Instead of bandpass-filtered signals, it is fed the raw signals as input. 
# The steps the architecture implements are as follows (also see {numref}`shallow-net-figure`):
# 1. **Temporal Filtering** Learnable temporal filters are indepedently convolved with the signals of each EEG electrode. Afterwards, the channel dimension of the network representation contains $\mathrm{electrodes} \cdot \mathrm{temporal~ filters}$ channels. 
# 2. **Spatial Filtering** Combining spatial filtering with mixing the outputs of the temporal filters, the network-channel dimension is linearly transformed by learned weights to a smaller dimensionality for further preprocessing. 
# 3. **Log Average Power** The resulting feature timeseries are then squared, average-pooled and log-transformed, which allows the network to more easily learn log-power-based features. Unlike the filterbank network, the average pooling does not collapse the feature timeseries into one value per trial. So after these processing steps, still some temporal information about the timecourse of the variance throughout the trial can be preserved. 
# 4. **Classifier** The final classification layer transforms these feature timecourses into class probabilities using a linear transformation and a softmax function.

# ## Deep Network Architecture

# ![title](images/3D_Diagram_MatplotLib.ipynb.1.png)

# ```{figure} images/3D_Diagram_MatplotLib.ipynb.1.png
# ---
# name: deep-net-figure
# width: 75%
# ---
# Deep network architecture, figure from {cite:p}`schirrmeisterdeephbm2017`. Conventions as in {numref}`shallow-net-figure`.
# ```

# The deep architecture is a more generic architecture, closer to network architectures used in computer vision, see  {numref}`deep-net-figure` for a schematic overview. The first two temporal convolution and spatial filtering layers are the same in the shallow network, which is followed by a ELU nonlinearity (ELUs, $f(x)=x$ for $x > 0$ and $f(x) = e^x-1$ for $x <= 0$ {cite}`clevert_fast_2016`) and max pooling. The following three blocks simply consist of a convolution, a ELU nonlinearity and a max pooling. In the end, there is again a final linear classification layer with a softmax function. Due to its less specific and more generic computational steps, the deep architecture should be able to capture a large variety of features. Hence, the learned features may also be less biased towards the amplitude features commonly used in task-related EEG decoding. 

# ## Residual Network

# ![title](images/residual_block.png)

# ```{figure} images/residual_block.png
# ---
# name: residual-net-figure
# ---
# 
# Residual block, Figure from {cite}`schirrmeisterdeephbm2017`. "Residual block used in the ResNet architecture and as described in original paper ({cite}`he_deep_2015`; see Fig. 2) with identity shortcut option A, except using ELU instead of ReLU nonlinearities."
# ```
# 

# ```{table} Residual network architecture hyperparameters. Number of kernels, kernel and output size for all subparts of the network. Output size is always time x height x channels. Note that channels here refers to input channels of a network layer, not to EEG channels; EEG channels are in the height dimension. Output size is only shown if it changes from the previous block. Second convolution and all residual blocks used ELU nonlinearities. Note that in the end we had seven outputs, i.e., predictions for the four classes, in the time dimension (**7**x1x4 final output size). In practice, when using cropped training as explained in the following chapter, we even had 424 predictions, and used the mean of these to predict the trial.
# :name: residual-architectures-table
# 
# 
# | Layer/Block | Number of Kernels | Kernel Size | Output Size |
# |---|---|---|---|
# | Input |  |  | 1000x44x1 |
# | Convolution (linear) | 48 | 3x1 | 1000x44x48 |
# | Convolution (ELU) | 48 | 1x44 | 1000x1x48 |
# | ResBlock (ELU) | 48 | 3x1 |  |
# | ResBlock (ELU) | 48 | 3x1 |  |
# | ResBlock (ELU) | 96 | 3x1 (Stride 2x1) | 500x1x96 |
# | ResBlock (ELU) | 96 | 3x1 |  |
# | ResBlock (ELU) | 144 | 3x1 (Stride 2x1) | 250x1x96 |
# | ResBlock (ELU) | 144 | 3x1 |  |
# | ResBlock (ELU) | 144 | 3x1 (Stride 2x1) | 125x1x96 |
# | ResBlock (ELU) | 144 | 3x1 |  |
# | ResBlock (ELU) | 144 | 3x1 (Stride 2x1) | 63x1x96 |
# | ResBlock (ELU) | 144 | 3x1 |  |
# | ResBlock (ELU) | 144 | 3x1 (Stride 2x1) | 32x1x96 |
# | ResBlock (ELU) | 144 | 3x1 |  |
# | ResBlock (ELU) | 144 | 3x1 (Stride 2x1) | 16x1x96 |
# | ResBlock (ELU) | 144 | 3x1 |  |
# | Mean Pooling |  | 10x1 | 7x1x144 |
# | Convolution + Softmax | 4 | 1x1 | 7x1x4 |
# ```

# We also developed a residual network (ResNet {cite:p}`he_deep_2015`) for EEG decoding. Residual networks add the input of a residual computational block back to its output, and empirically this allows to stably train much deeper networks. We use the same residual blocks as the original paper, described in Figure {numref}`residual-net-figure`. Our ResNet used ELU activation functions throughout the network (same as the deep ConvNet) and also starts with a splitted temporal and spatial convolution (same as the deep and shallow ConvNets), followed by 14 residual blocks, mean pooling and a final softmax dense classification layer. 
# 
# In total, the ResNet has 31 convolutional layers, a depth where ConvNets without residual blocks started to show problems converging in the original ResNet paper {cite}`he_deep_2015`. In layers where the number of channels is increased, we padded the incoming feature map with zeros to match the new channel dimensionality for the shortcut, as in option A of the original paper {cite:p}`he_deep_2015`. The overall architecture is also shown in {numref}`residual-architectures-table`.

# ```{admonition} Three diverse architectures...
# :class: tip
# * to be evaluated on motor-related decoding and other tasks
# ```
