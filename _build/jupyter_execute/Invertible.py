#!/usr/bin/env python
# coding: utf-8

# (invertible-networks)=
# # Invertible Networks

# Invertible networks are networks that are invertible by design, i.e., any network output can be mapped back to a corresponding input [refs] bijectively. The ability to invert any output back to the input enables different interpretability methods and furthermore allows training invertible networks as generative models via maximum likelihood. 
# 
# The background chapter starts by explaining how to design invertible networks, proceeds to detail their training methodologies as generative models or classifiers, and goes on to outline interpretability techniques that help reveal the learned features crucial for their classification tasks. We then present the results of invertible networks trained to decode pathology from EEG, showing competitive results with regular deep convolutional networks. Our visualizations show the network uses slowing, i.e. increased amplitudes in the lower frequencies as well as spikes/burts as markers that indicate pathology and a strong stable alpha rhythm, especially on the posterior electrodes as markers for normal EEG.

# ## Background

# [Figure coupling blocks]
# * you could have a figure here with time series and half of time series computed sth, then additive coefficient added 
# 
# Invertible networks, a type of neural networks, use layers constructed specifically to maintain invertibility, thereby rendering the entire network structure invertible. Often-used invertible layers are coupling layers, invertible linear layers and activation normalization layers as explained in the following.
# 
# Coupling layers work by splitting a d-dimensional input $x$ into two parts $x_1$ and $x_2$ of $d_1$ and $d_2$ dimensions with $d_1 + d_2 = d$. For example in a timeseries one may take the input at all the even time indices as $x_1$ and all the odd time indices as $x_2$ odd time indices. Then, one transforms $x_1$ in an invertible way (e.g., by adding something to it) based on computations only performed on $x_2$, while leaving $x_2$ unchanged. For, example additive coupling works as follows:
# $y_1 = x_1 + f(x_2); y_2=x_2$, with $f$ being an arbitrary function, e.g., any (potentially non-invertible) neural network. It can be inverted given $y_1$ and $y_2$ as follows: $x_1 = y_1 - f(y_2); x_2=y_2$. Instead of addition one may use any other invertible function, a common one is an affine transformation where $f$ produces translation and scaling coefficients $f_t$ and $f_s$:
# $y_1 = x_1 \cdot f_s(x_2) + f_t(x_2); y_2=x_2$, with inversion $x_1 = \frac{(y_1  - f_t(x_2))}{f_s(x_2)}; y_2=x_2$. 
# The splitting of dimensions can be done in multiply ways, like using odd or even indices or using difference and mean between two neighbouring samples (akin to one stage of a Haar Wavelet).
# 
# Invertible linear layers compute an invertible linear transformation (an automorphism) of their input. Concretely they multiply a $d$-dimensional vector $\mathbf{x}$ with a $dxd$-dimensional matrix $W$, where the $W$ has to be invertible, i.e., have nonzero determinant. 
# $y=W \mathbf{x}$ with inverse $x=W^{-1} \mathbf{y}$. For multidimensional arrays like feature maps in a convolutional network, these linear transformations are usually done per-position, as so-called invertible 1x1 convolutions in the 2d case.
# 
# Activation normalization layers perform an affine transformation with learned parameters, e.g., $y=x\cdot s+t$, with $s$ and $t$ learned scaling and translation parameters. These have also been used to replace batch normalization and initialized data-dependently to maintain unit variance at the beginning of training.
# 
# 

# ## Generative models via maximum likelihood

# Invertible networks can also be trained as generative models via maximum likelihood. In maximum likelihood training, the network is optimized to maximize the probability densities of the training inputs. For that, the network maps the inputs into a latent space such that the probability densities are maximized under a predefined prior within this latent space, e.g., a Gaussian distribution. In addition, for these probability densities to form a valid probability distribution in the input space, one has to account for how much the network's mapping function squeezes and expands volume. We'll proceed to illustrate this in an example below.
# 
# 
# 
# $p(x) = p_\textrm{prior}(f(x)) \cdot  | \det \left( \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right)|$
# 
# 
# * figure for that (map samples to latent space, shift achagne volume, latent dist just 3 bars, invert show how dist looks in input soace, what it integrates to 
# * etc.

# ## Generative classifiers

# Invertible networks trained as class-conditional generative models can also be used as classifiers. Class-conditional generative networks may be implemented in different ways, for example with a separate prior in latent space for each class. Given the class-conditional probability densities $p(x|c_i)$, one can obtain class probabilities via Bayes formula as $p(c_i|x)=\frac{p(x|c_i)}{\sum_jp(x|c_j)}$.
# 
# Pure class-conditional generative training may not yield networks that perform well as classifiers. The reason is that at least in theory, the relative reductions in maximum likelihood loss one obtains from knowing the class label are very small for high-dimensional inputs, for example much smaller than typical differences between two runs of the same network [REF]. This is understandable from a compression perspective, so using that under Shannon's theorem more probable inputs need less bits to encode than less likely inputs, or more precisely $\textrm{Number of bits needed}(x) = \log_2 p(x)$. Even if you assume some one needs only 1 bit per dimension, a high-dimensional input like our 2688-dimensional EEG signal will need 2688 bits to encode. How many bits are needed for the class information? To distinguish between n classes, one needs only $\log_2(n)$ bits, so in case of binary pathology classification, only 1 bit is needed, therefore the optimal class-conditional model will only be 1 bit better than the optimal class-independent model. In contrast, the loss difference between two training runs of the same network will typically be at least 1 to two orders of magnitude larger. In practice, the gains from using a class-conditional model, by e.g., using a separate prior per class in latent space, are usually larger, but it is not a priori clear if the reductions in loss from exploiting the class label are high enough to result in a good classification model.
# 
# Various methods have been proposed to improve the performance of using generative classifiers. For example, people have fixed the per-class latent gaussian priors so that they retain the same distance throughout training [Ref Pavel] or added a classification loss term $L_class(x)=\log_2 (...etc softmax) $ to the training [Ref VIB heidelberg]. In our work, we experimented with adding a classification loss term to the training, and also found using a learned temperature before the softmax helps the training. 

# ## Invertible Network for EEG Decoding

# We designed an invertible network for EEG Decoding using invertible components used in the literature, primarily from the Glow architecture [REF]. Our architecture consists of three stages that operate on sequentially lower temporal resolutions. Similar to glow, the individual stages consists of several blocks of Activation Normalization, Invertible Linear Channel Transformations and Coupling Layers. Between each stage, we downsample by computing the mean and difference of two neighbouring timepoints and moving these into the channel dimension. Unlike Glow, we keep processing all dimensions throughout all stages, finding this architecture to reach competitive accuracy on pathology decoding.
# 
# [diagram]

# ### Generative Invertible Networks for Pathology Decoding
# 
# We apply our EEG-InvNet to pathology decoding as in .. We chose pathology decoding as  (i)n the question which learned features the networks uses to diagnose pathology seemed especially fascinating to us (ii) the TUH dataset is especially large.

# * Same accuracy as other good networks

# |Deep|Shallow|TCN|EEGNet|EEG-InvNet Disc|EEG-InvNet Gen|
# |-|-|-|-|-|-|
# |84.6|84.1|86.2|83.4|85.5|77.2|

# ![title](images/net-disc-prototypes.png)

# ```{figure} images/net-disc-prototypes.png
# ---
# name: disc-invnet-prototypes
# ---
# Learned Class Prototypes from Invertible Network. Obtained by inverting learned means of class-conditional gaussian distributions from latent space to input space through the invertible network trained for pathology decoding.
# 
# ```

# ![title](images/marginal-chan.png)

# ```{figure} images/marginal-chan.png
# ---
# name: marginal-chan
# ---
# Learned Per-Channel Prototypes from Invertible Network. Each channels' input is optimized independently to increase the invertible networks prediction for the respective class. During that optimization, signals for the other non-optimized channels are sampled from the training data. 
# 
# ```

# ### Learning a small interpretable network

# ![cos-pattern](images/cos-sim-net-pattern.png)

# ```{figure} images/cos-sim-net-pattern.png
# ---
# name: marginal-chan
# ---
# Visualization of small interpretable network trained to mimic the EEG-InvNet. Signals are colored by weights of a model that learns to predict strength of output of that signals cosine similarity from overall output. 
# 
# ```

# In[ ]:





# Results:
# 
# * Best results
#   * Try recreating all results for n_class_independent
#   * Show that it works as good as others
#     * Small table for that
# * Prototypes for pathological 
# * Optimize per Chan Marginal Probability
# * Small Cos Net that recreates
# * (low freq?)
# * Mean Zs for medications etc.?excessive beta/excessive theta? and one medication
# 

# * 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ![](images/bcic_iv_2a_right_vs_rest.png)

# ```{figure} images/bcic_iv_2a_right_vs_rest.png
# ---
# name: bcic-iv-2a-right-vs-rest
# ---
# Right hand vs resting state class prototypes learned by an invertible network on the BCIC IV 2a dataset. Note the increased alpha oscillation for resting state, especially on the left side.
# ```

# ![](images/bcic_iv_2a_right_vs_rest_C3.png)

# ```{figure} images/bcic_iv_2a_right_vs_rest.png
# ---
# name: bcic-iv-2a-right-vs-rest-C3
# ---
# Right hand vs resting state class prototypes learned by an invertible network on the BCIC IV 2a dataset, showing the C3 electrode. Note the increased alpha oscillation.
# ```

# Class prototypes are a visualization obtainable from the trained invertible network. Here, the invertible network was trained as a class-conditional generative model via maximum likelihood with an extra classification loss [refs etc.]. To obtain the class protypes we first found the maxima of each class  distribution, i.e., of the learned class-conditional distribution $p_{\theta}(x|y_c)$. From that starting point, the synthetic prototypes were further optimized to minimize $L_{proto}(x) = -w_{prob}\log_{}\left(p_{\theta}\left(x\mid y_{c}\right)\right)\,-w_{class}\,\log_{}\left(p_{\theta}\left(y_{c}\mid x\right)\right)$. 
# 
# Prototypes for right hand vs resting state show a plausible discriminative pattern with increasing alpha oscillation of resting state compared to right hand, see {numref}`bcic-iv-2a-right-vs-rest`. Note that due to the discriminative training and the discriminative term in the optimization, these visualizations may show discriminative patterns that do not directly correspond to how actual signals for these classes look like. For example, in the actual data there may a decreasing right hand oscillation and a stable oscillation for resting state. See also the discussion in {ref}`perturbation-visualization-interpretation`.

# Todo:
# * Neural architecture search
# * Simplebits
