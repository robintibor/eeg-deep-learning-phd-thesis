#!/usr/bin/env python
# coding: utf-8

# (movement-related)=
# # Decoding Movement-Related Brain Activity

# ## Background and our Development Strategy

# Task-related, and especially movement-related decoding are among the most researched  paradigms in EEG decoding. A typical setup is that subjects receive a cue for a specific body part (e.g. right hand, feet, tongue, etc.) and either move this body part (motor execution) or just imagine to move this part (motor imagery). The EEG signals acquired during the imagined or executed movements then often contain patterns specific to the body part that was moved or thought about. These patterns can then be decoded using machine learning.
# 
# Many feature-based methods have been used to decode executed and imagined movements from EEG. 
# * most popular?
# * FBSCP, forward reference (explained in...)
# [...]
# 
# 
# 
# Deep neural network-based approaches had only been investigated more intensively from a much later date. 
# * beginning works blabla
# * not clear evaluation, also mention year
# 
# Development strategy:
# * first use something close to feature-based stuff (include master thesis)
# * then move to more complex/generic
# * always ablate hyperparameter choices
# 
# 
# 
# 
# ```{admonition} Most well-performing movement-intention-decoding methods have strong assumptions 
# * They use knowledge about brain signals 
# * Assume changes in amplitude of oscillations in specific frequency ranges predict the intention
# * These changes also are assumed to follow specific spatial patterns
# ```
# 
# 
# 
# ```{admonition} Movement-related tasks are the most researched tasks in EEG decoding.
# * Many studies have been done on it before the advent of deep learning
# * Hence, good feature-based baselines exist
# ```

# 

# [Explain background with beautiful pictures]

# ```{admonition} We followed the strategy of iteratively relaxing these assumptions in our deep learning development
# * First, use a network that exactly embeds the strategies of FBCSP, simply train jointly
# * Then a "shallow" network with large kernel sizes, mean pooling and squared activation to easily extract slow amplitude changes
# * Then a "deep" network with smaller kernel sizes and activation functions as networks in computer vision for a more generic network
# ```

# ## Filter Bank Common Spatial Patterns

# 

# ## Models Details

# In[ ]:





# ### Prior Work

# #### Filterbank Network
# 

# In a master thesis, we developed a network that closely mimics the processing steps of filter bank common spatial patterns

# 

# 

# ### Shallow Network

# In[ ]:





# ### Deep Network

# ### FBCSP Baseline

# ## Cropped Training

# ## Design Choices

# ## Results
# 
# ### FBCSP Network
# check master thesis
# explain why not checked further
# (OR JUST DO IT?)

# ### Shallow And Deep Results

# as in paper

# ## Design Choices etc.

# ## visualizations
# 
# * check what is actually used?
# * precise definition difficult
# 
# * also note extra results on other datasets and why and also mention further papers task related
# 
# * possibly comparison with current state

# ## old - merge

# * unclear how good deep learning can be for task-related activity
# * important benchmark as many decode on this
# * our approach to evaluate
# * our benchmark method (copy FBCSP text)
# * 

# 
