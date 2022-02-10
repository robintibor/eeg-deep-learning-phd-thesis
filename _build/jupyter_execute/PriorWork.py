#!/usr/bin/env python
# coding: utf-8

# (prior-work)=
# # Prior Work

# Maybe add a question that we address at end of each section?

# ## Decoding Problems and Baselines

# ```{admonition}  Deep Learning on EEG studies prior to 2017 only had limited comparisons to feature-based baselines 
# * From 19 identified studies, only 5 had an external baseline
# * Decoding problems were very varied, movement-related decoding the most frequent problem
# * Remained unclear how deep learning approaches compare to well-tuned feature-based approaches
# ```

# |  Decoding problem                                                                                                                     | Number of studies                                                               | With external baseline|
# |:-----------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
# |Imagined or Executed Movement|6|2|
# |Oddball/P300|5|1|
# |Epilepsy-related|4|2|
# |Music Rhythm|2|0|
# |Memory Performance/Cognitive Load|2|0|
# |Driver Performance|1|0|

# ```{admonition} We therefore focused on movement-related decoding with strong, externally validated baselines
# :class: tip
# * Movement-related decoding among most well-researched EEG decoding problems
# * Very strong feature-based baselines exist
# ```

# ## Input Domains and Frequency Ranges

# ```{admonition} Prior studies using time domain inputs mostly excluded higher frequencies
# * More studies used time domain inputs than frequency domain inputs
# * Only four studies used time domain inputs above 50 Hz
# ```

# In[10]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn
import numpy as np
import re
from myst_nb import glue
seaborn.set_palette('colorblind')
seaborn.set_style('darkgrid')
import re

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
#matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14
a = np.array(['Time,  8–30 Hz ', 'Time, 0.1–40 Hz ', 'Time, 0.05–15 Hz ',
       'Time, 0.3–20 Hz ', 'Frequency, 6–30 Hz ', ' Frequency, 0–200 Hz ',
       'Time,  1–50 Hz ', ' Time,  0–100 HZ ',
       'Frequency, mean amplitude for 0–7 Hz, 7–14 Hz, 14–49 Hz ',
       'Time, 0.5–50 Hz ', 'Time,  0–128 Hz ',
       ' Frequency, mean power for 4–7 Hz, 8–13 Hz, 13–30 Hz ',
       'Time, 0.5–30Hz ', 'Time, 0.1–50 Hz ',
       'Frequency, 4–40 Hz, using FBCSP ',
       ' Time and frequency evaluated, 0-200 Hz ', 'Frequency, 8–30 Hz ',
       'Time, 0.15–200 Hz ', ' Time, 0.1-20 Hz '])
domain_strings = [s.split(',')[0] for s in a]
start_fs = [float(re.sub(r'[a-z ]+',r'', re.split(r'[–-–-]'," ".join(s.split(',')[1:]))[0])) for s in a]
end_fs = [float(re.sub(r'[a-z HZFBCSP]+',r'', re.split(r'[–-–-]'," ".join(s.split(',')[1:]))[1])) for s in a]
domain_strings = np.array(domain_strings)
start_fs = np.array(start_fs)
end_fs = np.array(end_fs)

freq_mask = np.array(['freq' in s.lower() for s in domain_strings])
time_mask = np.array(['time' in s.lower() for s in domain_strings])

fig = plt.figure(figsize=(8,4))
rng = np.random.RandomState(98349384)
color = 'grey'
i_sort = np.flatnonzero(time_mask)[np.argsort(end_fs[time_mask])]
for i, (d,s,e) in enumerate(zip(
        domain_strings[i_sort], start_fs[i_sort], end_fs[i_sort])):
    offset = 0.6*i/len(i_sort) - 0.3
    plt.plot([offset,offset] , [s, e], marker='o', alpha=1, color=color, ls='-')
i_sort = np.flatnonzero(freq_mask)[np.argsort(end_fs[freq_mask])]
for i, (d,s,e) in enumerate(zip(
        domain_strings[i_sort], start_fs[i_sort], end_fs[i_sort])):
    offset = 0.6*i/len(i_sort) + 0.7
    plt.plot([offset,offset] , [s, e], marker='o', alpha=1, color=color, ls='-')

plt.xlim(-0.5,1.5)
plt.xlabel("Input domain")
plt.ylabel("Frequency [Hz]")
plt.xticks([0,1], ["Time", "Frequency"], rotation=45)
plt.title("Input domains and frequency ranges in prior work", y=1.05)
plt.yticks([0,25,50,75,100,150,200])
glue('input_domain_fig', fig)
plt.close(fig)
None


# ```{glue:figure} input_domain_fig
# 
# 
# *Input domains and frequency ranges in prior work*. Grey lines represent frequency ranges of individual studies. Note that many studies only include frequencies below 50 Hz, some use very restricted ranges (alpha/beta band).
# ```
# %:figclass: margin-caption

# ```{admonition} We used also high-gamma frequencies > 50 Hz on one dataset
# :class: tip
# * Used very suitable dataset for high-gamma nalysis 
# * Deep networks should be able to extract information from any frequency range 
# ```

# ## Networn Architectures

# ```{admonition}  Most investigated network architectures were fairly shallow (below 4 layers)
# * Unlike architectures in computer vision, most EEG DL architectures had only 1-3 convolutional layers
# * Unlike architectures in computer vision, many architectures used several dense layers
# ```

# In[1]:




ls = np.array([' 2/2 ', ' 3/1 ', ' 2/2 ', ' 3/2 ', ' 1/1 ', ' 1/2 ', ' 1/3 ',
       ' 1–2/2 ', ' 3/1 (+ LSTM as postprocessor) ', ' 4/3 ', ' 1-3/1-3 ',
       ' 3–7/2 (+ LSTM or other temporal post-processing (see design choices)) ',
       ' 2/1 ', ' 3/3 (Spatio-temporal regularization) ',
       ' 2/2 (Final fully connected layer uses concatenated output by convolutionaland fully connected layers) ',
       ' 1-2/1 ',
       '2/0 (Convolutional deep belief network, separately trained RBF-SVM classifier) ',
       ' 3/1 (Convolutional layers trained as convolutional stacked autoencoder with target prior) ',
       ' 2/2 '])

conv_ls = [l.split('/')[0] for l in ls]
low_conv_ls = [int(re.split(r'[–-]', c)[0])for c in conv_ls]
high_conv_ls = [int(re.split(r'[–-]', c)[-1])for c in conv_ls]
dense_ls = [l.split('/')[1] for l in ls]
low_dense_ls = [int(re.split(r'[–-]', c[:8])[0][:2])for c in dense_ls]
high_dense_ls = [int(re.split(r'[–-]', c[:8])[-1][:2])for c in dense_ls]

all_conv_ls = np.concatenate([np.arange(low_c, high_c+1) for low_c, high_c in zip(low_conv_ls, high_conv_ls)])
all_dense_ls = np.concatenate([np.arange(low_c, high_c+1) for low_c, high_c in zip(low_dense_ls, high_dense_ls)])
bincount_conv = np.bincount(all_conv_ls)
bincount_dense = np.bincount(all_dense_ls)
rng = np.random.RandomState(98349384)
color = 'grey'
fig = plt.figure(figsize=(8,4))
for low_c, high_c in zip(low_conv_ls, high_conv_ls):
    offset = rng.randn(1) * 0.1
    tried_cs = np.arange(low_c, high_c+1)
    plt.plot([offset,] * len(tried_cs), tried_cs, marker='o', alpha=0.5, color=color, ls=':')
    
for i_c, n_c in enumerate(bincount_conv):
    plt.scatter(0.4, i_c, color=color, s=n_c*40)
    plt.text(0.535, i_c, str(n_c)+ "x", ha='left', va='center')

for low_c, high_c in zip(low_dense_ls, high_dense_ls):
    offset = 1 + rng.randn(1) * 0.1
    tried_cs = np.arange(low_c, high_c+1)
    plt.plot([offset,] * len(tried_cs), tried_cs, marker='o', alpha=0.5, color=color, ls=':')
    
for i_c, n_c in enumerate(bincount_dense):
    plt.scatter(1.4, i_c, color=color, s=n_c*40)
    plt.text(1.535, i_c, str(n_c)+ "x", ha='left', va='center')

plt.xlim(-0.5,2)
plt.xlabel("Type of layer")
plt.ylabel("Number of layers")
plt.xticks([0,1], ["Convolutional", "Dense"], rotation=45)
plt.yticks([1,2,3,4,5,6,7]);
plt.title("Number of layers in prior works' architectures", y=1.05)
glue('layernum_fig', fig)
plt.close(fig)
None


# ```{glue:figure} layernum_fig
# 
# 
# *Number of layers in prior work*. Small grey markers represent individual architectures. Dashed lines indicate different number of layers investigated in a single study (e.g., a single study investigated 3-7 convolutional layers). Larger grey markers indicate sum of occurences of that layer number over all studies (e.g., 9 architectures used 2 convolutional layers). Note most architectures use only 1-3 convolutional layers.
# ```
# %:figclass: margin-caption

# ```{admonition} We will evaluate shallower and deeper architectures 
# :class: tip
# * ...
# ```

# ## Hyperparameter Evaluations

# ```{admonition}  Prior work varied widely in which design choices and training strategies were compared
# * Six studies did not compare any design choices or training strategies
# * Most common was to try different kernel sizes
# * Only one study evaluated a wider range of hyperparameters for both design choices and training strategies
# ```

# | Study                                                                                                                              | Design choices                                                                                                                                                                                  | Training strategies                                                                                                                      |
# |:-----------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
# |{cite}`lawhern_eegnet:_2016`                      | Kernel sizes                                                                                                                                                                                    |                                                                                                                                          |
# |{cite}`sun_remembered_2016`                            |                                                                                                                                                                                                 | Different time windows                                                                                                                   |
# |{cite}`tabar_novel_2017`                           | Addition of six-layer stacked autoencoder on ConvNet features <br> Kernel sizes                                                                                                              |                                                                                                                                          |
# | {cite}`liang_predicting_2016`           |                                                                                                                                                                                                 | Different subdivisions of frequency range <br>Different lengths of time crops <br>Transfer learning with auxiliary non-epilepsy datasets |
# |{cite}`hajinoroozi_eeg-based_2016`    | Replacement of convolutional layers by restricted Boltzmann machines with slightly varied network architecture}                                                                                 |                                                                                                                                          |
# |{cite}`antoniades_deep_2016`                                                    | 1 or 2 convolutional layers                                                                                                                                                                     |                                                                                                                                          |
# |{cite}`page_wearable_2016`                  |                                                                                                                                                                                                 | Cross-subject supervised training, within-subject finetuning of fully connected layers                                                   |
# |{cite}`bashivan_learning_2016`                | Number of convolutional layers <br>Temporal processing of ConvNet output by max pooling, temporal convolution, LSTM or temporal convolution + LSTM                                              |                                                                                                                                          |
# |{cite}`stober_learning_2016`                                                             | Kernel sizes                                                                                                                                                                                    | Pretraining first layer as convolutional autoencoder with different constraints                                                          |
# |{cite}`sakhavi_parallel_2015`                       | Combination ConvNet and MLP (trained on different features) vs. only ConvNet vs. only MLP                                                                                                       |                                                                                                                                          |
# |{cite}`stober_using_2014`  | Best values from automatic hyperparameter optimization: frequency cutoff, one vs two layers, kernel sizes, number of channels, pooling width                                                    | Best values from automatic hyperparameter optimization: learning rate, learning rate decay, momentum, final momentum                     |
# |{cite}`wang_deep_2013`                | Partially supervised CSA                                                                                                                                                                        |                                                                                                                                          |
# |{cite}`cecotti_convolutional_2011` | Electrode subset (fixed or automatically determined) <br>Using only one spatial filter <br>Different ensembling strategies                                                                      |                                                                                                                                          |
# 
# 

# ```{admonition} We evaluated many design choices and two training strategies
# :class: tip
# * Find out if network design improvements in computer vision carry over to EEG decoding
# * Test training strategies using whole trials or sliding windows within trials
# ```

# ## Visualizations

# ```{admonition}  Prior work mostly analyzed weights and activations 
# * Eight studies did not present any visualization
# * Visualizations in input space (maximally activating inputs, input occlusions, saliency maps) were used in three studies
# ```

# | Study                                                                                                                              | Visualization type(s)                                                                                            | Visualization findings   |
# |:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:-------------------------|
# |{cite}`sun_remembered_2016`                            | Weights (spatial)                                                                                                | Largest weights found over prefrontal and temporal cortex                                                                                                                                                                                                                                                                    |
# |{cite}`manor_multimodal_2016`             | Weights <br> Activations <br> Saliency maps by gradient                                                          | Weights showed typical P300 distribution <br>Activations were high at plausible times (300-500ms) <br>Saliency maps showed plausible spatio-temporal plots                                                                                                                      |
# |{cite}`tabar_novel_2017`                           | Weights (spatial + frequential)                                                                                  | Some weights represented difference of values of two electrodes on different sides of head                                                                                                                                                                                      |
# |{cite}`liang_predicting_2016`           | Weights <br> Clustering of weights                                                                               | Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)                                                                                                                                                                                |
# |{cite}`antoniades_deep_2016`                                                    | Weights <br>Correlation weights and interictal epileptic discharges (IED) <br>Activations                        | Weights increasingly correlated with IED waveforms with increasing number of training iterations <br>Second layer captured more complex and well-defined epileptic shapes than first layer <br>IEDs led to highly synchronized activations for neighbouring electrodes          |
# |{cite}`thodoroff_learning_2016`                      | Input occlusion and effect on prediction accuracy                                                                | Allowed to locate areas critical for seizure                                                                                                                                                                                                                                    |
# |{cite}`george_single-trial_2016`                         | Weights (spatial)                                                                                                | Some filter weights had expected topographic distributions for P300 <br>Others filters had large weights on areas not traditionally associated with P300                                                                                                                        |
# |{cite}`bashivan_learning_2016`                | Inputs that maximally activate given filter <br>Activations of these inputs <br>"Deconvolution" for these inputs | Different filters were sensitive to different frequency bands <br>Later layers had more spatially localized activations <br>Learned features had noticeable links to well-known electrophysiological markers of cognitive load <br>                                             |
# |{cite}`stober_learning_2016`                                                             | Weights (spatial+3 timesteps, pretrained as autoencoder)                                                         | Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings |
# |{cite}`manor_convolutional_2015`             | Weights <br> Mean and single-trial activations                                                                   | Spatiotemporal regularization led to softer peaks in weights <br>Spatial weights showed typical P300 distribution <br>Activations mostly had peaks at typical times (300-400ms)                                                                                                 |
# |{cite}`cecotti_convolutional_2011` | Weights                                                                                                          | Spatial filters were similar for different architectures <br>Spatial filters were different (more focal, more diffuse) for different subjects                                                                                                                                   |

# ```{admonition} We developed visualizations investigating known frequency amplitude features
# :class: tip
# * Investigate in how far networks exploit features known to work well for movement-related decoding
# ```

# ## References
# 
# ```{bibliography} ./references.bib
# ```
# 
