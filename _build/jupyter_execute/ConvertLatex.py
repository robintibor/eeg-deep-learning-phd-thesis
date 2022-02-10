#!/usr/bin/env python
# coding: utf-8

# 
# ## old, copied

# 
# 
# |Study |  Decoding problem |External Baseline|
# | :--- | --- | --- |
# | This manuscript, Schirrmeister et. al (2017) |Imagined and executed movement classes, within subject |FBCSP + rLDA|
# | Single-trial EEG classification of motor imagery using deep convolutional neural networks, citet{tang_single-trial_2017} |Imagined movement classes, within-subject | FBCSP ||
# | EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, {cite}`lawhern_eegnet:_2016` | Oddball response (RSVP), error response (ERN), movement classes (voluntarily started and imagined) | [recheck] |
# | Remembered or Forgotten? –- An EEG-Based Computational Prediction Approach, {cite}`sun_remembered_2016` | Memory performance, within-subject ||
# |Multimodal Neural Network for Rapid Serial Visual Presentation Brain Computer Interface, {cite}`manor_multimodal_2016`|Oddball response using RSVP and image (combined image-EEG decoding), within-subject|Time, 0.3–20 Hz | 3/2 | | |  | Weights <br> Activations <br> Saliency maps by gradient |Weights showed typical P300 distribution <br>Activations were high at plausible times (300-500ms) <br>Saliency maps showed plausible spatio-temporal plots
# | A novel deep learning approach for classification of EEG motor imagery signals, {cite}`tabar_novel_2017` |Imagined and executed movement classes, within-subject |Frequency, 6–30 Hz | 1/1 | multicolumn{2}{p{0.285	extwidth}}{Addition of six-layer stacked autoencoder on ConvNet features <br> Kernel sizes} | FBCSP, Twin SVM, DDFBS, Bi-spectrum, RQNN  | Weights (spatial + frequential) |Some weights represented difference of values of two electrodes on different sides of head
# | Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, {cite}`liang_predicting_2016` |Seizure prediction, within-subject | Frequency, 0–200 Hz | 1/2 | | Different subdivisions of frequency range <br>Different lengths of time crops <br>Transfer learning with auxiliary non-epilepsy datasets || Weights <br> Clustering of weights |Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)
# | EEG-based prediction of driver's cognitive performance by deep convolutional neural network, {cite}`hajinoroozi_eeg-based_2016` |Driver performance, within- and cross-subject |Time,  1–50 Hz | 1/3 |multicolumn{2}{p{0.285	extwidth}}{Replacement of convolutional layers by restricted Boltzmann machines with slightly varied network architecture}  | |
# | Deep learning for epileptic intracranial EEG data, {cite}`antoniades_deep_2016` |Epileptic discharges, cross-subject | Time,  0–100 HZ | 1–2/2 | 1 or 2 convolutional layers |  | |Weights <br>Correlation weights and interictal epileptic discharges (IED) <br>Activations |Weights increasingly correlated with IED waveforms with increasing number of training iterations <br>Second layer captured more complex and well-defined epileptic shapes than first layer <br>IEDs led to highly synchronized activations for neighbouring electrodes
# | Learning Robust Features using Deep Learning for Automatic Seizure Detection, {cite}`thodoroff_learning_2016` |Start of epileptic seizure, within- and cross-subject |Frequency, mean amplitude for 0–7 Hz, 7–14 Hz, 14–49 Hz | 3/1 (+ LSTM as postprocessor) | | | Hand crafted features + SVM | Input occlusion and effect on prediction accuracy |Allowed to locate areas critical for seizure 
# |Single-trial EEG RSVP classification using convolutional neural networks, {cite}`george_single-trial_2016` |Oddball response (RSVP), groupwise (ConvNet trained on all subjects) |Time, 0.5–50 Hz | 4/3 | |  |  |Weights (spatial) |Some filter weights had expected topographic distributions for P300 <br>Others filters had large weights on areas not traditionally associated with P300
# |Wearable seizure detection using convolutional neural networks with transfer learning, {cite}`page_wearable_2016` |Seizure detection, cross-subject, within-subject, groupwise |Time,  0–128 Hz | 1-3/1-3 | | Cross-subject supervised training, within-subject finetuning of fully connected layers |Multiple: spectral features, higher order statistics + linear-SVM, RBF-SVM, ...| | 
# |Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, {cite}`bashivan_learning_2016`  |Cognitive load (number of characters to memorize), cross-subject | Frequency, mean power for 4–7 Hz, 8–13 Hz, 13–30 Hz | 3–7/2 (+ LSTM or other temporal post-processing (see design choices)) |Number of convolutional layers <br>Temporal processing of ConvNet output by max pooling, temporal convolution, LSTM or temporal convolution + LSTM | | |Inputs that maximally activate given filter <br>Activations of these inputs <br>"Deconvolution" for these inputs |Different filters were sensitive to different frequency bands <br>Later layers had more spatially localized activations <br>Learned features had noticeable links to well-known electrophysiological markers of cognitive load <br>
# |Deep Feature Learning for EEG Recordings, {cite}`stober_learning_2016` |Type of music rhythm, groupwise (ensembles of leave-one-subject-out trained models, evaluated on separate test set of same subjects) |Time, 0.5–30Hz | 2/1 | Kernel sizes | Pretraining first layer as convolutional autoencoder with different constraints |  | Weights (spatial+3 timesteps, pretrained as autoencoder) | Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings
# |Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI, {cite}`manor_convolutional_2015` |Oddball response (RSVP), within-subject |Time, 0.1–50 Hz | 3/3 (Spatio-temporal regularization) || ||Weights <br> Mean and single-trial activations |Spatiotemporal regularization led to softer peaks in weights <br>Spatial weights showed typical P300 distribution <br>Activations mostly had peaks at typical times (300-400ms)
# |Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, {cite}`sakhavi_parallel_2015`  |Imagined movement classes, within-subject |Frequency, 4–40 Hz, using FBCSP | 2/2 (Final fully connected layer uses concatenated output by convolutionaland fully connected layers) |Combination ConvNet and MLP (trained on different features) vs. only ConvNet vs. only MLP | | | | 
# |Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, {cite}`stober_using_2014` |Type of music rhythm, within-subject | Time and frequency evaluated, 0-200 Hz | 1-2/1 |Best values from automatic hyperparameter optimization: frequency cutoff, one vs two layers, kernel sizes, number of channels, pooling width |Best values from automatic hyperparameter optimization: learning rate, learning rate decay, momentum, final momentum ||
# |Convolutional deep belief networks for feature extraction of EEG signal, {cite}`ren_convolutional_2014`  |Imagined movement classes, within-subject |Frequency, 8–30 Hz |2/0 (Convolutional deep belief network, separately trained RBF-SVM classifier) | | 
# |Deep feature learning using target priors with applications in ECoG signal decoding for BCI, {cite}`wang_deep_2013`  |Finger flexion trajectory (regression), within-subject |Time, 0.15–200 Hz | 3/1 (Convolutional layers trained as convolutional stacked autoencoder with target prior) |Partially supervised CSA | 
# |Convolutional neural networks for P300 detection with application to brain-computer interfaces, {cite}`cecotti_convolutional_2011`  |Oddball / attention response using P300 speller, within-subject | Time, 0.1-20 Hz | 2/2 |Electrode subset (fixed or automatically determined) <br>Using only one spatial filter <br>Different ensembling strategies | |Multiple: Linear SVM, gradient boosting, E-SVM, S-SVM, mLVQ, LDA, ... |Weights |Spatial filters were similar for different architectures <br>Spatial filters were different (more focal, more diffuse) for different subjects
# 

# 
# |Study |  Decoding problem |  Input domain |  Conv/dense layers |  Design choices |  Training strategies |  External baseline |  Visualization type(s) |  Visualization findings|
# | :--- | --- | --- | --- | --- | --- | --- | --- | --- | 
# | This manuscript, Schirrmeister et. al (2017) |Imagined and executed movement classes, within subject |Time,  0–125 Hz | 5/1 |Different ConvNet architectures <br>Nonlinearities and pooling modes <br>Regularization and intermediate normalization layers <br>Factorized convolutions <br>Splitted vs one-step convolutions |Trial-wise vs. cropped training strategy |FBCSP + rLDA | Feature activation correlation <br>Feature-perturbation prediction correlation |See Section \ref{subsec:results-visualization}
# | Single-trial EEG classification of motor imagery using deep convolutional neural networks, citet{tang_single-trial_2017} |Imagined movement classes, within-subject |Time,  8–30 Hz | 2/2 | | | FBCSP | | |
# | EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, {cite}`lawhern_eegnet:_2016` | Oddball response (RSVP), error response (ERN), movement classes (voluntarily started and imagined) |Time, 0.1–40 Hz | 3/1 |  Kernel sizes | |   | |
# | Remembered or Forgotten? –- An EEG-Based Computational Prediction Approach, {cite}`sun_remembered_2016` | Memory performance, within-subject |Time, 0.05–15 Hz | 2/2 | | Different time windows |  |Weights (spatial) | Largest weights found over p\refrontal and temporal cortex
# |Multimodal Neural Network for Rapid Serial Visual Presentation Brain Computer Interface, {cite}`manor_multimodal_2016`|Oddball response using RSVP and image (combined image-EEG decoding), within-subject|Time, 0.3–20 Hz | 3/2 | | |  | Weights <br> Activations <br> Saliency maps by gradient |Weights showed typical P300 distribution <br>Activations were high at plausible times (300-500ms) <br>Saliency maps showed plausible spatio-temporal plots
# | A novel deep learning approach for classification of EEG motor imagery signals, {cite}`tabar_novel_2017` |Imagined and executed movement classes, within-subject |Frequency, 6–30 Hz | 1/1 | multicolumn{2}{p{0.285	extwidth}}{Addition of six-layer stacked autoencoder on ConvNet features <br> Kernel sizes} | FBCSP, Twin SVM, DDFBS, Bi-spectrum, RQNN  | Weights (spatial + frequential) |Some weights represented difference of values of two electrodes on different sides of head
# | Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, {cite}`liang_predicting_2016` |Seizure prediction, within-subject | Frequency, 0–200 Hz | 1/2 | | Different subdivisions of frequency range <br>Different lengths of time crops <br>Transfer learning with auxiliary non-epilepsy datasets || Weights <br> Clustering of weights |Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)
# | EEG-based prediction of driver's cognitive performance by deep convolutional neural network, {cite}`hajinoroozi_eeg-based_2016` |Driver performance, within- and cross-subject |Time,  1–50 Hz | 1/3 |multicolumn{2}{p{0.285	extwidth}}{Replacement of convolutional layers by restricted Boltzmann machines with slightly varied network architecture}  | |
# | Deep learning for epileptic intracranial EEG data, {cite}`antoniades_deep_2016` |Epileptic discharges, cross-subject | Time,  0–100 HZ | 1–2/2 | 1 or 2 convolutional layers |  | |Weights <br>Correlation weights and interictal epileptic discharges (IED) <br>Activations |Weights increasingly correlated with IED waveforms with increasing number of training iterations <br>Second layer captured more complex and well-defined epileptic shapes than first layer <br>IEDs led to highly synchronized activations for neighbouring electrodes
# | Learning Robust Features using Deep Learning for Automatic Seizure Detection, {cite}`thodoroff_learning_2016` |Start of epileptic seizure, within- and cross-subject |Frequency, mean amplitude for 0–7 Hz, 7–14 Hz, 14–49 Hz | 3/1 (+ LSTM as postprocessor) | | | Hand crafted features + SVM | Input occlusion and effect on prediction accuracy |Allowed to locate areas critical for seizure 
# |Single-trial EEG RSVP classification using convolutional neural networks, {cite}`george_single-trial_2016` |Oddball response (RSVP), groupwise (ConvNet trained on all subjects) |Time, 0.5–50 Hz | 4/3 | |  |  |Weights (spatial) |Some filter weights had expected topographic distributions for P300 <br>Others filters had large weights on areas not traditionally associated with P300
# |Wearable seizure detection using convolutional neural networks with transfer learning, {cite}`page_wearable_2016` |Seizure detection, cross-subject, within-subject, groupwise |Time,  0–128 Hz | 1-3/1-3 | | Cross-subject supervised training, within-subject finetuning of fully connected layers |Multiple: spectral features, higher order statistics + linear-SVM, RBF-SVM, ...| | 
# |Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, {cite}`bashivan_learning_2016`  |Cognitive load (number of characters to memorize), cross-subject | Frequency, mean power for 4–7 Hz, 8–13 Hz, 13–30 Hz | 3–7/2 (+ LSTM or other temporal post-processing (see design choices)) |Number of convolutional layers <br>Temporal processing of ConvNet output by max pooling, temporal convolution, LSTM or temporal convolution + LSTM | | |Inputs that maximally activate given filter <br>Activations of these inputs <br>"Deconvolution" for these inputs |Different filters were sensitive to different frequency bands <br>Later layers had more spatially localized activations <br>Learned features had noticeable links to well-known electrophysiological markers of cognitive load <br>
# |Deep Feature Learning for EEG Recordings, {cite}`stober_learning_2016` |Type of music rhythm, groupwise (ensembles of leave-one-subject-out trained models, evaluated on separate test set of same subjects) |Time, 0.5–30Hz | 2/1 | Kernel sizes | Pretraining first layer as convolutional autoencoder with different constraints |  | Weights (spatial+3 timesteps, pretrained as autoencoder) | Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings
# |Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI, {cite}`manor_convolutional_2015` |Oddball response (RSVP), within-subject |Time, 0.1–50 Hz | 3/3 (Spatio-temporal regularization) || ||Weights <br> Mean and single-trial activations |Spatiotemporal regularization led to softer peaks in weights <br>Spatial weights showed typical P300 distribution <br>Activations mostly had peaks at typical times (300-400ms)
# |Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, {cite}`sakhavi_parallel_2015`  |Imagined movement classes, within-subject |Frequency, 4–40 Hz, using FBCSP | 2/2 (Final fully connected layer uses concatenated output by convolutionaland fully connected layers) |Combination ConvNet and MLP (trained on different features) vs. only ConvNet vs. only MLP | | | | 
# |Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, {cite}`stober_using_2014` |Type of music rhythm, within-subject | Time and frequency evaluated, 0-200 Hz | 1-2/1 |Best values from automatic hyperparameter optimization: frequency cutoff, one vs two layers, kernel sizes, number of channels, pooling width |Best values from automatic hyperparameter optimization: learning rate, learning rate decay, momentum, final momentum ||
# |Convolutional deep belief networks for feature extraction of EEG signal, {cite}`ren_convolutional_2014`  |Imagined movement classes, within-subject |Frequency, 8–30 Hz |2/0 (Convolutional deep belief network, separately trained RBF-SVM classifier) | | 
# |Deep feature learning using target priors with applications in ECoG signal decoding for BCI, {cite}`wang_deep_2013`  |Finger flexion trajectory (regression), within-subject |Time, 0.15–200 Hz | 3/1 (Convolutional layers trained as convolutional stacked autoencoder with target prior) |Partially supervised CSA | 
# |Convolutional neural networks for P300 detection with application to brain-computer interfaces, {cite}`cecotti_convolutional_2011`  |Oddball / attention response using P300 speller, within-subject | Time, 0.1-20 Hz | 2/2 |Electrode subset (fixed or automatically determined) <br>Using only one spatial filter <br>Different ensembling strategies | |Multiple: Linear SVM, gradient boosting, E-SVM, S-SVM, mLVQ, LDA, ... |Weights |Spatial filters were similar for different architectures <br>Spatial filters were different (more focal, more diffuse) for different subjects
# |  |

# In[1]:


import re


# In[2]:


a = r"""
This manuscript, Schirrmeister et. al (2017) &
Imagined and executed movement classes, within subject &
Time, \hspace{1cm} 0--125 Hz & 5/1 &
Different ConvNet architectures \cellbr
Nonlinearities and pooling modes \cellbr
Regularization and intermediate normalization layers \cellbr
Factorized convolutions \cellbr
Splitted vs one-step convolutions &
Trial-wise vs. cropped training strategy &
FBCSP + rLDA & 
Feature activation correlation \cellbr
Feature-perturbation prediction correlation &
See Section \ref{subsec:results-visualization}
\\
\hdashline 

Single-trial EEG classification of motor imagery using deep convolutional neural networks, \citet{tang_single-trial_2017} &
Imagined movement classes, within-subject &
Time, \hspace{1cm} 8--30 Hz & 2/2 & & 
& FBCSP & &"""

b = a.replace("&", "|").replace("\n", "").replace("\cellbr", "<br>").replace('\hdashline', '\n|').replace('\\', '')
b = b.replace("ref", r"\ref").replace("--", '–')
b = re.sub(r'hspace{[^}]+}', r'', b)
b = "| " + b + " |"
print(b)


# In[3]:


a = """
EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, \citet{lawhern_eegnet:_2016} & 
Oddball response (RSVP), error response (ERN), movement classes (voluntarily started and imagined) &
Time, 0.1--40 Hz & 3/1 &  Kernel sizes & 
&   & &
\\
\hdashline
 
Remembered or Forgotten? --- An EEG-Based Computational Prediction Approach, \citet{sun_remembered_2016} & 
Memory performance, within-subject &
Time, 0.05--15 Hz & 2/2 & & Different time windows &  &
Weights (spatial) & 
Largest weights found over prefrontal and temporal cortex
\\
\hdashline

Multimodal Neural Network for Rapid Serial Visual Presentation Brain Computer Interface, \citet{manor_multimodal_2016}
&
Oddball response using RSVP and image (combined image-EEG decoding), within-subject&
Time, 0.3--20 Hz & 3/2 & & &  & 
Weights \cellbr Activations \cellbr Saliency maps by gradient &
Weights showed typical P300 distribution \cellbr
Activations were high at plausible times (300-500ms) \cellbr
Saliency maps showed plausible spatio-temporal plots
\\
\hdashline
 
A novel deep learning approach for classification of EEG motor imagery signals, \citet{tabar_novel_2017} &
Imagined and executed movement classes, within-subject &
Frequency, 6--30 Hz & 1/1 & 
\multicolumn{2}{p{0.285\textwidth}}{Addition of six-layer stacked autoencoder on ConvNet features \cellbr Kernel sizes} 
& FBCSP, Twin SVM, DDFBS, Bi-spectrum, RQNN  & Weights (spatial + frequential) &
Some weights represented difference of values of two electrodes on different sides of head
\\
\hdashline
 
Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, \citet{liang_predicting_2016} &
Seizure prediction, within-subject & Frequency, 0--200 Hz & 1/2 & & 
Different subdivisions of frequency range \cellbr
Different lengths of time crops \cellbr
Transfer learning with auxiliary non-epilepsy datasets &
& Weights \cellbr Clustering of weights &
Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)
\\
\hdashline
 
EEG-based prediction of driver's cognitive performance by deep convolutional neural network, \citet{hajinoroozi_eeg-based_2016} &
Driver performance, within- and cross-subject &
Time, \hspace{1cm} 1--50 Hz & 1/3 &
\multicolumn{2}{p{0.285\textwidth}}{Replacement of convolutional layers by restricted Boltzmann machines with slightly varied network architecture}  & 
&
\\
\hdashline
 
Deep learning for epileptic intracranial EEG data, \citet{antoniades_deep_2016} &
Epileptic discharges, cross-subject & Time, \hspace{1cm} 0--100 HZ & 1--2/2 & 1 or 2 convolutional layers &  & &
Weights \cellbr
Correlation weights and interictal epileptic discharges (IED) \cellbr
Activations &
Weights increasingly correlated with IED waveforms with increasing number of training iterations \cellbr
Second layer captured more complex and well-defined epileptic shapes than first layer \cellbr
IEDs led to highly synchronized activations for neighbouring electrodes
\\
\hdashline
 
Learning Robust Features using Deep Learning for Automatic Seizure Detection, \citet{thodoroff_learning_2016} &
Start of epileptic seizure, within- and cross-subject &
Frequency, mean amplitude for 0--7 Hz, 7--14 Hz, 14--49 Hz & 3/1 (+ LSTM as postprocessor) & &
 & Hand crafted features + SVM & Input occlusion and effect on prediction accuracy &
Allowed to locate areas critical for seizure 
\\
\hdashline

Single-trial EEG RSVP classification using convolutional neural networks, \citet{george_single-trial_2016} &
Oddball response (RSVP), groupwise (ConvNet trained on all subjects) &
Time, 0.5--50 Hz & 4/3 & &  &  &
Weights (spatial) &
Some filter weights had expected topographic distributions for P300 \cellbr
Others filters had large weights on areas not traditionally associated with P300
\\
\hdashline

Wearable seizure detection using convolutional neural networks with transfer learning, \citet{page_wearable_2016} &
Seizure detection, cross-subject, within-subject, groupwise &
Time, \hspace{1cm} 0--128 Hz & 1-3/1-3 & & Cross-subject supervised training, within-subject finetuning of fully connected layers &
Multiple: spectral features, higher order statistics + linear-SVM, RBF-SVM, ...& & 
\\
\hdashline

Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, \citet{bashivan_learning_2016}  &
Cognitive load (number of characters to memorize), cross-subject & 
Frequency, mean power for 4--7 Hz, 8--13 Hz, 13--30 Hz & 3--7/2 (+ LSTM or other temporal post-processing (see design choices)) &
Number of convolutional layers \cellbr
Temporal processing of ConvNet output by max pooling, temporal convolution, LSTM or temporal convolution + LSTM & & &
Inputs that maximally activate given filter \cellbr
Activations of these inputs \cellbr
"Deconvolution" for these inputs &
Different filters were sensitive to different frequency bands \cellbr
Later layers had more spatially localized activations \cellbr
Learned features had noticeable links to well-known electrophysiological markers of cognitive load \cellbr
\\
\hdashline

Deep Feature Learning for EEG Recordings, \citet{stober_learning_2016} &
Type of music rhythm, groupwise (ensembles of leave-one-subject-out trained models, evaluated on separate test set of same subjects) &
Time, 0.5--30Hz & 2/1 & Kernel sizes & 
Pretraining first layer as convolutional autoencoder with different constraints &  & 
Weights (spatial+3 timesteps, pretrained as autoencoder) & 
Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings
\\
\hdashline

Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI, \citet{manor_convolutional_2015} &
Oddball response (RSVP), within-subject &
Time, 0.1--50 Hz & 3/3 (Spatio-temporal regularization) && &&
Weights \cellbr Mean and single-trial activations &
Spatiotemporal regularization led to softer peaks in weights \cellbr
Spatial weights showed typical P300 distribution \cellbr
Activations mostly had peaks at typical times (300-400ms)
\\
\hdashline

Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, \citet{sakhavi_parallel_2015}  &
Imagined movement classes, within-subject &
Frequency, 4--40 Hz, using FBCSP & 2/2 (Final fully connected layer uses concatenated output by convolutional
and fully connected layers) &
Combination ConvNet and MLP (trained on different features) vs. only ConvNet vs. only MLP & 
& & & 
\\
\hdashline

Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, \citet{stober_using_2014} &
Type of music rhythm, within-subject & Time and frequency evaluated, 0-200 Hz & 1-2/1 &
Best values from automatic hyperparameter optimization: frequency cutoff, one vs two layers, kernel sizes, number of channels, pooling width &
Best values from automatic hyperparameter optimization: learning rate, learning rate decay, momentum, final momentum &
&
\\
\hdashline

Convolutional deep belief networks for feature extraction of EEG signal, \citet{ren_convolutional_2014}  &
Imagined movement classes, within-subject &
Frequency, 8--30 Hz &
2/0 (Convolutional deep belief network, separately trained RBF-SVM classifier) & & 
\\
\hdashline

Deep feature learning using target priors with applications in ECoG signal decoding for BCI, \citet{wang_deep_2013}  &
Finger flexion trajectory (regression), within-subject &
Time, 0.15--200 Hz & 3/1 (Convolutional layers trained as convolutional stacked autoencoder with target prior) &
Partially supervised CSA & 
\\
\hdashline

Convolutional neural networks for P300 detection with application to brain-computer interfaces, \citet{cecotti_convolutional_2011}  &
Oddball / attention response using P300 speller, within-subject & Time, 0.1-20 Hz & 2/2 &
Electrode subset (fixed or automatically determined) \cellbr
Using only one spatial filter \cellbr
Different ensembling strategies & 
&
Multiple: Linear SVM, gradient boosting, E-SVM, S-SVM, mLVQ, LDA, ... &
Weights &
Spatial filters were similar for different architectures \cellbr
Spatial filters were different (more focal, more diffuse) for different subjects
\\
\hdashline
 """

b = a.replace("&", "|").replace("\n", "").replace("\cellbr", "<br>").replace('\hdashline', '\n|').replace('\\', '')
b = b.replace("ref", r"\ref").replace("--", '–').replace("cite", r"\cite").replace("citet", r"cite")
b = re.sub(r'hspace{[^}]+}', r'', b)
b = re.sub(r"\\cite{([^}]+)}", "{cite}`\g<1>`", b)
b = "| " + b + " |"
print(b)


# In[4]:


a =  "network, \cite{hajinoroozi_eeg-based_2016} |D"


# In[5]:


re.sub(r"\\cite{([^}]+)}", "{cite}`\g<1>`", a)


# In[45]:


a = """
| This manuscript, Schirrmeister et. al (2017) |Imagined and executed movement classes, within subject |Time,  0–125 Hz | 5/1 |Different ConvNet architectures <br>Nonlinearities and pooling modes <br>Regularization and intermediate normalization layers <br>Factorized convolutions <br>Splitted vs one-step convolutions |Trial-wise vs. cropped training strategy |FBCSP + rLDA | Feature activation correlation <br>Feature-perturbation prediction correlation |See Section \ref{subsec:results-visualization}
| Single-trial EEG classification of motor imagery using deep convolutional neural networks, citet{tang_single-trial_2017} |Imagined movement classes, within-subject |Time,  8–30 Hz | 2/2 | | | FBCSP | | 
| EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, {cite}`lawhern_eegnet:_2016` | Oddball response (RSVP), error response (ERN), movement classes (voluntarily started and imagined) |Time, 0.1–40 Hz | 3/1 |  Kernel sizes | |   | |
| Remembered or Forgotten? –- An EEG-Based Computational Prediction Approach, {cite}`sun_remembered_2016` | Memory performance, within-subject |Time, 0.05–15 Hz | 2/2 | | Different time windows |  |Weights (spatial) | Largest weights found over p\refrontal and temporal cortex
|Multimodal Neural Network for Rapid Serial Visual Presentation Brain Computer Interface, {cite}`manor_multimodal_2016`|Oddball response using RSVP and image (combined image-EEG decoding), within-subject|Time, 0.3–20 Hz | 3/2 | | |  | Weights <br> Activations <br> Saliency maps by gradient |Weights showed typical P300 distribution <br>Activations were high at plausible times (300-500ms) <br>Saliency maps showed plausible spatio-temporal plots
| A novel deep learning approach for classification of EEG motor imagery signals, {cite}`tabar_novel_2017` |Imagined and executed movement classes, within-subject |Frequency, 6–30 Hz | 1/1 |Addition of six-layer stacked autoencoder on ConvNet features <br> Kernel sizes} |  | FBCSP, Twin SVM, DDFBS, Bi-spectrum, RQNN  | Weights (spatial + frequential) |Some weights represented difference of values of two electrodes on different sides of head
| Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, {cite}`liang_predicting_2016` |Seizure prediction, within-subject | Frequency, 0–200 Hz | 1/2 | | Different subdivisions of frequency range <br>Different lengths of time crops <br>Transfer learning with auxiliary non-epilepsy datasets || Weights <br> Clustering of weights |Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)
| EEG-based prediction of driver's cognitive performance by deep convolutional neural network, {cite}`hajinoroozi_eeg-based_2016` |Driver performance, within- and cross-subject |Time,  1–50 Hz | 1/3 |Replacement of convolutional layers by restricted Boltzmann machines with slightly varied network architecture}  | | |
| Deep learning for epileptic intracranial EEG data, {cite}`antoniades_deep_2016` |Epileptic discharges, cross-subject | Time,  0–100 HZ | 1–2/2 | 1 or 2 convolutional layers |  | |Weights <br>Correlation weights and interictal epileptic discharges (IED) <br>Activations |Weights increasingly correlated with IED waveforms with increasing number of training iterations <br>Second layer captured more complex and well-defined epileptic shapes than first layer <br>IEDs led to highly synchronized activations for neighbouring electrodes
| Learning Robust Features using Deep Learning for Automatic Seizure Detection, {cite}`thodoroff_learning_2016` |Start of epileptic seizure, within- and cross-subject |Frequency, mean amplitude for 0–7 Hz, 7–14 Hz, 14–49 Hz | 3/1 (+ LSTM as postprocessor) | | | Hand crafted features + SVM | Input occlusion and effect on prediction accuracy |Allowed to locate areas critical for seizure 
|Single-trial EEG RSVP classification using convolutional neural networks, {cite}`george_single-trial_2016` |Oddball response (RSVP), groupwise (ConvNet trained on all subjects) |Time, 0.5–50 Hz | 4/3 | |  |  |Weights (spatial) |Some filter weights had expected topographic distributions for P300 <br>Others filters had large weights on areas not traditionally associated with P300
|Wearable seizure detection using convolutional neural networks with transfer learning, {cite}`page_wearable_2016` |Seizure detection, cross-subject, within-subject, groupwise |Time,  0–128 Hz | 1-3/1-3 | | Cross-subject supervised training, within-subject finetuning of fully connected layers |Multiple: spectral features, higher order statistics + linear-SVM, RBF-SVM, ...| | 
|Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, {cite}`bashivan_learning_2016`  |Cognitive load (number of characters to memorize), cross-subject | Frequency, mean power for 4–7 Hz, 8–13 Hz, 13–30 Hz | 3–7/2 (+ LSTM or other temporal post-processing (see design choices)) |Number of convolutional layers <br>Temporal processing of ConvNet output by max pooling, temporal convolution, LSTM or temporal convolution + LSTM | | |Inputs that maximally activate given filter <br>Activations of these inputs <br>"Deconvolution" for these inputs |Different filters were sensitive to different frequency bands <br>Later layers had more spatially localized activations <br>Learned features had noticeable links to well-known electrophysiological markers of cognitive load <br>
|Deep Feature Learning for EEG Recordings, {cite}`stober_learning_2016` |Type of music rhythm, groupwise (ensembles of leave-one-subject-out trained models, evaluated on separate test set of same subjects) |Time, 0.5–30Hz | 2/1 | Kernel sizes | Pretraining first layer as convolutional autoencoder with different constraints |  | Weights (spatial+3 timesteps, pretrained as autoencoder) | Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings
|Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI, {cite}`manor_convolutional_2015` |Oddball response (RSVP), within-subject |Time, 0.1–50 Hz | 3/3 (Spatio-temporal regularization) || ||Weights <br> Mean and single-trial activations |Spatiotemporal regularization led to softer peaks in weights <br>Spatial weights showed typical P300 distribution <br>Activations mostly had peaks at typical times (300-400ms)
|Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, {cite}`sakhavi_parallel_2015`  |Imagined movement classes, within-subject |Frequency, 4–40 Hz, using FBCSP | 2/2 (Final fully connected layer uses concatenated output by convolutionaland fully connected layers) |Combination ConvNet and MLP (trained on different features) vs. only ConvNet vs. only MLP | | | | 
|Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, {cite}`stober_using_2014` |Type of music rhythm, within-subject | Time and frequency evaluated, 0-200 Hz | 1-2/1 |Best values from automatic hyperparameter optimization: frequency cutoff, one vs two layers, kernel sizes, number of channels, pooling width |Best values from automatic hyperparameter optimization: learning rate, learning rate decay, momentum, final momentum ||
|Convolutional deep belief networks for feature extraction of EEG signal, {cite}`ren_convolutional_2014`  |Imagined movement classes, within-subject |Frequency, 8–30 Hz |2/0 (Convolutional deep belief network, separately trained RBF-SVM classifier) | | 
|Deep feature learning using target priors with applications in ECoG signal decoding for BCI, {cite}`wang_deep_2013`  |Finger flexion trajectory (regression), within-subject |Time, 0.15–200 Hz | 3/1 (Convolutional layers trained as convolutional stacked autoencoder with target prior) |Partially supervised CSA | 
|Convolutional neural networks for P300 detection with application to brain-computer interfaces, {cite}`cecotti_convolutional_2011`  |Oddball / attention response using P300 speller, within-subject | Time, 0.1-20 Hz | 2/2 |Electrode subset (fixed or automatically determined) <br>Using only one spatial filter <br>Different ensembling strategies | |Multiple: Linear SVM, gradient boosting, E-SVM, S-SVM, mLVQ, LDA, ... |Weights |Spatial filters were similar for different architectures <br>Spatial filters were different (more focal, more diffuse) for different subjects
"""


# In[46]:


parts = [l.split('|')[1:] for l in a.split('\n')[1:-1]]


# In[47]:


[len(p) for p in parts]


# In[48]:


headings = ['Study', 'Decoding problem', 'Input domain', 'Conv/dense layers', 'Design choices', 
'Training strategies', 'External baseline', 'Visualization type(s)', 'Visualization findings']


# In[49]:


parts = [p + [''] * (9 - len(p)) for p in parts]


# In[50]:


list(zip(headings, parts[3]))


# In[51]:


[len(p) for p in parts]


# In[52]:


import numpy as np
part_arr = np.array(parts)


# In[53]:


import pandas as pd
import re


# In[54]:


df  = pd.DataFrame(data=part_arr, columns=headings)


# In[55]:


print(df.loc[:, ['Study', 'Design choices', 'Training strategies']].to_markdown(showindex=False))


# | Study                                                                                                                              | Design choices                                                                                                                                                                                  | Training strategies                                                                                                                      |
# |:-----------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------|
# | EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, {cite}`lawhern_eegnet:_2016`                      | Kernel sizes                                                                                                                                                                                    |                                                                                                                                          |
# | Remembered or Forgotten? –- An EEG-Based Computational Prediction Approach, {cite}`sun_remembered_2016`                            |                                                                                                                                                                                                 | Different time windows                                                                                                                   |
# | A novel deep learning approach for classification of EEG motor imagery signals, {cite}`tabar_novel_2017`                           | Addition of six-layer stacked autoencoder on ConvNet features <br> Kernel sizes}                                                                                                                |                                                                                                                                          |
# | Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, {cite}`liang_predicting_2016`           |                                                                                                                                                                                                 | Different subdivisions of frequency range <br>Different lengths of time crops <br>Transfer learning with auxiliary non-epilepsy datasets |
# | EEG-based prediction of driver's cognitive performance by deep convolutional neural network, {cite}`hajinoroozi_eeg-based_2016`    | Replacement of convolutional layers by restricted Boltzmann machines with slightly varied network architecture}                                                                                 |                                                                                                                                          |
# | Deep learning for epileptic intracranial EEG data, {cite}`antoniades_deep_2016`                                                    | 1 or 2 convolutional layers                                                                                                                                                                     |                                                                                                                                          |
# | Wearable seizure detection using convolutional neural networks with transfer learning, {cite}`page_wearable_2016`                  |                                                                                                                                                                                                 | Cross-subject supervised training, within-subject finetuning of fully connected layers                                                   |
# | Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, {cite}`bashivan_learning_2016`                | Number of convolutional layers <br>Temporal processing of ConvNet output by max pooling, temporal convolution, LSTM or temporal convolution + LSTM                                              |                                                                                                                                          |
# | Deep Feature Learning for EEG Recordings, {cite}`stober_learning_2016`                                                             | Kernel sizes                                                                                                                                                                                    | Pretraining first layer as convolutional autoencoder with different constraints                                                          |
# | Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, {cite}`sakhavi_parallel_2015`                       | Combination ConvNet and MLP (trained on different features) vs. only ConvNet vs. only MLP                                                                                                       |                                                                                                                                          |
# | Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, {cite}`stober_using_2014`  | Best values from automatic hyperparameter optimization: frequency cutoff, one vs two layers, kernel sizes, number of channels, pooling width                                                    | Best values from automatic hyperparameter optimization: learning rate, learning rate decay, momentum, final momentum                     |
# | Deep feature learning using target priors with applications in ECoG signal decoding for BCI, {cite}`wang_deep_2013`                | Partially supervised CSA                                                                                                                                                                        |                                                                                                                                          |
# | Convolutional neural networks for P300 detection with application to brain-computer interfaces, {cite}`cecotti_convolutional_2011` | Electrode subset (fixed or automatically determined) <br>Using only one spatial filter <br>Different ensembling strategies                                                                      |                                                                                                                                          |
# 
# 

# In[20]:


df


# In[56]:


print(df.loc[1:, ['Study', 'Visualization type(s)', 'Visualization findings']].to_markdown(showindex=False))


# In[65]:


b = """| Study                                                                                                                              | Visualization type(s)                                                                                            | Visualization findings                                                                                                                                                                                                                                                          |
|:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Single-trial EEG classification of motor imagery using deep convolutional neural networks, citet{tang_single-trial_2017}           |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, {cite}`lawhern_eegnet:_2016`                      |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Remembered or Forgotten? –- An EEG-Based Computational Prediction Approach, {cite}`sun_remembered_2016`                            | Weights (spatial)                                                                                                | Largest weights found over prefrontal and temporal cortex                                                                                                                                                                                                                                                                    |
| Multimodal Neural Network for Rapid Serial Visual Presentation Brain Computer Interface, {cite}`manor_multimodal_2016`             | Weights <br> Activations <br> Saliency maps by gradient                                                          | Weights showed typical P300 distribution <br>Activations were high at plausible times (300-500ms) <br>Saliency maps showed plausible spatio-temporal plots                                                                                                                      |
| A novel deep learning approach for classification of EEG motor imagery signals, {cite}`tabar_novel_2017`                           | Weights (spatial + frequential)                                                                                  | Some weights represented difference of values of two electrodes on different sides of head                                                                                                                                                                                      |
| Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, {cite}`liang_predicting_2016`           | Weights <br> Clustering of weights                                                                               | Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)                                                                                                                                                                                |
| EEG-based prediction of driver's cognitive performance by deep convolutional neural network, {cite}`hajinoroozi_eeg-based_2016`    |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Deep learning for epileptic intracranial EEG data, {cite}`antoniades_deep_2016`                                                    | Weights <br>Correlation weights and interictal epileptic discharges (IED) <br>Activations                        | Weights increasingly correlated with IED waveforms with increasing number of training iterations <br>Second layer captured more complex and well-defined epileptic shapes than first layer <br>IEDs led to highly synchronized activations for neighbouring electrodes          |
| Learning Robust Features using Deep Learning for Automatic Seizure Detection, {cite}`thodoroff_learning_2016`                      | Input occlusion and effect on prediction accuracy                                                                | Allowed to locate areas critical for seizure                                                                                                                                                                                                                                    |
| Single-trial EEG RSVP classification using convolutional neural networks, {cite}`george_single-trial_2016`                         | Weights (spatial)                                                                                                | Some filter weights had expected topographic distributions for P300 <br>Others filters had large weights on areas not traditionally associated with P300                                                                                                                        |
| Wearable seizure detection using convolutional neural networks with transfer learning, {cite}`page_wearable_2016`                  |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, {cite}`bashivan_learning_2016`                | Inputs that maximally activate given filter <br>Activations of these inputs <br>"Deconvolution" for these inputs | Different filters were sensitive to different frequency bands <br>Later layers had more spatially localized activations <br>Learned features had noticeable links to well-known electrophysiological markers of cognitive load <br>                                             |
| Deep Feature Learning for EEG Recordings, {cite}`stober_learning_2016`                                                             | Weights (spatial+3 timesteps, pretrained as autoencoder)                                                         | Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings |
| Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI, {cite}`manor_convolutional_2015`             | Weights <br> Mean and single-trial activations                                                                   | Spatiotemporal regularization led to softer peaks in weights <br>Spatial weights showed typical P300 distribution <br>Activations mostly had peaks at typical times (300-400ms)                                                                                                 |
| Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, {cite}`sakhavi_parallel_2015`                       |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, {cite}`stober_using_2014`  |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Convolutional deep belief networks for feature extraction of EEG signal, {cite}`ren_convolutional_2014`                            |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Deep feature learning using target priors with applications in ECoG signal decoding for BCI, {cite}`wang_deep_2013`                |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
| Convolutional neural networks for P300 detection with application to brain-computer interfaces, {cite}`cecotti_convolutional_2011` | Weights                                                                                                          | Spatial filters were similar for different architectures <br>Spatial filters were different (more focal, more diffuse) for different subjects                                                                                                                                   |"""


c = b.split('\n')[2:]

print('\n'.join(['|' + l[l.index('cite') - 1:] for l in c]))


# ### visulization table with empty papers

# | Study                                                                                                                              | Visualization type(s)                                                                                            | Visualization findings   |
# |:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|:-------------------------|
# | {cite}`tang_single-trial_2017`          |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`lawhern_eegnet:_2016`                      |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`sun_remembered_2016`                            | Weights (spatial)                                                                                                | Largest weights found over prefrontal and temporal cortex                                                                                                                                                                                                                                                                    |
# |{cite}`manor_multimodal_2016`             | Weights <br> Activations <br> Saliency maps by gradient                                                          | Weights showed typical P300 distribution <br>Activations were high at plausible times (300-500ms) <br>Saliency maps showed plausible spatio-temporal plots                                                                                                                      |
# |{cite}`tabar_novel_2017`                           | Weights (spatial + frequential)                                                                                  | Some weights represented difference of values of two electrodes on different sides of head                                                                                                                                                                                      |
# |{cite}`liang_predicting_2016`           | Weights <br> Clustering of weights                                                                               | Clusters of weights showed typical frequency band subdivision (delta, theta, alpha, beta, gamma)                                                                                                                                                                                |
# |{cite}`hajinoroozi_eeg-based_2016`    |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`antoniades_deep_2016`                                                    | Weights <br>Correlation weights and interictal epileptic discharges (IED) <br>Activations                        | Weights increasingly correlated with IED waveforms with increasing number of training iterations <br>Second layer captured more complex and well-defined epileptic shapes than first layer <br>IEDs led to highly synchronized activations for neighbouring electrodes          |
# |{cite}`thodoroff_learning_2016`                      | Input occlusion and effect on prediction accuracy                                                                | Allowed to locate areas critical for seizure                                                                                                                                                                                                                                    |
# |{cite}`george_single-trial_2016`                         | Weights (spatial)                                                                                                | Some filter weights had expected topographic distributions for P300 <br>Others filters had large weights on areas not traditionally associated with P300                                                                                                                        |
# |{cite}`page_wearable_2016`                  |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`bashivan_learning_2016`                | Inputs that maximally activate given filter <br>Activations of these inputs <br>"Deconvolution" for these inputs | Different filters were sensitive to different frequency bands <br>Later layers had more spatially localized activations <br>Learned features had noticeable links to well-known electrophysiological markers of cognitive load <br>                                             |
# |{cite}`stober_learning_2016`                                                             | Weights (spatial+3 timesteps, pretrained as autoencoder)                                                         | Different constraints led to different weights, one type of constraints could enforce weights that are similar across subjects; other type of constraints led to weights that have similar spatial topographies under different architectural configurations and preprocessings |
# |{cite}`manor_convolutional_2015`             | Weights <br> Mean and single-trial activations                                                                   | Spatiotemporal regularization led to softer peaks in weights <br>Spatial weights showed typical P300 distribution <br>Activations mostly had peaks at typical times (300-400ms)                                                                                                 |
# |{cite}`sakhavi_parallel_2015`                       |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`stober_using_2014`  |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`ren_convolutional_2014`                            |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`wang_deep_2013`                |                                                                                                                  |                                                                                                                                                                                                                                                                                 |
# |{cite}`cecotti_convolutional_2011` | Weights                                                                                                          | Spatial filters were similar for different architectures <br>Spatial filters were different (more focal, more diffuse) for different subjects                                                                                                                                   |

# In[45]:


a = df.loc[:,'Input domain']

import numpy as np
a = np.array(a)[1:] # exclude our own study


# In[56]:


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


# In[ ]:





# In[46]:


import re
domain_strings = [s.split(',')[0] for s in a]
start_fs = [float(re.sub(r'[a-z ]+',r'', re.split(r'[–-–-]'," ".join(s.split(',')[1:]))[0])) for s in a]
end_fs = [float(re.sub(r'[a-z HZFBCSP]+',r'', re.split(r'[–-–-]'," ".join(s.split(',')[1:]))[1])) for s in a]


# In[50]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn
seaborn.set_palette('colorblind')
seaborn.set_style('darkgrid')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
#matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14


# In[53]:


freq_mask = np.array(['freq' in s.lower() for s in domain_strings])
time_mask = np.array(['time' in s.lower() for s in domain_strings])


# In[54]:


rng = np.random.RandomState(98349384)
color = 'grey'
domain_strings = np.array(domain_strings)
start_fs = np.array(start_fs)
end_fs = np.array(end_fs)
i_sort = np.flatnonzero(time_mask)[np.argsort(end_fs[time_mask])]
for i, (d,s,e) in enumerate(zip(
        domain_strings[i_sort], start_fs[i_sort], end_fs[i_sort])):
    offset = 0.6*i/len(i_sort) - 0.3
    plt.plot([offset,offset] , [s, e], marker='o', alpha=1, color=color, ls=':')
i_sort = np.flatnonzero(freq_mask)[np.argsort(end_fs[freq_mask])]
for i, (d,s,e) in enumerate(zip(
        domain_strings[i_sort], start_fs[i_sort], end_fs[i_sort])):
    offset = 0.6*i/len(i_sort) + 0.7
    plt.plot([offset,offset] , [s, e], marker='o', alpha=1, color=color, ls=':')

plt.xlim(-0.5,1.5)
plt.xlabel("Input domain")
plt.ylabel("Frequency [Hz]")
plt.xticks([0,1], ["Time", "Frequency"], rotation=45)
plt.title("Input domains and frequency ranges in prior work", y=1.05)


# In[ ]:


1


# In[ ]:


rng = np.random.RandomState(98349384)
color = 'grey'
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


# ## layer stuff

# In[ ]:


ls = np.array(df.loc[:, 'Conv/dense layers'])[1:] # exclude ourstudy


# In[ ]:


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


# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn
seaborn.set_palette('colorblind')
seaborn.set_style('darkgrid')

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")
#matplotlib.rcParams['figure.figsize'] = (12.0, 1.0)
matplotlib.rcParams['font.size'] = 14


# In[ ]:


all_conv_ls = np.concatenate([np.arange(low_c, high_c+1) for low_c, high_c in zip(low_conv_ls, high_conv_ls)])
all_dense_ls = np.concatenate([np.arange(low_c, high_c+1) for low_c, high_c in zip(low_dense_ls, high_dense_ls)])
bincount_conv = np.bincount(all_conv_ls)
bincount_dense = np.bincount(all_dense_ls)


# In[ ]:


rng = np.random.RandomState(98349384)
color = 'grey'
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


# In[ ]:





# In[ ]:


print(df.loc[:, ['Study', 'Decoding problem', 'External baseline']].to_markdown(showindex=False))


# | Study                                                                                                                              | Decoding problem                                                                                                                     | External baseline                                                               |
# |:-----------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------|
# | This manuscript, Schirrmeister et. al (2017)                                                                                       | Imagined and executed movement classes, within subject                                                                               | FBCSP + rLDA                                                                    |
# | Single-trial EEG classification of motor imagery using deep convolutional neural networks, citet{tang_single-trial_2017}           | Imagined movement classes, within-subject                                                                                            | FBCSP                                                                           |
# | EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces, {cite}`lawhern_eegnet:_2016`                      | Oddball response (RSVP), error response (ERN), movement classes (voluntarily started and imagined)                                   |                                                                                 |
# | Remembered or Forgotten? –- An EEG-Based Computational Prediction Approach, {cite}`sun_remembered_2016`                            | Memory performance, within-subject                                                                                                   |                                                                                 |
# | Multimodal Neural Network for Rapid Serial Visual Presentation Brain Computer Interface, {cite}`manor_multimodal_2016`             | Oddball response using RSVP and image (combined image-EEG decoding), within-subject                                                  |                                                                                 |
# | A novel deep learning approach for classification of EEG motor imagery signals, {cite}`tabar_novel_2017`                           | Imagined and executed movement classes, within-subject                                                                               | Weights (spatial + frequential)                                                 |
# | Predicting Seizures from Electroencephalography Recordings: A Knowledge Transfer Strategy, {cite}`liang_predicting_2016`           | Seizure prediction, within-subject                                                                                                   |                                                                                 |
# | EEG-based prediction of driver's cognitive performance by deep convolutional neural network, {cite}`hajinoroozi_eeg-based_2016`    | Driver performance, within- and cross-subject                                                                                        |                                                                                 |
# | Deep learning for epileptic intracranial EEG data, {cite}`antoniades_deep_2016`                                                    | Epileptic discharges, cross-subject                                                                                                  |                                                                                 |
# | Learning Robust Features using Deep Learning for Automatic Seizure Detection, {cite}`thodoroff_learning_2016`                      | Start of epileptic seizure, within- and cross-subject                                                                                | Hand crafted features + SVM                                                     |
# | Single-trial EEG RSVP classification using convolutional neural networks, {cite}`george_single-trial_2016`                         | Oddball response (RSVP), groupwise (ConvNet trained on all subjects)                                                                 |                                                                                 |
# | Wearable seizure detection using convolutional neural networks with transfer learning, {cite}`page_wearable_2016`                  | Seizure detection, cross-subject, within-subject, groupwise                                                                          | Multiple: spectral features, higher order statistics + linear-SVM, RBF-SVM, ... |
# | Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks, {cite}`bashivan_learning_2016`                | Cognitive load (number of characters to memorize), cross-subject                                                                     |                                                                                 |
# | Deep Feature Learning for EEG Recordings, {cite}`stober_learning_2016`                                                             | Type of music rhythm, groupwise (ensembles of leave-one-subject-out trained models, evaluated on separate test set of same subjects) |                                                                                 |
# | Convolutional Neural Network for Multi-Category Rapid Serial Visual Presentation BCI, {cite}`manor_convolutional_2015`             | Oddball response (RSVP), within-subject                                                                                              |                                                                                 |
# | Parallel Convolutional-Linear Neural Network for Motor Imagery Classification, {cite}`sakhavi_parallel_2015`                       | Imagined movement classes, within-subject                                                                                            |                                                                                 |
# | Using Convolutional Neural networks to Recognize Rhythm Stimuli form Electroencephalography Recordings, {cite}`stober_using_2014`  | Type of music rhythm, within-subject                                                                                                 |                                                                                 |
# | Convolutional deep belief networks for feature extraction of EEG signal, {cite}`ren_convolutional_2014`                            | Imagined movement classes, within-subject                                                                                            |                                                                                 |
# | Deep feature learning using target priors with applications in ECoG signal decoding for BCI, {cite}`wang_deep_2013`                | Finger flexion trajectory (regression), within-subject                                                                               |                                                                                 |
# | Convolutional neural networks for P300 detection with application to brain-computer interfaces, {cite}`cecotti_convolutional_2011` | Oddball / attention response using P300 speller, within-subject                                                                      | Multiple: Linear SVM, gradient boosting, E-SVM, S-SVM, mLVQ, LDA, ...           |
