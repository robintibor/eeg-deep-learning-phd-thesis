#!/usr/bin/env python
# coding: utf-8

# (perturbation-visualization)=
# # Perturbation Visualization

# Understanding what features the network has learned to extract from the EEG signal is scientifically interesting. Due to the end-to-end training, the networks may learn a variety of features to solve their task. The networks may learn task-informative brain-signal features or task-informative non-brain-signal features, e.g., eye movements that correlate to a movement. The learned features may be already known from prior research on brain-signal decoding or represent novel features that had not been described in the literature. However, there is no straightforward way to find out what the deep networks have learned from the brain signals.

# Therefore, we developed an input amplitude perturbation method to investigate in how far the deep networks learn to extract spectral amplitude features, which are very commonly used in many EEG decoding pipelines. For example, it is known that the amplitudes, for example of the alpha, beta and gamma bands, provide class-discriminative information for motor tasks [Ball et al., 2008; Pfurtscheller, 1981; Pfurtscheller and Aranibar, 1979]. Hence, it seems a priori very likely that the deep networks learn to extract such features and worthwhile to check whether they indeed do so.

# ## Input-perturbation network-prediction correlation map
# 
# > To investigate the causal effect of changes in power on the deep ConvNet, we correlated changes in ConvNet predictions with changes in amplitudes by perturbing the original trial amplitudes (see Fig. 8 for an overview). Concretely, we transformed all training trials into the frequency domain by a Fourier transformation. Then we randomly perturbed the amplitudes by adding Gaussian noise (with mean 0 and variance 1) to them. The phases were kept unperturbed. After the perturbation, we retransformed to the time domain by the inverse Fourier transformation. We computed predictions of the deep ConvNet for these trials before and after the perturbation (predictions here refers to outputs of the ConvNet directly before the softmax activation). We repeated this procedure with 400 perturbations sampled from aforementioned Gaussian distribution and then correlated the change in input amplitudes (i.e., the perturbation/noise we added) with the change in the ConvNet predictions. To ensure that the effects of our perturbations reflect the behavior of the ConvNet on realistic data, we also checked that the perturbed input does not cause the ConvNet to misclassify the trials (as can easily happen even from small perturbations, see Szegedy et al. [2014]). For that, we computed accuracies on the perturbed trials. For all perturbations of the training sets of all subjects, accuracies stayed above 99.5% of the accuracies achieved with the unperturbed data.

# ![title](images/input-perturbation-overview.png)

# ```{figure} images/input-perturbation-overview.png
# ---
# name: input-perturbation-overview-figure
# width: 70%
# ---
# **Computation overview for input-perturbation network-prediction correlation map.**  (a) Example spectral amplitude perturbation and resulting classification difference. Top: Spectral amplitude perturbation as used to perturb the trials. Bottom: unit-output difference between unperturbed and perturbed trials for the classification layer units before the softmax. (b) Input-perturbation network-prediction correlations and corresponding network correlation scalp map for alpha band. Left: Correlation coefficients between spectral amplitude perturbations for all frequency bins and differences of the unit outputs for the four classes (differences between unperturbed and perturbed trials) for one electrode. Middle: Mean of the correlation coefficients over the the alpha (7–13 Hz), beta (13–31 Hz) and gamma (71–91 Hz) frequency ranges. Right: An exemplary scalp map for the alpha band, where the color of each dot encodes the correlation of amplitude changes at that electrode and the corresponding prediction changes of the ConvNet. Negative correlations on the left sensorimotor hand/arm areas show an amplitude decrease in these areas leads to a prediction increase for the Hand (R) class, whereas positive correlations on the right sensorimotor hand/arm areas show an amplitude decrease leads to a prediction decrease for the Hand (R) class. This complements the information from the input-feature unit-output network correlation map (Fig. 6b), which showed band power in these areas is strongly correlated with unit outputs in the penultimate layer.

# ## Gradient-based implementation

# An simpler way to implement the idea of testing the sensitivity of the network to spectral amplitude features is through gradient-based analysis [^krm]. There, we directly compute the gradient of the output unit with respect to the amplitudes of all frequency bins of all electrodes of the original unperturbed trial. To practically implement this, one must first transform the time domain input signal into the frequency domain and to a amplitude/phase representation via the Fourier transform. Then, one can transform the amplitude/phase representation back to the time domain using the inverse Fourier Transform. Since the inverse Fourier transform is differentiable one can backpropagate the gradients from the output unit through the time domain input back to the amplitudes. The more the network behaves locally linear around the input, the closer the results of this variant would be to the original perturbation variant. It is computationally substantially faster as everything is computed in one forward-backward pass, without needing to iterate over many perturbations.
# 
# 
# [^krm]: This idea was suggested to us in personal communication by Klaus-Robert Müller

# ## Interpretation and limitations
# 
# The perturbation-based visualization reflects network behavior and one cannot directly draw inferences about the data-generating process from them. This is because a prediction change caused by an amplitude perturbation may reflect both learned task-relevant factors as well as learned noise correlations. For example, increasing the amplitude in the alpha frequency range at C4, an electrode on the right side, may increase the predicted probability for right hand movement. That would likely not be because the alpha amplitude actually increases at C4 during right hand movement, but because the amplitude *decreases* on C3 *and* is correlated between C3 and C4. Hence, first subtracting the C4 amplitude from the C3 amplitude and then decoding negative values of this computation as indicating right hand movement is a reasonable learned prediction function. And this learned prediction function would cause the amplitude-perturbation function to show that an alpha increase at C4 causes an increase in the predicted probability for right hand movement. For a more detailed discussion of this effect in the context of linear models, see {cite}`haufe_interpretation_2014`.
# 
# 

# 
