#!/usr/bin/env python
# coding: utf-8

# (introduction)=
# # Introduction

# ```{admonition} Deep Learning (DL) is a very promising method to decode brain signals from EEG
# * Deep learning may extract different information from EEG signals then humans are able to
# * Deep learning may improve EEG-based diagnosis, enable new assistive technologies and advance scientific understanding of the EEG signal
# * Deep learning ... (Results this thesis)
# ```
# 
# 
# Machine learning (ML), i.e., using data to train programs to solve tasks, has the potential to benefit medical applications. Compared to humans processing medical data, machine-learning programs can process larger amounts of data and extract different information. For example, machine-learning algorithms have been used to process the large amounts of signals recorded in an intensive-care unit to predict kidney failure, diagnosed breast cancer using high-frequency features usually ignored by doctors, translated brain signals into control signals for external devices in real-time and detected pathology from long brain signal recordings. Used in this way, machine learning can improve medical interventions, enable new assistive devices and advance scientific understanding.
# 
# TODO:refs
# 
# 
# 
# 
# 

# Brain-signal decoding is an especially interesting problem to tackle with machine learning. Brain signals contain a lot of information, yet are hard to interpret for humans. Additionally, more brain signals can be recorded than humans could possibly process. Machine-learning algorithms can help doctors triage patients by quickly detecting stroke biomarkers from computed tomography (CT) and enable brain-computer interfaces by recognizing people's intentions from electroencephalographic (EEG) in real time. Also, as brain signals are far from being fully understood, machine-learning-based algorithms have the potential to advance scientific understanding by finding new brain-signal biomarkers for different pathologies.
# 
# % [maybe ref ML for scientific discovery]

# Electroencephalographic (EEG) recordings that measure the electric signals produced by the brain, are a very suitable signal type for machine learning. Large EEG datasets can be created for training ML algorithms as EEG recordings are fairly cheap to acquire and can be recorded without substantial side effects.  Furthermore, EEG signals are especially hard for humans to read, making them a promising target to extract information via machine learning. Some uses of EEG such as diagnosis of pathologies may be improved by applying machine learning, while others such as brain-computer interfaces are only possible through machine learning. Finally, since the information contained in the EEG signal is far from being fully understood, machine learning may even help understand the EEG signal itself better.
# 

# Deep learning is a very promising approach for brain-signal decoding from EEG. The umbrella term deep learning describes brain-inspired machine-learning models with multiple computational stages, where the computational stages are typically trained jointly to solve a given task. Deep neural networks (DNNs) are deep learning models that are  inspired by computational structures in the brain and have in recent times become the most successful models for a wide variety of tasks, including object detection in images, speech recognition from audio or machine translation.  DNNs only have very general assumptions about the properties of their training signals embedded into them (such as smoothness, locality, hierarchical structure) and have shown great success on a variety of natural signals. Therefore, DNNs are very promising to apply to hard-to-understand natural signals like EEG signals.

# Prior to the work presented in this thesis, it was unclear how well DNN architectures can decode EEG signals compared to hand-engineered feature-based approaches. The high dimensionality, low signal-to-noise ratio and large signal variability (e.g., from person to person or even session to session for the same person) are among the challenges that may  favor feature-based approaches that exploit more specific assumptions about the EEG signal. While there had been a long history of applying deep learning to EEG signals, a more systematic study of the performance of modern DNNs on EEG signal decoding, including effects of different design choices had been missing.

# (Results and structure, also mention visualization)

# In[ ]:




