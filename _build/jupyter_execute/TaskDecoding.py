#!/usr/bin/env python
# coding: utf-8

# (task-related)=
# # Further Task-Related Decoding

# After our initial work designing and evaluating convolutional neural networks for movement decoding from EEG, we evaluated the resulting networks on a wide variety of other EEG decoding tasks.
# 
# * mention you helped with writing and setting up code for papers past xx

# ## Decoding different mental imageries

# > The Mixed Imagery Dataset (MID) was obtained from 4 healthy subjects (3 female, all right-handed, age
# 26.75±5.9 (mean±std)) with a varying number of trials (S1: 675, S2: 2172, S3: 698, S4: 464) of imagined
# movements (right hand and feet), mental rotation and mental word generation. All details were the same as
# for the High Gamma Dataset, except: a 64-electrode subset of electrodes was used for recording, recordings
# were not performed in the electromagnetically shielded cabin, thus possibly better approximating conditions
# of real-world BCI usage, and trials varied in duration between 1 to 7 seconds. The dataset was analyzed
# by cutting out time windows of 2 seconds with 1.5 second overlap from all trials longer than 2 seconds (S1:
# 6074 windows, S2: 21339, S3: 6197, S4: 4220), and both methods were evaluated using the accuracy of the
# predictions for all the 2-second windows for the last two runs of roughly 130 trials (S1: 129, S2: 160, S3:
# 124, S4: 123).

# For the mixed imagery dataset, we find the deep ConvNet to perform slightly better and the shallow ConvNet to perform slightly worse than the FBCSP algorithm, as can be seen in {numref}`mixed-imagery-dataset-results`.
# 
# 
# ```{table} Accuracies on the Mixed-Imagery dataset. ConvNet accuracies show the difference to the FBCSP accuracy.
# :name: mixed-imagery-dataset-results
# 
# | FBCSP   | Deep ConvNet | Shallow ConvNet |
# |--|--|--|
# |71.2|+1.0|-3.5|
# ```

# ## Decoding error-related signals
# 
# In this dataset, subjects watched videos of a robot either successfully lifting ball from the ground or letting the ball fall while trying to lift it. The decoding task was to classify whether the person watched a successful or an unsuccessful video from the EEG during either the whole video duration (0-7s) or the part where the actual grasphing and lifting happened (4-7s). 
# Results in {numref}`robot-ball-results` show that the deep ConvNet outperforms regularized linear discriminant analysis (rLDA) as well as FBCSP.
# 
# 
# ```{table} Accuracies for decoding watching of successful or unsuccessful robot ball-lifting.
# :name: robot-ball-results
# 
# |    | 0-7s  | 4-7s |
# |--|--|--|
# |Deep ConvNet|78.31 ± 8.09|73.80 ± 7.52|
# |rLDA|68.29 ± 8.00|64.71 ± 7.36|
# |FBCSP|55.71 ± 4.54|56.80 ± 3.92|
# ```
# 
#     Behncke J., Schirrmeister R. T., Burgard W., and Ball T., "The role of robot design in decoding error-related information from EEG signals of a human observer". 6th International Congress on Neurotechnology, Electronics and Informatics 2018. doi.org/10.5220/0006934900610066
# 
#     Behncke J., Schirrmeister R. T., Burgard W., and Ball T., "The signature of robot action success in EEG signals of a human observer: Decoding and visualization using deep convolutional neural networks". IEEE The 6th International Winter Conference on Brain-Computer Interface 2018. doi.org/10.1109/IWW-BCI.2018.8311531
# 
#     Völker M., Schirrmeister R. T., Fiederer L.D.J., Burgard W., and Ball T., "Deep Transfer Learning for Error Decoding from Non-Invasive EEG". IEEE The 6th International Winter Conference on Brain-Computer Interface 2018. doi.org/10.1109/IWW-BCI.2018.8311491
# 
# 
# ## BCI Robot
#     Burget F.*, Fiederer L.D.J.*, Kuhner D.*, Völker M.*, Aldinger J., Schirrmeister R.T., Do C., Boedecker J., Nebel B., Ball T. and Burgard W., *equally contributing. "Acting Thoughts: Towards a Mobile Robotic Service Assistant for Users with Limited Communication Skill". Proceedings of the 2017 IEEE European Conference on Mobile Robotics. https://arxiv.org/abs/1707.06633
#     
#     
# 
# ## Intracranial
#    
# Behncke J., Schirrmeister R. T., Völker M., Hammer J., Marusič P., Schulze-Bonhage A., Burgard W., and Ball T., "Cross-paradigm pretraining of convolutional networks improves intracranial EEG decoding". IEEE International Conference on Systems, Man, and Cybernetics 2018 arxiv.org/abs/1806.09532
#     
#     
#    Wang, X., Gkogkidis, C. A., Schirrmeister, R. T., Heilmeyer, F. A., Gierthmuehlen, M., Kohler, F., Schuettler, M., Stieglitz, T., Ball, T., 2018.
# Deep Learning for micro-Electrocorticographic (μECoG) Data. Conference paper for IEEE EMBS CONFERENCE ON BIOMEDICAL ENGINEERING AND SCIENCES (IECBES 2018), 16 Sep., accepted.
# https://ieeexplore.ieee.org/document/8626607
# 
# 
# Völker M., Hammer J., Schirrmeister R.T., Behncke J., Fiederer L.D., Schulze-Bonhage A., Marusič P., Burgard W., and Ball T., "Intracranial Error Detection via Deep Learning". IEEE International Conference on Systems, Man, and Cybernetics 2018 https://arxiv.org/abs/1805.01667
# 
# 
# ## Large scale evaluation
# Heilmeyer F.A., Schirrmeister R.T., Fiederer L.D.J., Völker M., Behncke J., Ball T., "A framework for large-scale evaluation of deep learning for EEG". IEEE International Conference on Systems, Man, and Cybernetics 2018 https://arxiv.org/abs/1806.07741
# 
# 
# 
# 
# ### Interpretability (remove?)
# 
#     Hartmann K.G., Schirrmeister R. T., and Ball T., "Hierarchical internal representation of spectral features in deep convolutional networks trained for EEG decoding". IEEE The 6th International Winter Conference on Brain-Computer Interface 2018. doi.org/10.1109/IWW-BCI.2018.8311493
# 
