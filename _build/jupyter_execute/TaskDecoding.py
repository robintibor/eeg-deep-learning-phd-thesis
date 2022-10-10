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

# ### Decoding Observation of Robots Making Errors

# In two datasets about observing robots making errors, subjects watched videos of a robot either successfully or unsuccessfully attempting one of two tasks: lifting ball from the ground (failure: letting it fall to the ground) or pouring liquid into a glass (failure: pouring the liquid outside of the glass).The decoding task was to classify whether the person watched a successful or an unsuccessful video from the EEG recorded during the observation of the corresponding video. 
# Results for both tasks and two decoding intervals in {numref}`robot-ball-results` show that the deep ConvNet outperforms regularized linear discriminant analysis (rLDA) as well as FBCSP.
# 
# 
# ```{table} Accuracies for decoding watching of successful or unsuccessful robot-liquid pouring or ball-lifting.
# :name: robot-ball-results
# 
# |  robot task | time interval | Deep ConvNet | rLDA | FBCSP|
# |--|--|--|--|--|
# |Pouring Liquid|2-5s|78.2 ± 8.4| 67.5 ± 8.5|60.1 ± 3.7|
# |Pouring Liquid|3.3-7.5s|71.9 ± 7.6|63.0 ± 9.3|66.5 ± 5.7|
# |Lifting Ball|4.8-6.3s|59.6 ± 6.4|58.1 ± 6.6|52.4 ± 2.8|
# |Lifting Ball|4-7s|64.6 ± 6.1|58.5 ± 8.2|53.1 ± 2.5|
# {cite}`behncke2018signature`
# ```

# (flanker-and-gui-section)=
# ### Decoding of Eriksen Flanker Task Errors and Errors during Online GUI Control

# In two further error-related decoding experiments, we evaluated an Eriksen flanker task and errors during an the online control of a graphical user interface through a brain-computer-interface. The Eriksen flanker task required the students to press a left or a right button on a gamepad depending on whether a 'L' or an 'R' was the middle character of a 5-letter string displayed on the screen. For the online GUI control the subjects were given an aim to reach using the GUI. They had to thinki of one of the classes of the aforementioned Mixed Imagery Dataset to choose one of four possible GUI actions. The correct GUI action was always determined by the specificed aim for the subject, hence an erroneous action could be detected. The decoding task in this paper was to distinguish whether the BCI-selected action was correct or erroneous. Results in {numref}`within-subject-flanker-gui-fig` and {numref}`cross-subject-flanker-gui-fig` show that deep ConvNets outperform rLDA in all settings except cross-subject error-decoding for online GUI control, where the low number of subjects (4) may prevent the ConvNets to learn enough to outperform rLDA.

# ![title](images/within-subject-flanker-gui.png)

# ![title](images/cross-subject-flanker-gui.png)

# ````{panels}
# :container: container-fluid 
# :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 
# :card: shadow-none border-0
# 
# ```{figure} images/within-subject-flanker-gui.png
# :width: 100%
# :name: within-subject-flanker-gui-fig
# 
# Comparison of within-subject decoding by rLDA and deep ConvNets. Error bars show the SEM. A) Eriksen flanker task (mean of 31 subjects), last 20% of subject data as test set. Deep ConvNets were 7.12% better than rLDA, pval = 6.24 *10-20 (paired t-test). B) Online GUI control (mean of 4 subjects), last session of each subject as test data {cite}`volker2018deep`
# ```
# 
# ---
# 
# ```{figure} images/cross-subject-flanker-gui.png
# :width: 100%
# :name: cross-subject-flanker-gui-fig
# 
# Mean normalized decoding accuracy on unknown subjects. Error bars show the SEM. A) Eriksen flanker task, trained on 30 subjects, tested on 1 subject.  Deep ConvNets were 5.05% better than rLDA, p = 3.16 *10-4 (paired t-test). B) Online GUI control. Trained on 3 subjects, tested on the  respective remaining subject. {cite}`volker2018deep`
# ```
# 
# ````
# 

# ## Proof-of-concept assistive system
# 
# We also evaluated the use of our deep ConvNet as part of a assistive robot system where the brain-computer interface was sending high-level commands to a robotic arm. In this proof of concept system, the robotic arm could be instructed by the user via the BCI to fetch a cup and directly move the cup to the persons mouth to drink from it. An overview can be seen in {numref}`robot-bci-overview-fig`. Results from {numref}`bci-robot-results` show that 3 out of 4 subjects had a command accuracy of more than 75% and were able to reach the target using less than twice the steps of the minimal path through the GUI (path optimality > %50%).
# 

# ![](images/robot-bci-overview.png)

# ```{figure} images/robot-bci-overview.png
# :name: robot-bci-overview-fig
# 
# Overview of the proof-of-concept assistive system from {cite}`burget2017acting` using the deep ConvNet in the BCI component. Robotic arm could be given high-level commands via the BCI, high-level commands were extracted from a knowledge base. The commands were then autonomously planned and executed by the robotic arm.
# ```

# 
# 
# ```{table} Results for BCI control of the GUI. Accuracy is fraction of correct commands, time is time per command, steps is steps needed to reach the aim, path optimality is ratio of miniminally needed  nubmer of steps to actually used number of steps when every step is optimal, and time/step is time per step.
# :name: bci-robot-results
# 
# | Subject | Runs | Accuracy* [%] | Time [s] | Steps | Path Optimality [%] | Time/Step [s] |
# |--|--|--|--|--|--|--|
# | S1 | 18 | 84.1$\pm$6.1 | 125$\pm$84 | 13.0$\pm$7.8 | 70.1$\pm$22.3 | 9$\pm$2 |
# | S2 | 14 | 76.8$\pm$14.1 | 150$\pm$32 | 10.1$\pm$2.8 | 91.3$\pm$12.0 | 9$\pm$3 |
# | S3 | 17 | 82.0$\pm$7.4 | 200$\pm$159 | 17.6$\pm$11.4 | 65.7$\pm$28.9 | 11$\pm$4 |
# | S4 | 3 | 63.8$\pm$15.6 | 176$\pm$102 | 26.3$\pm$11.2 | 34.5$\pm$1.2 | 6$\pm$2 |
# |  Average  | 13 | 76.7$\pm$9.1 | 148$\pm$50 | 16.7$\pm$7.1 | 65.4$\pm$23.4 | 9$\pm$2 |  
# {cite}`burget2017acting`
# ```

# ## Intracranial EEG decoding

# ### Intracranial EEG Decoding of Eriksen Flanker Task 
# We further evaluated whether the same networks developed for noninvasive EEG decoding can successfully learn to decode intracranial EEG. Therefore, in one work we used the same Eriksen flanker task as described in {ref}`flanker-and-gui-section`, but recorded intracranial EEG from 23 patients who had pharmacoresistant epilepsy {cite}`volker2018intracranial`. 

# ```{table} Results for single-channel intracranial decoding of errors during an Eriksen flanker task. Balanced Accuracy is the mean of the accuracies for correct class ground truth labels and error class ground truth labels.
# :name: intracranial-error-results-table
# 
# | Classifier | Balanced Accuracy  [%] | Accuracy Correct Class [%] | Accuracy Error Class  [%] |
# |--|--|--|--|
# | Deep4Net | 59.28 ± 0.50 | 69.37 ± 0.44 | 49.19 ± 0.56 |
# | ShallowNet | 58.42 ± 0.32 | 74.83 ± 0.25 | 42.01 ± 0.40 |
# | EEGNet | 57.73 ± 0.52 | 57.78 ± 0.48 | 57.68 ± 0.56 |
# | rLDA | 53.76 ± 0.32 | 76.12 ± 0.26 | 31.40 ± 0.38 |
# | ResNet | 52.45 ± 0.21 | 95.47 ± 0.14 | 09.43 ± 0.28 |
# {cite}`volker2018intracranial`
# ```

# ![](images/IntracranialError.png)

# ```{figure} images/IntracranialError.png
# :name: intracranial-error-results-fig
# 
# Results for all-channel intracranial decoding of errors during an Eriksen flanker task {cite}`volker2018intracranial`.
# ```

# ### Transfer Learning for Intracranial Error Decoding

# ![](images/eriksen-flanker-car-driving-tasks.png)

# ```{figure} images/eriksen-flanker-car-driving-tasks.png
# :name: eriksen-flanker-car-driving-tasks-fig
# 
# Sketch of the Eriksen flanker task (A) and screenshot of the car driving task (B). {cite}`behncke2018cross`.
# ```

# We further tested the potential of ConvNets to transfer knowledge learned from decoding intracranial signals in error-decoding paradigm to decoding signals in another a different error-decoding paradigm {cite}`behncke2018cross`. The two error-decoding paradigms were the aforementioned Eriksen flanker task (EFT) and a car driving task (CDT), where subjects had to use a steering wheel to steer a car in a computer game and avoid hitting obstacles, where hitting an obstacle was considered an error event (see {numref}`eriksen-flanker-car-driving-tasks-fig`). Results in {numref}`cross-training-eft-cdt-results-fig` show that pretraining on CDT helps EFT decoding when few EDT data is available.

# ![](images/cross-training-eft-cdt-results.png)

# ```{figure} images/cross-training-eft-cdt-results.png
# :name: cross-training-eft-cdt-results-fig
# 
# Results for transfer learning on the Eriksen flanker task (EFT) and the car driving task (CDT) {cite}`behncke2018cross`. All results are computed for a varying fraction of available data for the target decoding task (bottom row). **A** compares CDT accuracies after training only on CDT or pretraining on EFT and  finetuning on CDT. **B** compares EFT accuracies after only training on EFT or after  pretraining on CDT and finetuning on EFT. As a sanity check for the results in **B**, **C** compares EFT accuracies when pretraining on original CDT data and finetuning on EFT to pretraining on CDT data with shuffled labels (CDT*) and finetuning on EFT. Results show that pretraining on CDT helps EFT decoding when little EFT data is available. 
# 
# ```

# ### Microelectrocorticography decoding of auditory evoked responses in sheep

# ![](images/sheep-sounds.jpg)

# ```{figure} images/sheep-sounds.jpg
# :name: sheep-sounds-fig
# 
# Overview over decoding tasks for auditory evoked responses in a sheep {cite}`wangsheep`. First task (top) was to distingish 3 seconds when the sound was playing from the second before and the second after. Second task (bottom) was to distinguish the first, second and third second during theplaying of the sound. Signals are averaged responses from one electrode during different days, with black and grey being signals while the sheep was awake and red ones while the sheep was under general anesthesia.
# 
# ```

# In this study, we evaluated the ConvNets for decoding auditory evoked responses played to a sheep that was chronically implanted with  a μECoG-based neural interfacing device {cite}`wangsheep`. 3-seconds-long sounds were presented to the sheep and two decoding tasks were defined from those 3 seconds as well as the second immediately before and after the playing of the sound. The first decoding task was to distinguish the 3 seconds when the sound was playing from the second  immediately before and the second immediately after the sound. The second task was distinguishing the first, second and third second of the playing of the sound to discriminate early, intermediate and late auditory evoked response (see {numref}`sheep-sounds-fig`). Results in {numref}`sheep-accuracies-fig` show that the  deep ConvNet can perform as good as FBSCP and rLDA, and perform well on both tasks, whereas rLDA performs competitively only on the first and FBSCP only on the second task. 

# ![](images/sheep-accuracies.png)

# ```{figure} images/sheep-accuracies.png
# :name: sheep-accuracies-fig
# 
# Results of decoding auditory evoked responses from sheep with rlDA and FBSCP or the deep ConvNet. Open circles represent accuracies for individual experiment days and closed circles represent the average over these accuracies.
# 
# ```

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

# In[ ]:




