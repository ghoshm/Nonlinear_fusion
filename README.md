# Nonlinear fusion

Code for the paper Ghosh et al., 2024 *Nonlinear fusion is optimal for a wide class of multisensory tasks*: 
* [Paper]()
* [Preprint](https://doi.org/10.1101/2023.07.24.550311)

If you have any questions regarding the code please contact either [Marcus Ghosh](https://profiles.imperial.ac.uk/m.ghosh) or [Dan Goodman](https://neural-reckoning.org). 

Note some code uses AtF and FtA to refer to linear and nonlinear fusion. 

## Dependencies
A minimal dependency file is provided in environment.yml.

Our Matplotlib style sheet can be found in *style_sheet.mplstyle*.  

## Tasks 
Sample Python code for each task can be found in the paper's supporting information.  

Otherwise, functions to generate trials for each task can be found in *multisensory.py*. 

To use the latter, install the necessary dependencies, add *multisensory.py* to your working directory then:   
```python
# Import the task(s) you wish to use.
from multisensory import DetectionTask

# Set task parameters - note that these differ per task. 
task = DetectionTask(pm=2 / 3, pe=0.057, pc=0.95, pn=1 / 3, pi=0.01) 

# Generate n_trials, each of length n_steps.
trials = task.generate_trials(n_trials, n_steps) 
```
``trials`` will be a class object containing the following: 
* Repeats: an integar denoting the number of trials.  
* time_steps: an integar denoting the number of time steps per trial.
* task: the task and parameters used to generate the trials. 
* M: a numpy array with a label per trial (-1: left, 0:neutral, 1:right). 
* A: a (repeats x time_steps) numpy array with the stimulus directions for one sensory channel (-1: left, 0:neutral, 1:right).   
* V: a (repeats x time_steps) numpy array with the stimulus directions for the other sensory channel (-1: left, 0:neutral, 1:right).      

## Models 
### MAP estimators
Our Bayesian observer functions can be found in *multisensory.py*. 

For example, to determine the accuracy of linear and nonlinear fusion on the task above:
```python
# Import the classifier you wish to use. 
from multisensory import MAPClassifier
classifier_type = MAPClassifier

# Calculate accuracy
accuracy = [] 
for pairs in [False, True]: # this flag determines linear (False) vs nonlinear (True) fusion.
	classifier = classifier_type(task, pairs=pairs)
	res = classifier.test(trials)
	accuracy.append(res.accuracy)
print("Linear fusion accuracy: " + str(accuracy[0]))
print("Nonlinear fusion accuracy: " + str(accuracy[1]))

```

### ANNs 
Our code for training artificial neural networks, with different activation functions, on our tasks can be found in *Task_Space_AF_Comparisons.ipynb*.

Note:
* This notebook will run much faster if you reduce the number of: ``tasks``, trials per task (``nb_trials``) and tested activation functions (``multisensory_activations``).
* The ``ideal_data`` loaded at the start of the notebook can be downloaded from [here]() or regenerated from *Ideal_data.ipynb*.       

### SNNs 
To train our spiking neural networks we created a pip installable package (m2snn). 

A minimal example of using this to train an SNN can be found in *minimaL_training.py*.  

## Paper figures
The code for generating each figure can be found in the following files. For all SNN results we used our m2snn package - so simply point to that. 

**1C** - m2snn.

**2C** - *detection_params.ipynb*.  
**2D** - *detection_params.ipynb*.  
**2E**:
* Data can be downloaded from [here]() or
* Regenerated by *figure_cns.py* and,
* Plotted with *detection-family-figure.ipynb*.  
**2F** - m2snn.  
**2G** - m2snn.  

**3** - *detection_task_multidim.ipynb*.

**4A** - *Task_Space_AF_Comparisons.ipynb*.   
**4B** - *Single_Unit.ipynb*.  
**4C** - m2snn.  
**4D** - m2snn.  

**5**:
* Data can be downloaded from [here]() or
* Regenerated by m2snn and, 
* Plotted with *Task_Space_Optimal.ipynb*.

**S1** - m2snn.   
**S2** - m2snn.     
**S3** - *continuous_valued_version.ipynb*.   
**S4** - *Task_Space_AF_Comparisons.ipynb*.  
**S5** - m2snn.   
**S6** - m2snn.  