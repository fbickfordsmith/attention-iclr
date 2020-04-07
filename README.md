# The perceptual boost of visual attention is task-dependent in naturalistic settings
*Published at "Bridging AI and Cognitive Science" (ICLR 2020) and available at http://arxiv.org/abs/2003.00882.*

Freddie Bickford Smith, Xiaoliang Luo, Brett D Roads, Bradley C Love\
University College London

## Abstract
Top-down attention allows people to focus on task-relevant visual information. Is the resulting perceptual boost task-dependent in naturalistic settings? We aim to answer this with a large-scale computational experiment. First, we design a collection of visual tasks, each consisting of classifying images from a chosen task set (subset of ImageNet categories). The nature of a task is determined by which categories are included in the task set. Second, on each task we train an attention-augmented neural network and then compare its accuracy to that of a baseline network. We show that the perceptual boost of attention is stronger with increasing task-set difficulty, weaker with increasing task-set size and weaker with increasing perceptual similarity within a task set.

>![](/analysis/accuracy_change_plots.png)
>Accuracy change produced by attention on 25 difficulty-based task sets (left), 20 size-based task sets (middle) and 40 similarity-based task sets (right). Task-set size is transformed logarithmically with base 2. Least-squares linear regression is applied to each subset of results, from (A) to (F); predictions of the linear models are shown as broken lines.

## A brief guide to reproducing our results
Run the files in the table below, all found in `attention/experiments/`. Then run `accuracy_change_analysis.ipynb`, found in `results/`, to produce the plots and statistics presented in the paper. The code in this repository should work with TensorFlow v2.1.

\# | Step | File
-|-|-
1 | Train baseline network | `baseline_network_training.py`
2 | Test baseline network | `baseline_network_testing.py`
3 | Compute mean accuracy of VGG16 on each ImageNet category | `vgg16_testing.py`
4 | Compute mean VGG16 representation of each ImageNet category | `representations.py`
5 | Define task sets | `define_task_sets_difficulty.py` `define_task_sets_size.py` `define_task_sets_similarity.py`
6 | Check attention works as intended | `attention_networks_check.py`
7 | Train attention networks | `attention_networks_training.py`
8 | Test attention networks | `attention_networks_testing.py`
9 | Combine results from testing attention networks | `combine_results.py`

## A note on previous versions of our experiment
You might look at filenames like `diff_v3_results.csv` and wonder what happened to previous versions. Did we simply repeat our experiment until we found the positive results we were looking for? No. Were there a number of experiment-design iterations before we were satisfied with the setup? Yes. For full transparency, below is a summary of why we rejected previous versions of our experiment.

Weights | Results | Task set | Reason for rejection
-|-|-|-
`diff_v1_weights.npy` | `diff_v1_results.csv` | `diff_v1_wnids.csv` | Too few task sets (insufficient coverage of task-set difficulty)
`diff_v2_weights.npy` | `diff_v2_results.csv` | `diff_v2_wnids.csv` | Wrong `patience` value for early-stopping criterion
`size_v1_weights.npy` | `size_v1_results.csv` | `size_v1_wnids.csv` | Wrong initialisation for attention weights
`size_v2_weights.npy` | `size_v2_results.csv` | `size_v1_wnids.csv` | Wrong sizes for task sets
`size_v3_weights.npy` | `size_v3_results.csv` | `size_v2_wnids.csv` | Wrong sizes for task sets
`size_v4_weights.npy` | `size_v4_results.csv` | `size_v2_wnids.csv` | Ran with `use_multiprocessing=True` to debug training progress
`size_v5_weights.npy` | `size_v5_results.csv` | `size_v3_wnids.csv` | Wrong task sets due to a bug in `define_task_sets_size.py`
`sim_v1_weights.npy` | `sim_v1_results.csv` | `sim_v1_wnids.csv` | Wrong initialisation for attention weights
`sim_v2_weights.npy` | `sim_v2_results.csv` | `sim_v2_wnids.csv` | Wrong `patience` value for early-stopping criterion

## Get in touch
Contact Freddie if you find an error in this repository or if you have questions about the associated research. His email address is in the arXiv paper linked above.
