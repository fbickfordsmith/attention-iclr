# The perceptual boost of visual attention is task-dependent in naturalistic settings
*Published as a workshop paper at "Bridging AI and Cognitive Science" (ICLR 2020)*

Freddie Bickford Smith, Xiaoliang Luo, Brett D Roads, Bradley C Love\
University College London

## Abstract
Top-down attention allows people to focus on task-relevant visual information. Is the resulting perceptual boost task-dependent in naturalistic settings? We aim to answer this with a large-scale computational experiment. First, we design a collection of visual tasks, each consisting of classifying images from a chosen task set (subset of ImageNet categories). The nature of a task is determined by which categories are included in the task set. Second, on each task we train an attention-augmented neural network and then compare its accuracy to that of a baseline network. We show that the perceptual boost of attention is stronger with increasing task-set difficulty, weaker with increasing task-set size and weaker with increasing perceptual similarity within a task set.

>![](/results/accuracy_change_plots.png)
>Accuracy change produced by attention on 25 difficulty-based task sets (left), 20 size-based task sets (middle) and 40 similarity-based task sets (right). Task-set size is transformed logarithmically with base 2. Least-squares linear regression is applied to each subset of results, from (A) to (F); predictions of the linear models are shown as broken lines.

## Repository guide
All `.py` files contain a docstring describing what they do.

For each sub-experiment we use
1. `experiments/define_task_sets_[type_context].py` to define a collection of task sets
2. `experiments/make_dataframes.py` to build dataframes defining the training set for each task set
3. `experiments/train_on_task_set.py` to train an attention network on each task set
4. `experiments/test_on_task_set.py` to test these networks

## Citation
Bickford Smith, Luo, Roads, Love (2020). The perceptual boost of visual attention is task-dependent in naturalistic settings. ICLR workshop on "Bridging AI and Cognitive Science".
