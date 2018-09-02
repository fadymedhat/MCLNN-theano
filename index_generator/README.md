

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fadymedhat/MCLNN/blob/master/LICENSE)

MCLNN index generation
========
In a 10-fold cross-validation, samples of a dataset are split into 10 subsets, where 8-folds are used for training and 1 fold 
is used for validation and the remaining one is for testing. The folds rotate among each other for each trial of a cross-validation.

This behavior is applied through the index generator by creating 10 subsets of indices following the index assigned to a sound 
clip in Dataset.hdf5 file generated through the dataset transformer. 


## Configuration 

In this section, we will refer to possible scenarios and their corresponding configurations using the datasets used in the experiments as examples for clarification.


#### A balanced dataset without Augmentation

The ESC10 environmental sound dataset:
 * Composed of 400 sound file for 10 environmental categories. 
 * The dataset is balanced, i.e. each category has 40 samples. 
 * The dataset is released into 5-folds. 

So each fold has 8 samples of a specific category. The below listing shows the required configuration to generate
 the training, testing and validation indices for the 5-fold cross-validation.


```
class ESC10:

    # dataset name
    DATASET = 'esc10'
    
    # Destination path for the indices to be generated
    DST_PATH = 'I:/ESC10-for-MCLNN'
    
    # Folds count
    FOLD_COUNT = 5
    
    # parent folder for the indices generated
    FOLDER_NAME = DATASET + '_folds_' + str(FOLD_COUNT) + '_index'
    
    # ESC10 is released with predefined folds, so no need for shuffling 
    SHUFFLE_CATEGORY_CLIPS = False
    
    # disable augmentation
    AUGMENTATION_VARIANTS_COUNT = 0
    
    # samples per category
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    
    # batch of samples assigned per fold in a single instance of assignment. 
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8
  
```


#### An unbalanced dataset without Augmentation
The Ballroom dataset is another example:
* It is made up of 698 music file 
* Unbalanced in distribution among 8 music genres. 
* There is no specific arrangement or folds defined for the dataset. 

The index generator will handle the shuffling of the samples across the folds during the index generation.
The assigned batch for each fold in this case will be 1 sample at a time that is iteratively assigned to the folds 
in turn until the samples are consumed.  

``` 	
class BALLROOM:

    # dataset name
    DATASET = 'ballroom'
    
    # Destination path for the indices to be generated
    DST_PATH = 'I:/Ballroom-for-MCLNN'
    
    # Folds count
    FOLD_COUNT = 10
    
    # parent folder for the indices generated
    FOLDER_NAME = DATASET + '_folds_indices'
    
    # enable suffling the samples while being assigned to the folds 
    SHUFFLE_CATEGORY_CLIPS = True
    
    # disable augmentation
    AUGMENTATION_VARIANTS_COUNT = 0
    
    # samples per category following the category order: ('CC', 'Ji', 'QS', 'Ru', 'Sa', 'Ta', 'VW', 'Wa')
    CLIP_COUNT_PER_CATEGORY_LIST = [111, 60, 82, 98, 86, 86, 65, 110]
    
    # samples are assigned one instance at a time
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

```


#### Balanced dataset with Augmentation
This is a different experiment to the ESC10 dataset in which augmentation is applied. Augmentation is a method to apply certain controlled deformations to the dataset that enhances the generalization of the model during training while still keeping the properties of the original sample to a certain extent. In the below listings, we applied 12 augmentation variants for the ESC10 dataset. Accordingly, the index generator will handle this generation and assignment of the samples across the folds. Keeping in mind that augmentation is applied on the training data only and the samples.hdf5 will include the original and the augmentated version, so it is up to the generator to ensure that the training indices include the original and the augmented versions, while constraining the validation and test data to the original data only. This is carried on for all the folds of the cross-validtion operation.

```
class ESC10AUGMENTED:
    DATASET = 'esc10_12augment'
    FOLD_COUNT = 5
    FOLDER_NAME = 'folds_indices_esc10_aug'
    SHUFFLE_CATEGORY_CLIPS = False # dataset has predefined folds
    AUGMENTATION_VARIANTS_COUNT = 12 # 12 augmentations are applied for each sample
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8 # number of samples of a category assigned per fold.
```    
    
    

