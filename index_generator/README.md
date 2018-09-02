

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fadymedhat/MCLNN/blob/master/LICENSE)

MCLNN index generation
========
In a 10-fold cross-validation, samples of a dataset are split into 10 subsets, where 8-folds are used for training and 1 fold 
is used for validation and the remaining one is for testing. The folds rotate among each other for each trial of a cross-validation.

This behavior is applied through the index generator by creating 10 subsets of indices following the index assigned to a sound 
clip in Dataset.hdf5 file generated through the dataset transformer. 


## Configuration 

We will refer, in this section, to possible scenarios and their corresponding configurations using the datasets used in the experiments as examples for clarification.


#### A balanced dataset without Augmentation
___


The ESC10 environmental sound dataset:
 * Composed of 400 sound file for 10 environmental categories. 
 * The dataset is balanced, i.e. each category has 40 samples. 
 * The dataset is released into 5-folds. 

So each fold has 8 samples of a specific category. The below listing shows the required configuration to generate
 the training, testing and validation indices for the 5-fold cross-validation.


```
class ESC10:
    DATASET = 'esc10' # the name of the dataset
    
    FOLD_COUNT = 5 # the number of folds for the dataset
    
    FOLDER_NAME = 'folds_indices_esc10' # the name of the folder that will hold the generated .hdf5 indices.
    
    SHUFFLE_CATEGORY_CLIPS = False # this dataset is already released with prespecified folds. So we do not need random shuffling. 
    
    AUGMENTATION_VARIANTS_COUNT = 0 # no augmentation is required for this experiment. So a value of Zero disables augmentation.
    
    CLIP_COUNT_PER_CATEGORY_LIST = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40] # number of clips for each category.
    
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8 # number of samples of a specific category assigned to a fold.
```


#### Unbalanced dataset without Augmentation
The Ballroom dataset is another example. This dataset is made up of 698 music file that are unbalanced in distribution among 8 music genres. There is no specific arrangement or folds defined for the dataset. Accordingly, the index generator will handle the shuffling of the samples across the folds during the index generation. The assigned batch for each fold in this case will be 1 sample at a time that is iteratively assigned to the folds in turn until the samples are consumed.  

``` 	
class BALLROOM:
    DATASET = 'ballroom'
    FOLD_COUNT = 10 # 10-folds cross-validation
    FOLDER_NAME = 'folds_indices_ballroom'
    SHUFFLE_CATEGORY_CLIPS = True # allow shuffling the samples of each category separately before assiging them to the folds
    AUGMENTATION_VARIANTS_COUNT = 0 # augmentation is disabled
    CLIP_COUNT_PER_CATEGORY_LIST = [111, 60, 82, 98, 86, 86, 65, 110] # the samples of each class following the alphabetical order of the class name.
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1 # samples are assigned to the folds a sample at a time until they are consumed.
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
    
    

