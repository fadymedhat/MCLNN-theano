

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/fadymedhat/MCLNN/blob/master/LICENSE)

MCLNN dataset transformation
========
The transformation involves the generation of a single file containing the intermediate representation of a signal,
e.g. spectrogram in case of sound. 

The transformer loads each file in order from the dataset, applied the intermediate representation and stores the resulting
transformation to a single hdf5 file (referred later as Dataset.hdf5) for the whole dataset.


## Configuration 

In this section, we will refer to possible scenarios and their corresponding configurations demonstrated over datasets used 
in the experiments as examples for clarification.


#### A balanced dataset without Augmentation

The ESC10 environmental sound dataset:
 * Composed of 400 sound file for 10 environmental categories. 
 * The dataset is balanced, i.e. each category has 40 samples. 
 * The dataset is released into 5-folds. 

<p align='center'><img height="300"  src='imgs/esc10folderstructure.png'></p>

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
The assigned batch for each fold in this case will be one sample at a time that is iteratively assigned to the folds 
in turn until the samples are consumed.  

``` 	
class BALLROOM:

                        .
                        .
                        .                                                
      
    # enable suffling the samples while being assigned to the folds 
    SHUFFLE_CATEGORY_CLIPS = True
    
    # disable augmentation
    AUGMENTATION_VARIANTS_COUNT = 0
    
                        .
                        .
                        .                                                   
                        
    # samples are assigned one instance at a time
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

```


#### A Balanced dataset with Augmentation

Augmentation is a method to apply certain controlled deformations to the dataset while keeping the properties of the 
original sample to a certain extent. This process enhances the generalization of a model during training.  

This is a different experiment to the ESC10 dataset in which augmentation is applied. In the below listings, we applied 12 augmentation variants for the ESC10 dataset. 

__NOTE:__
 Augmentation is applied on the training data only and the Dataset.hdf5 will include the original and the augmentated
 version, so it is up to the generator to ensure that the training indices include the original and the augmented versions, 
 while constraining the validation and test splits to the original data only. This is carried on for all the folds of the 
 cross-validation operation.

```
class ESC10AUGMENTED:

                        .
                        .
                        .
    
    # shuffling is disabled for the dataset since it is released with predefined splits    
    SHUFFLE_CATEGORY_CLIPS = False
    
    # Augmentation counts applied
    AUGMENTATION_VARIANTS_COUNT = 12
    
                        .
                        .
                        .
    
    # samples assigned for a fold per instance of assignment 
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 8
    
```    
    
    
#### Loading index from CSV

If the dataset is accompanied with a CSV file, specifiying the samples assignment to folds. Below is a listing for the 
 required configuration.
 
 The below figure shows a chunk of the CSV file released with the Urbansound8k. The file is not exactly the original one,
 but rather a modified version interms of the rows ordering in the csv file without changing the data and the sequenc folder added.
 
 The indices of the three highlighted columns are required for the configuration as shown in the below listing. 
 
<p align='center'><img height='200' src='imgs/urbansound8kcsv.png'/></p>
 
 
```

class URBANSOUND8K:

                        .
                        .
                        .

    # shuffling is disabled for the dataset since it is released with predefined splits
    SHUFFLE_CATEGORY_CLIPS = False
    
    # Augmentation is disabled
    AUGMENTATION_VARIANTS_COUNT = 0

                        .
                        .
                        .

    # samples assigned for a fold per instance of assignment
    BATCH_SIZE_PER_FOLD_ASSIGNMENT = 1

    # the name of the CSV file located in the DST_PATH
    CSV_FILE_PATH = os.path.join(DST_PATH, 'UrbanSound8KwithFileSeq.csv')
    
    # csv column index for file sequence - file sequence zero indexed
    COL_FILE_SEQ = 0 
    
    # csv column index for fold id of a file - fold id is 1 indexed
    COL_FOLD_ID = 7 
    
    # csv column index for class id of a file - class id is zero indexed
    COL_CLASS_ID = 8
```