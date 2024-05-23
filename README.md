# AK Pipeline

## File Functions

### JPG_converter

Converts tiff images from the microscopes to jpg format.

### annotation_crop

This file is used to preprocess the manual annotation data. Data is generated using the VIA tool, and more instructions can be found in the file.

### AK pipeline

The main file containing all steps of the pipeline, and uses all associated classes for mask building and diagnosis strategy. Contains the entire train-validate-test process.

### patient_wise_split

Class used to generate splits for the pipeline, and cross validation.

### lasso_mask_builder

Most important file, contains all steps before the DL model in thesis figure 3.1. Processes raw images and prepares object images for DL models.

### grid_srategy

Example strategy file showcasing gird strategy, which is used in the thesis to present results. Similar strategies may be implemented in the same way.

## The Dataset

Due to ethical concerns, the dataset and manual annotation data cannot be uploaded to a third party, or released publicly.
