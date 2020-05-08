# Flowability Regression and Classification Toolkit 

## Summary
This Github hosts an API for:
 * Feature engineering
 * PCA (linear and nonlinear) dimensionality reduction
 * Training and Testing Regression and Classification models for predicting flowability (raw values or classes)

 This work was created for WPI Army Research Laboratory in the Spring '20 semester. The goal of this project is to use Microtrac parameters (as well as potentially augmented features) to predict the Hall Flow values of new metal particle samples. For more details, please see the Research Summary Presentation. 

## Files 
### API & helper files 
There are 2 main classes: a DataPreprocessor (*DataPreprocessor.py*), and a TrainTestPipeline (*TrainTestPipeline.py*).

**DataPreprocessor**  
This class handles feature engineering on raw data, correlation computations, and PCA computations. It also allows for visualizing correlation matrices and PCA (linear and nonlinear) and saving it to a file.  
Dependent files: 
* *data_preprocessing.py* : This contains helper functions for cleaning & preparing the dataset 

To initialize, specify the:
 - **Data Folder** with raw, unprocessed CSV files of samples data (e.g., Ti-Nb-Zr (0-63) Particles.csv is in here)
 - **Flow Values Excel Sheet** with the target flow values and target flow classes.  
 This also includes the Augmented Density (AugDensity) that the materials science team provided for each sample.


**IMPORTANT:**   
* In the Flow Values Excel sheet, do **not** type outside of the colored rows, especially in rows below the data. If adding more data, make sure to follow the same format as in the sheet.     
* Make sure the names of your samples, and size ranges of the samples, match **exactly** with how they are named in each raw Excel data sheet in your data folder.  

**TrainTestPipeline**  
This class takes:
* Preprocessed X (predictors, after feature engineering or dimensionality reduction) 
* Y data (raw flow values for prediction or flow classes)
* A list of samples corresponding to each row in the X &amp; y data
* A model type (a string, see below. Don't mistype this!)
* A list of heldout samples for testing. Each sample is a string, such as 'Ti-Nb-Zr (0-63)'. By default, 3 heldout samples will be chosen at random from the 8 samples in our dataset. 

The models supported are:

**Regression:** 
* DecisionTreeRegressor
* SupportVectorRegressor 
* RandomForestRegressor 
* kNeighborsRegressor

**Classification**:
* LogisticRegression
* DecisionTreeClassifier
* RandomForestClassifier
* KNeighborsClassifier

This class can run a full train/test pipeline using one command, ```do_train_test(...)```.

You can also use it to visualize test performance, visualize the internals of a tree model, and see the feature importance rankings produced by a tree model. All visualizations and feature rankings are saved to an output file. 

To find a best training subset, use the ```exhaustive_train()``` command. 

### Jupyter Notebooks (.ipynb)
Start with these! There are 3 notebooks: 
* Flowability Regression: Pearson correlation matrix feature selection with a DecisionTreeRegressor 
* Flowability Classification - Binary or Multiclass: Pearson correlation matrix feature selection with a DecisionTreeClassifier
* Linear and NonLinear PCA:


Each of these gives a detailed walkthrough of how to utilize the two classes above, as well as how to save visualizations from data preprocessing and model testing. Please start here, and make sure you have **RawData/** and **TrueFlowValues_.xlsx** downloaded. 


*If you have any questions, please email Alissa Ostapenko at aostapenko@wpi.edu.