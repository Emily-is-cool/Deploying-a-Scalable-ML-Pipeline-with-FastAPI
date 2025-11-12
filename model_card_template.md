# Model Card



## Model Details
 This model uses a Random Forest Classifier trained on the Census Income (Adult) dataset for predicting whether or not an individual earns more than 50K per year based on various attributes including sex, race, age, marital status, occupation and education among others. 
> Framework: scikit-learn version 1.3
> Algorithm: Random Forest Classifier (n_estimators=100, random_state=42)
> Target: salary greater than 50K
> Author: Emily Brune
> Date: November 2025


## Intended Use
>This model is used to predict whether an individual makes more than 50K per year. It is used to learn about training a model and implementing a pipeline with FastAPI. This should not be used in real-world decision that may impact individuals. 


## Training Data
> Source: UCI Adult Dataset / Census Income dataset 
> Rows: ~32,000 (after cleaning)
> Split: 80% training / 20% testing
> Features: Age, education, marital-status, occupation, sex, race, native country etc. 


## Evaluation Data
>Evaluation data is derived from 20% of the dataset and both sets are preprocessed using OneHotEncoder and LabelBinarizer. 


## Metrics

>The model was evaluated using standard classification metrics:

| Metric	        | Score           | 
--------------------|------------------
| Precision	        |          0.7419 |
| Recall	        |          0.6384 |
| F1 (F-beta)	    |          0.6863 |

Additional slice-based evaluations (via performance_on_categorical_slice) analyze model bias across categories such as race, sex, and education.


## Ethical Considerations
> The datset contains sensitive attributes such as sex, race, marital status. Predictions about income or economic status can reinforce existing biases present in historical data. This model is intended for educational use only. 


## Caveats and Recommendations
> This dataset is from 1994, so for current predictions, the data would need to be updated and retrained. 
> Model accuracy may differ for underrepresented slices.