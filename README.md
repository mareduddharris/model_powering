# model_powering - Notes
Tools to estimate required model performance to achieve a desired ROC curve 
This repository contains a stream lit to show what model performance is required achieve a desired outcome.

## How to run the code
```bash
streammlit run overview.py
```
shift command P --> markdown --> preview to side. Allow for the preview box.




## Plan for app structure
The app should include the following 


### Version with 1 outcome
* Only a single outcome. Only a single input probability for background prevalance. 
* eg. An intervention could be a blood test, which highlights a disease or doesn't.
* This is a simple prevalance of disease and combined with True Positive and False positive rates of a model


### Version with 2 outcome
* Examples of trade off in outcomes after an intervention on a patient group.
    * Eg. IUmmunosuppressants --> opportunitisc infections vs organ rejection
* Want to be able to have independant outcomes.
    * Eg Surgical intervention --> outcomes of not having surgery vs complications of surgery. 

* Jobs
    * For version 1 - write the code for this and try to delvier a ROC curve for this.


## Notes
bits n bobs. 