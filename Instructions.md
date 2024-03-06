# ASOS Machine Learning Engineer Assignment

This exercise is designed to be representative of the types of machine learning problems we face at ASOS and we hope you enjoy the challenge!

Please take your time to carefully read the task description and instructions below. We would like to stress that this is an open-ended assignment with different ways of solving it.

You will be judged on the code quality, engineering practices, machine learning understanding and software lifecycle considerations. Our commitment is to provide ample feedback on how your responded to the challenge.

Should you be successful, we will ask you to present your work during the final round of the interview in the form of a presentation. The time allocated to the final round of the interview process is approximatively 1 hour 30 mins, including a discussion of your solution and Q&A.

## Task

We would like to understand more about ASOS products. In particular, we would like to understand the characteristics/factors that are predictive of customer preferences, future sales and ultimately fashion trends. The problem presented in this task is to predict the material pattern (e.g. spotty, stripy, coloured) of a product from other metadata about the product, including images and text descriptions.

### The objective of this task is to prepare an R&D model prototype developed by a scientist for production

We would like you to build this from the prototype and data provided by following these steps:

1. Refactor the prototype code to prepare for production, considering the needs of data scientists using it for continuous experimentation and evaluation. If there are code design elements that you would include on a real project but do not have time to build for this exercise, please do mention it in your solution, we will make sure to talk about it during the interview.

2. When assessing your solution, we want to be able to run the refactored code and check the behavior and performance of the model matches the prototype’s. Please note that the task is about refactoring and not about optimising the model.

3. Create an API using provided template that accepts a POST request with a JSON payload containing the product metadata and responds with JSON containing the predicted pattern.

4. When assessing your solution, we want to be able to run your API and evaluate the returned predictions.

5. Consider what metrics you would use for assessing API performance and model accuracy. If you have generated a report for model accuracy metrics, please do submit it in your solution, we will talk about it during the interview.

Data

The simplified tab-delimited dataset provided contains information about a subset of the dresses available in our product catalog. For each product, the dataset includes the product name, product images and a text-description. Additionally, the train set also contains the label column (“pattern”).

Product Data:

· Train: 2.4 MB (exercise_train.tsv)

· Test: 0.6 MB (exercise_test.tsv)

### Files provided

* [requirements.txt](requirements.txt) - libraries required and versions
* [data/exercise_train.tsv](data/exercise_train.tsv) - training dataset, contains ground truth labels for each product, as well as descriptions (for text model) and image filenames (for image model)
* [data/exercise_test.tsv](data/exercise_test.tsv) - holdout set, to use for making predictions (no ground truth labels provided)
* [prototype/proc-text.py](prototype/proc-text.py) - prototype text model script, predicts labels using text (outputs predictions_test_text.csv)
* [FastAPI/main.py](FastAPI/main.py) - FastAPI template to create an API

### Create the python environment (e.g.)

``` conda
conda create -n asos-exercise python==3.7
conda activate asos-exercise
pip install -r requirements.txt
```

### Running a FastAPI

``` bash
cd api
uvicorn main:app --reload
```

### Example of an API call

Call: [{"name": "ASOS CURVE Embellished Neck Dress","description": "Lightweight dress by ASOS CURVE;Embroidery and bead embellishment to the top;Button detail to the back with crop style sleeves;","product_id": 1026288}]

Response: [{"pattern":"Embellished","product_id":1026288}]

## .github directory

Please do not modify any files in the .github directory.

## Submitting the test

Please commit your changes to a branch and when you are finished, open a Pull Request into `main` and inform your recruiter. Your access to the repository will be removed shortly afterwards.
