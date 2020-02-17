# Import dependencies
import pandas as pd
import numpy as np

# Ch1. Introduction to Data Preprocessing
volunteer = pd.read_csv('data/volunteer_opportunities.csv')
volunteer['hits'] = volunteer['hits'].astype('object')
volunteer.loc[:, 'category_desc'] = volunteer['category_desc'].ffill().bfill()

volunteer_processed = pd.read_csv('data/volunteer_opportunities.csv', usecols=['vol_requests', 'title', 'hits', 'category_desc', 'locality', 'region', 'postalcode', 'created_date'])
volunteer_processed = volunteer_processed.dropna(axis=1, thresh=3)
volunteer_processed = volunteer_processed.dropna(subset=['category_desc'])
volunteer_processed['vol_requests_lognorm'] = np.log(volunteer_processed['vol_requests'])
volunteer_processed["created_date"] = pd.to_datetime(volunteer_processed["created_date"])
volunteer_processed['created_month'] = volunteer_processed['created_date'].apply(lambda row: row.month)
volunteer_processed = pd.concat([volunteer_processed, pd.get_dummies(volunteer_processed['category_desc'])],axis=1)

wine = pd.read_csv('data/wine_types.csv')

hiking = pd.read_json('data/hiking.json')

credit_card = pd.read_csv('data/credit-card-full.csv', index_col=0)
credit_card = pd.get_dummies(data=credit_card, columns=['SEX', 'EDUCATION', 'MARRIAGE'])
credit_card = credit_card.drop(['SEX_1', 'EDUCATION_0', 'MARRIAGE_0'], axis=1)

running_times_5k = pd.DataFrame({
    'name': {0: 'Sue', 1: 'Mark', 2: 'Sean', 3: 'Erin', 4: 'Jenny', 5: 'Russell'},
    'run1': {0: 20.1, 1: 16.5, 2: 23.5, 3: 21.7, 4: 25.8, 5: 30.9},
    'run2': {0: 18.5, 1: 17.1, 2: 25.1, 3: 21.1, 4: 27.1, 5: 29.6},
    'run3': {0: 19.6, 1: 16.9, 2: 25.2, 3: 20.9, 4: 26.1, 5: 31.4},
    'run4': {0: 20.3, 1: 17.6, 2: 24.6, 3: 22.1, 4: 26.7, 5: 30.4},
    'run5': {0: 18.3, 1: 17.3, 2: 23.9, 3: 22.2, 4: 26.9, 5: 29.9}
})

def when_to_standardize(input=0):
    if input not in [1, 2, 3, 4]:
        print('Enter 1, 2, 3 or 4 as the answer')

    if input == 1:
        print("Incorrect. A lot of models in scikit-learn expect features to look normally distributed.")

    if input == 2:
        print("Incorrect. You would want to scale your data so you can correctly compare these columns to each other.")

    if input == 3:
        print("Incorrect. Any time you're modeling in a linear space, you want to standardize your data before modeling.")

    if input == 4:
        print("Correct! Standardization is a preprocessing task performed on numerical, continuous data.")

def feature_engineering_puzzle(input=0):
    answers = [
        "Incorrect. This isn't the only statement that's true.",
        "Incorrect. This isn't the only statement that's true.",
        "Incorrect. Weight measurements would be ready for modeling since they are continuous numerical values.",
        "Correct! Timestamps can be broken into days or months, and headlines can be used for natural language processing.",
        "Incorrect"
    ]

    if input not in [1, 2, 3, 4, 5]:
        print('Enter 1, 2, 3, 4 or 5 as the answer')
    else:
        print(answers[input - 1])


def identity_features_puzzle(input=0):
    answers = [
        "Incorrect. This column is already numerical and continuous.",
        "Incorrect. This isn't the only statement that's true.",
        "Incorrect. This isn't the only statement that's true.",
        "Incorrect. This isn't the only statement that's true.",
        "Correct! All three of these columns will require some feature engineering before modeling."
    ]

    if input not in [1, 2, 3, 4, 5]:
        print('Enter 1, 2, 3, 4 or 5 as the answer')
    else:
        print(answers[input - 1])

def feature_selction_puzzle(input=0):
    answers = [
        "Incorrect. Because we've generated an average from these columns, it's likely we don't need them in the final model.",
        "Correct! The text field needs to be vectorized before we can eliminate it, otherwise we might miss out on important data.",
        "Incorrect. We can likely eliminate this field since we've already extracted the float.",
        "Incorrect. The categorical field can be dropped since it's been one-hot encoded.",
        "Incorrect. We could probably drop one of these fields since they're all related to each other."
    ]

    if input not in [1, 2, 3, 4, 5]:
        print('Enter 1, 2, 3, 4 or 5 as the answer')
    else:
        print(answers[input - 1])


def which_is_param(input=0):
    answers = [
        "Not quite, this is something you set so not a parameter.",
        "Yes! coef_ contains the important information about coefficients on our variables in the model. We do not set this, it is learned by the algorithm through the modeling process.",
        "Not quite, this is something you set so not a parameter.",
        "This is how you create a logistic regression estimator, it is not a parameter."
    ]

    if input not in [1, 2, 3, 4]:
        print('Enter 1, 2, 3 or 4 as the answer')
    else:
        print(answers[input - 1])


def which_is_hyperparam(input=0):
    answers = [
        "That's correct! oob_score set to True or False decides whether to use out-of-bag samples to estimate the generalization accuracy.",
        "This is an output, not a hyperparameter.",
        "Whilst trees are important for this model, this is not the name of the hyperparameter that controls them.",
        "Unfortunately we cannot simply set the level of randomness in this model. A nice idea though!"
    ]

    if input not in [1, 2, 3, 4]:
        print('Enter 1, 2, 3 or 4 as the answer')
    else:
        print(answers[input - 1])

def how_many_models(input=0):
    answers = [
        "Close! Though we don't just run one model for each value of each hyperparameter.",
        "Not quite, there are two hyperparameters missing and the calculation is incorrect.",
        "In a grid search we make many (many!) models, not just one big one.",
        "Excellent! For every value of one hyperparameter, we test EVERY value of EVERY other hyperparameter. So you correctly multiplied the number of values (the lengths of the lists)."
    ]

    if input not in [1, 2, 3, 4]:
        print('Enter 1, 2, 3 or 4 as the answer')
    else:
        print(answers[input - 1])

def which_grid_search(input=0):
    answers = [
        "Not quite, there are two hyperparameters missing and the calculation is incorrect.",
        "Not quite, these hyperparameters and options are all valid for this algorithm.",
        "Correct! By looking at the Scikit Learn documentation we know that number_attempts is not a valid hyperparameter. This GridSearchCV will not fit to our data.",
        "There is definitely one model that has incorrect hyperparameters specified in the param_grid."
    ]

    if input not in [1, 2, 3, 4]:
        print('Enter 1, 2, 3 or 4 as the answer')
    else:
        print(answers[input - 1])

