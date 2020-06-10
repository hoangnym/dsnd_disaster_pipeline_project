# import libraries
import nltk
nltk.download(['punkt', 'wordnet'])

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import warnings
import joblib
warnings.filterwarnings("ignore")


def load_data(database_filepath):
    """
    Load data from the sqlite database
    Input - data_file
    Output - X, y, category names
    """
    # read in file
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', con=engine)
    
    # clean data
    df = df.dropna()
    
    # define features and label arrays
    X = df['message']
    y = df[df.columns[4:]]
    category_names = list(y.columns)

    return X, y, category_names


def tokenize(text):
    '''
    Tokenize and lemmatize each word in a given text
    Input:
        text: Message data from any given input for tokenization.
    Output:
        clean_tokens: Resulting list after tokenization.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Create a machine learning pipeline with grid search
    input - none
    output - model_pipeline
    """
    # text processing and model pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
                        ])

    # define parameters for GridSearchCV
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)

    return model_pipeline


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluating trained model.
    inputs - model_pipeline, X_test, y_test, category_names
    output - evaluated model
    """
    # output model test results
    y_pred = model.predict(X_test)
    y_test = y_test.values

    for i in range(y_test.shape[1]):
         print('%30s accuracy : %.2f' %(category_names[i], accuracy_score(y_test[:,i], y_pred[:,i])))


    return model


def save_model(model, model_filepath):
    '''
    Export model as a pickle file
    Input - model and model_filepath
    Output - None, saves model in working directory
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()