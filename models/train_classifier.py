import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import pickle
import pandas as pd
import sys
import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    '''This function loads cleaned data from database'''

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('MessageCategory', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names


def tokenize(text):
    '''
    This function preprocess messages by
    tokenizing, normalizing, and lemmatizing text.

    '''
    # tokenize text
    tokenized = word_tokenize(text)
    # initiate Lemmatizer
    lem = WordNetLemmatizer()
    clean_token = [lem.lemmatize(tok).lower().strip() for tok in tokenized]
    return clean_token


def build_model():
    '''
    This function instantiates Pipeline and GridSearch objects,
    and returns a GridSearchCV object as cv

    '''
    pipeline = Pipeline([('text_pipeline',
                          Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                    ('tfidf', TfidfTransformer())])),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'text_pipeline__vect__ngram_range': ((1, 1), (1, 2))}
    # 'text_pipeline__vect__max_df': (0.5, 0.75, 1.0)

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def train_model(model, X, Y):
    '''This function split train/test data, train model, and return a trained model'''

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2)

    model.fit(X_train, Y_train)

    return model, X_test, Y_test


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function reports f1 score, precision, and recall for each output
    category of the dataset.
    '''
    Y_pred = model.predict(X_test)
    for index, output in enumerate(category_names):
        cr = classification_report(
            Y_test[:, index], Y_pred[:, index], zero_division=0)
        print(output, cr)


def save_model(model, model_filepath):
    '''
    This function saves machine learning model into a pickle file.
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model, X_test, Y_test = train_model(model, X, Y)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
