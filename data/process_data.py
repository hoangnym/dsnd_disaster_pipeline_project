# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from CSV files and returns them as pandas dataframes
    
    Input:
        - messages_filepath: csv of twitter messages
        - categories_filepath: csv of categories
    Output:
        - df: joined but uncleaned version of the two csv files
    """

    # importing files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merging dataframes
    df = pd.merge(messages, categories, how='inner', on='id').drop('id', axis=1)

    return df
 
def clean_data(df):
    """Clean the dataframe
    Input:
        -df: uncleaned version of df
    Output:
        -df: cleaned version of df
    """
    # create a dataframe of the individual category columns
    categories = df['categories'].str.split(";", expand=True)
    print(categories)
    
    # select the first row of the categories dataframe
    row = categories.loc[1,:]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x[:-2]))
    
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save dataframe in working directory.
    
    Input:
        -df: cleaned version of df
        -database_filename: name of database in sql format
    Output:
    -None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()