import os
import numpy as np
import pandas as pd
import chardet
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

# Download the NLTK data needed
nltk.download('punkt')

# Path to the CSV file
file_path = r'C:\Users\WWD_5\OneDrive\Desktop\INternship Tasks\firstTask\spam.csv'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Detect the encoding of the file
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")

    # Read the CSV file using the detected encoding
    df = pd.read_csv(file_path, encoding=encoding)

    # Display sample of the dataframe and its shape
    print(df.sample(5))
    print(df.shape)

    # Drop the last 3 columns as they have no use
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)

    # Display sample of the dataframe after dropping columns
    print(df.sample(5))

    # Rename the remaining 2 columns
    df.rename(columns={'v1': 'Value', 'v2': 'Message'}, inplace=True)

    # Encode the 'Value' column
    encoder = LabelEncoder()
    df['Value'] = encoder.fit_transform(df['Value'])

    # Check for missing values
    print("Missing values:\n", df.isnull().sum())

    # Check for duplicate values
    print("Number of duplicate rows:", df.duplicated().sum())

    # Drop duplicate rows
    df = df.drop_duplicates(keep='first')

    # Confirm no duplicates remain
    print("Number of duplicate rows after dropping:", df.duplicated().sum())

    # Display the shape of the dataframe
    print("Dataframe shape:", df.shape)

    # Plot a pie chart of the 'Value' column
    plt.figure(figsize=(8, 8))
    plt.pie(df['Value'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f%%")
    plt.title("Distribution of Spam vs Ham")
    plt.show()

    # Create new features for analysis
    df['num_char'] = df['Message'].apply(len)
    df['num_words'] = df['Message'].apply(lambda x: len(nltk.word_tokenize(x)))
    df['num_sentences'] = df['Message'].apply(lambda x: len(nltk.sent_tokenize(x)))

    # Describe messages for ham
    print("Description of ham messages:")
    print(df[df['Value'] == 0][['num_char', 'num_words', 'num_sentences']].describe())

    # Describe messages for spam
    print("Description of spam messages:")
    print(df[df['Value'] == 1][['num_char', 'num_words', 'num_sentences']].describe())

    # Plot histogram of number of characters in messages
    plt.figure(figsize=(12, 8))
    sns.histplot(df[df['Value'] == 0]['num_char'], label='Ham', color='blue', kde=True)
    sns.histplot(df[df['Value'] == 1]['num_char'], label='Spam', color='red', kde=True)
    plt.legend()
    plt.title('Distribution of Number of Characters in Messages')
    plt.show()

    # Pairplot with seaborn
    sns.pairplot(df, hue='Value')
    plt.show()
