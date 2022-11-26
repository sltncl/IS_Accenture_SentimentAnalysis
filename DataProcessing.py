import pandas as pd
import spacy
import nltk
import re
import csv

class DataProcessing:

    # path = path to the .csv file
    # core = language for spacy
    # disable = pipeline phase that you want to deactivate
    def __init__(self, oldPath, newPath, core, disable):
        self.core=core
        self.disable = disable
        # Defines nlp with core and disabled passed in the constructor
        self.nlp = self.getInitNlp(self.core, self.disable)
        self.oldPath = oldPath
        self.newPath = newPath
        # Defines the dataframe to use passed in the constructor parameters
        self.df = self.readOldCSV()

    def readOldCSV(self):
        # Returns the reading of the csv file passed in the constructor
        return pd.read_csv(self.oldPath)

    def getInitNlp(self, core, disable):
        # Initialize spacy ‘en’ model, keeping only component needed for lemmatization and creating an engine
        return spacy.load(core, disable = disable)

    # It does data processing
    def text_preprocessing(self, str_input):
        # tokenization, remove punctuation, lemmatization
        words = [token.lemma_ for token in self.nlp(str_input) if not token.is_punct]
        # remove symblos, websites, email addresses
        words = [re.sub(r"[^A-Za-z@]", "", word) for word in words]
        words = [re.sub(r"\S+com", "", word) for word in words]
        words = [re.sub(r"\S+@\S+", "", word) for word in words]
        # remove empty spaces
        words = [word for word in words if word != ' ']
        words = [word for word in words if len(word) != 0]
        # import lists of stopwords
        with open('Stopwords/stopWords.txt', 'r') as f:
            x_stw = f.readlines()
        with open('Stopwords/stopSecondWords.txt', 'r') as f:
            x_stw2 = f.readlines()
        with open('Stopwords/months.txt', 'r') as f:
            x_stwm = f.readlines()
        with open('Stopwords/name.txt', 'r') as f:
            x_stwn = f.readlines()
        with open('Stopwords/weekWords.txt', 'r') as f:
            x_stww = f.readlines()
        # import nltk stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        # combine all stopwords
        [stopwords.append(x.rstrip()) for x in x_stw]
        [stopwords.append(x.rstrip()) for x in x_stw2]
        [stopwords.append(x.rstrip()) for x in x_stwm]
        [stopwords.append(x.rstrip()) for x in x_stwn]
        [stopwords.append(x.rstrip()) for x in x_stww]
        # change all stopwords into lowercase
        stopwords_lower = [s.lower() for s in stopwords]
        # remove stopwords
        words = [word.lower() for word in words if word.lower() not in stopwords_lower]
        # combine a list into one string
        string = " ".join(words)

        return string

    def createNewDataSet(self):
        # Declare a .csv file in write mode.
        # Pass the column names in the csv.DictWriter() method with the file to save.
        # This method returns a writer with which we can add the desired values row by row.
        count = 0
        with open(self.newPath, 'w') as csv_file:
            columnsname = ['review', 'sentiment']
            writer = csv.DictWriter(csv_file, fieldnames=columnsname)
            writer.writeheader()
            # Iterates on original dataset and modifies the new one that will contain the reviews processed
            for index in self.df['review']:
                textReview = self.df.iloc[count]['review']
                textSentiment = self.df.iloc[count]['sentiment']
                writer.writerow({'review': self.text_preprocessing(textReview), 'sentiment': textSentiment})
                count = count + 1













