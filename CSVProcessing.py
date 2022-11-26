from DataProcessing import DataProcessing

# Defines a csvProcessing variable as a DataProcessing object and inserts the old processed dataset into New2DataSet.csv
csvProcessing = DataProcessing('DataSet/IMDB_Dataset.csv','DataSet/New2DataSet.csv','en_core_web_sm',['parser','ner'])
csvProcessing.createNewDataSet()