from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open(''))
# tfidf helps to find the weighted count of the
# frequency of the words in the article/corpus
tfidf_vectorizer = pickle.load(open(''))
label_encoder = pickle.load(open(''))

def process(inPath, outPath):
    # read input file
    input_df = pd.read_csv(inPath)
    # vectorize the data
    features = tfidf_vectorizor.transform(input_df['body'])
    # predict the classes
    predictions = model.predict(features)
    # convert output labels to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    #save results to csv
    output_df = input_df[['id', 'category']]
    output_df,to_csv(outPath, index=False)
    
grav.wait_for_request(process)