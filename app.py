from flask import Flask, request, render_template
import sys
from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.exceptions import MissingSchema
import text_normalization as norm 
import pickle
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt, mpld3
import numpy as np


app = Flask(__name__)
lr_model = pickle.load(open('saved_lr_model.sav', 'rb')) # loading in logistic regression model
cv = pickle.load(open('cv.pickel',"rb")) # loading in count vectorizer model

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/sentiment', methods=['POST'])
def sentiment():

    url = request.form['url'] # obtaining user input 
    review_data = retrieve_data(url) # retrieving reviews from html
    norm_review_data = norm.normalize_data(review_data) # normalizing review text
    cv_review_features = feature_engineering(norm_review_data) # feature engineering
    predictions = lr_model.predict(cv_review_features) # predicting sentiment
    predictions_df = pd.DataFrame(predictions)

    fig = visualization(predictions_df) # creating a visualization
    return render_template('index.html', visualization=fig)

@app.route('/features')
def features():
    return render_template('features.html')   

@app.route('/contact')
def contact():
    return render_template('contact.html')

def retrieve_data(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html5lib')
    reviews = []
    table = soup.find('div', attrs = {'id':'reviews'})
    for review in table.find_all_next('div', attrs = {'class':'margin-b2__09f24__CEMjT border-color--default__09f24__NPAKY'}):
        text = review.span
        if text != None:
            reviews.append(review.span.text)
    
    reviews_df = pd.DataFrame(reviews)
    reviews_df.to_csv("extracted_reviews.csv", index=True) # saving reviews to local drive
    return reviews

def feature_engineering(text):
    cv_features = cv.transform(text)
    return cv_features

def visualization(df):
    # building the visualization
    total_reviews = df.shape[0]
    counts = df.value_counts()
    positive_percent = counts['positive']/total_reviews
    
    sentiment_breakdown = np.array([positive_percent, 1-positive_percent])
    explode = [.05,0]
    labels = ["Positive", "Negative"]
    textprops = {"fontsize":15}
    plt.figure(figsize=(10,10))
    fig = plt.pie(sentiment_breakdown, labels=labels, textprops=textprops, autopct='%.1f%%', explode=explode)
    plt.title("Sentiment Breakdown - Customer Reviews", fontsize=24, fontweight='bold')
    plt.savefig("Static/img/sentiment_visualization.svg") # saving image to local drive
    mpld3.show()
    return fig
   

if __name__ == "__main__":
    app.run(debug=True)