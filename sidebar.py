# Importing all the necessary libraries
# Libraries for basic functions
import keras
import pandas as pd
import numpy as np
from datetime import date
import dateutil
import pandas_datareader as pdr
import yfinance as yf
from math import sqrt

# Libraries fpr data plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 16, 8

# Libraries for scaling, data preprocessing and evaluation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Libraries for LSTM model
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import Adam

# Libraries for the ARIMA model
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

import streamlit as st
from plotly import graph_objs as go
import plotly.express as px

# Libraries for stock news
from urllib.request import urlopen,Request
from bs4 import BeautifulSoup

# Libraries for news  and twitter analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from pprint import pprint
import re
import string
from nltk.corpus import stopwords
import Twitter_Credentials
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# ML Libraries
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from PIL import Image
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
from wordcloud import STOPWORDS


begin = "1970-01-01"

now = date.today()

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.max_colwidth', None)


@st.cache
def load_data(ticker):
    data = yf.download(ticker, begin, now)
    data.reset_index(inplace=True)
    return data


def get_stock_news(ticker):
    data_url = 'https://finviz.com/quote.ashx?t='
    news_table = {}

    url = data_url + ticker
    req = Request(url=url, headers={'user-agent': 'My-project'})
    res = urlopen(req)
    source_code = BeautifulSoup(res, 'html')
    news = source_code.find(id="news-table")
    news_table[ticker] = news

    data = []

    for ticker, x in news_table.items():

        for row in x.findAll('tr'):

            x = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
            else:
                dates = date_data[0]
                time = date_data[1]

            data.append([dates, time, x])

    news = pd.DataFrame(data, columns=['Date', 'Time', 'Title'])
    return news


def get_tweets():

    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

    # Running the Twitter Authentication Code:

    authenticate = tweepy.OAuthHandler(Twitter_Credentials.Consumer_Key, Twitter_Credentials.Consumer_Secret)
    authenticate.set_access_token(Twitter_Credentials.Access_Token, Twitter_Credentials.Access_Token_Secret)

    api = tweepy.API(authenticate, wait_on_rate_limit=True)

    # function to display data of each tweet
    def rawtweetdata(n, ith_tweet):
        print()
        print(f"Tweet {n}:")
        print(f"Username:{ith_tweet[0]}")
        print(f"Tweet Text:{ith_tweet[1]}")
        print(f"Hashtags Used:{ith_tweet[2]}")

    # function to perform data extraction
    def scrape(words, date_since, numtweet):

        # Creating DataFrame using pandas
        db = pd.DataFrame(columns=['username', 'text', 'hashtags'])

        # We are using .Cursor() to search through twitter for the required tweets.
        # The number of tweets can be restricted using .items(number of tweets)
        tweets = tweepy.Cursor(api.search, q=words, lang="en",
                               since=date_since, tweet_mode='extended').items(numtweet)

        # .Cursor() returns an iterable object. Each item in
        # the iterator has various attributes that you can access to
        # get information about each tweet
        list_tweets = [tweet for tweet in tweets]

        # Counter to maintain Tweet Count
        i = 1

        # we will iterate over each tweet in the list for extracting information about each tweet
        for tweet in list_tweets:
            username = tweet.user.screen_name
            hashtags = tweet.entities['hashtags']

            # Retweets can be distinguished by a retweeted_status attribute,
            # in case it is an invalid reference, except block will be executed
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:
                text = tweet.full_text
            hashtext = list()
            for j in range(0, len(hashtags)):
                hashtext.append(hashtags[j]['text'])

                # Here we are appending all the extracted information in the DataFrame
            ith_tweet = [username, text, hashtext]
            db.loc[len(db)] = ith_tweet

            # Function call to print tweet data on screen
            rawtweetdata(i, ith_tweet)
            i = i + 1
        filename = 'scraped_twitter_data.csv'

        # we will save our database as a CSV file.
        db.to_csv(filename, encoding='utf-8')


    def preprocess_tweet_text(tweet):
        tweet.lower()
        # Remove urls
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
        tweet = re.sub("(@[A-Za-z0-9_]+)", "", tweet)

        # Remove stopwords
        tweet_tokens = word_tokenize(tweet)

        filtered_words = [w for w in tweet_tokens if not w in stop_words]

        return " ".join(filtered_words)

    def get_feature_vector(train_fit):
        vector = TfidfVectorizer(sublinear_tf=True)
        vector.fit(train_fit)
        return vector

    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

    if __name__ == '__main__':
        auth = tweepy.OAuthHandler(Twitter_Credentials.Consumer_Key, Twitter_Credentials.Consumer_Secret)
        auth.set_access_token(Twitter_Credentials.Access_Token, Twitter_Credentials.Access_Token_Secret)

        api = tweepy.API(auth)

        words = st.text_input("Enter the twitter hashtag", "#aapl")

        date_since = st.text_input("Enter the date you want to get tweets from", "YYYY-MM-DD")

        # number of tweets you want to extract in one run

        numtweet = 1000
        scrape(words, date_since, numtweet)
        print('Scraping has completed!')

    df = pd.read_csv('scraped_twitter_data.csv')

    data = df.drop(["Unnamed: 0", 'username', 'hashtags'], axis=1)

    data['text'] = data['text'].apply(preprocess_tweet_text)

    data['text'] = data['text'].apply(remove_emoji)

    return data


def main():
    menu = ["Data","News Sentiments","Twitter Sentiments","Forecast"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Data":
        st.title("Stock Data and Trend")

        stock = st.text_input('Enter stock ticker', "AAPL")
        st.button("Submit")
        company_name = yf.Ticker(stock).info['longName']

        data1 = load_data(stock)

        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data")
            st.write(data1.tail())

        price_type = ('Open', 'Close', 'High', 'Low', 'Adj Close')
        column = st.selectbox("Select the type of price you want to predict", price_type)

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data1['Date'], y=data1[column], name=column + " data"))
            fig.layout.update(title_text='Time Series data for ' + company_name + column + ' price data',
                              xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        if st.button("Enter"):
            plot_raw_data()

        data1 = data1.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0, len(data1)), columns=['Date', column])

        for i in range(0, len(data1)):
            new_data['Date'][i] = data1.Date[i]
            new_data[column][i] = data1[column][i]

        period = (5, 10, 15, 20, 50, 100, 200, 500)
        period = st.selectbox("Select a rolling period", period)

        def plot_ma():
            # fig = go.Figure()
            fig = px.line(new_data, x=new_data['Date'], y=new_data[column])
            fig.add_scatter(x=new_data['Date'], y=new_data['MA'])
            fig.layout.update(title_text='Averages for ' + company_name + column + ' price data',
                              xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        # the simple moving average over a period
        new_data['MA'] = data1[column].rolling(period, min_periods=1).mean()

        if st.button('Done'):
            plot_ma()

        # get stock news

        news_data = get_stock_news(stock)

        if st.checkbox('Show news Data'):
            st.subheader("Showing news data")
            st.write(news_data.tail())

        tweets = get_tweets()

        if st.checkbox('Show Tweets'):
            st.subheader("Showing tweets for selected hashtag")
            st.write(tweets.tail())


    elif choice == "News Sentiments":
        st.header("Are the news for the stock Positive or Negative")

        stock = st.text_input('Enter stock ticker', "AAPL")
        st.button("Submit")
        company_name = yf.Ticker(stock).info['longName']

        news_data = get_stock_news(stock)

        sia = SIA()
        results = []

        for line in news_data['Title']:
            pol_score = sia.polarity_scores(line)
            pol_score['headline'] = line
            results.append(pol_score)

        # pprint(results[:3], width=100)

        df = pd.DataFrame.from_records(results)
        df['label'] = 0
        df.loc[df['compound'] > 0, 'label'] = 1
        df.loc[df['compound'] < 0, 'label'] = -1
        df.head()

        counts = df.label.value_counts(normalize=True) * 100

        x = ['Negative','Neutral','Positive']

        def news_sentiment_plot():
            fig = px.bar(df, x=x, y=counts,color=x)
            fig.layout.update(xaxis_title='Sentiments',
                              yaxis_title='Sentiment Counts',
                              title_text='News Sentiments for ' + company_name,
                              font_size=15,
                              title={'font_color': 'purple',
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     'x': 0.5
                                     }
                             )
            st.plotly_chart(fig)

        news_sentiment_plot()

    elif choice == "Twitter Sentiments":

        stock = st.text_input('Enter stock ticker', "AAPL")
        st.button("Submit")
        company_name = yf.Ticker(stock).info['longName']


        st.header("What people are saying on Twitter about the stock")

        tweets = get_tweets()

        # allwords = ''.join([twts for twts in tweets['text']])
        #
        # mask = np.array(Image.open('Cloud.png'))
        # wordcloud = WordCloud(stopwords=STOPWORDS, mask=mask, background_color='white', width=mask.shape[1],
        #                       height=mask.shape[0], random_state=7, max_font_size=200).generate(allwords)
        #
        # plt.imshow(wordcloud, interpolation="bilinear")
        # plt.axis('off')
        #
        # plt.show()

        def getSubjectivity(text):
            return TextBlob(text).sentiment.subjectivity

        def getPolarity(text):
            return TextBlob(text).sentiment.polarity

        tweets['Subjectivity'] = round(tweets['text'].apply(getSubjectivity), 3)
        tweets['Polarity'] = round(tweets['text'].apply(getPolarity), 3)

        def getAnalysis(score):
            if -1.00 <= score < -0.75:
                return 'Extremely Negative'
            elif -0.75 <= score < -0.50:
                return 'Very Negative'
            elif -0.50 <= score < -0.25:
                return 'Negative'
            elif -0.25 <= score < 0:
                return 'Slighly Negative'
            if score == 0:
                return 'Neutral'
            elif 0 < score < 0.25:
                return 'Slightly Positive'
            elif 0.25 <= score <= 0.50:
                return 'Positive'
            elif 0.50 < score <= 0.75:
                return 'Very Positive'
            else:
                return 'Extremely Positive'

        tweets['Analysis'] = tweets['Polarity'].apply(getAnalysis)

        extremely_positive_tweets = tweets[tweets.Analysis == 'Extremely Positive']

        round((extremely_positive_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        very_positive_tweets = tweets[tweets.Analysis == 'Very Positive']

        round((very_positive_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        positive_tweets = tweets[tweets.Analysis == 'Positive']

        round((positive_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        slightly_positive_tweets = tweets[tweets.Analysis == 'Slightly Positive']

        round((slightly_positive_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        neutral_tweets = tweets[tweets.Analysis == 'Neutral']

        round((neutral_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        slightly_negative_tweets = tweets[tweets.Analysis == 'Slightly Negative']

        round((slightly_negative_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        negative_tweets = tweets[tweets.Analysis == 'Negative']

        round((negative_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        very_negative_tweets = tweets[tweets.Analysis == 'Very Negative']

        round((very_negative_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        extremely_negative_tweets = tweets[tweets.Analysis == 'Extremely Negative']

        round((extremely_negative_tweets.shape[0] / tweets.shape[0]) * 100, 1)

        count = tweets['Analysis'].value_counts()
        print(count)

        x = ['extremely_positive', 'very_positive', 'positive', 'slightly_positive',
             'neutral', ' slightly_negative', 'negative', 'very_negative',
             'extremely_negative']

        def tweets_sentiment_plot():
            fig = px.bar(tweets, x=tweets['Analysis'].unique(), y=count)
            fig.layout.update(xaxis_title='Sentiments',
                              yaxis_title='Sentiment Counts',
                              title_text='User Sentiments for ' + company_name,
                              font_size=15,
                              title={'font_color': 'purple',
                                     'xanchor': 'center',
                                     'yanchor': 'top',
                                     'x': 0.5
                                     }
                             )
            st.plotly_chart(fig)

        tweets_sentiment_plot()

    else:

        st.header("Price forecast for next 30 days")

        stock = st.text_input('Enter stock ticker', "e.g. AAPL")
        st.button("Submit")
        company_name = yf.Ticker(stock).info['longName']

        data1 = load_data(stock)

        price_type = ('Open', 'Close', 'High', 'Low', 'Adj Close')
        column = st.selectbox("Select the type of price you want to predict", price_type)
        st.button("Enter")

        data = data1.reset_index()[column]

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(np.array(data).reshape(-1, 1))

        training_size = int(len(data) * 0.85)
        test_size = len(data) - training_size
        train_data, test_data = data[0:training_size, :], data[training_size:len(data), :1]

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]  ###i=0, 0,1,2,3-----99   100
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=X_train.shape[1:]))
        Dropout(0.1)
        model.add(LSTM(units=50, return_sequences=True))
        Dropout(0.1)
        model.add(LSTM(units=50))

        # Dropout(0.05)
        model.add(Dense(units=25))
        Dropout(0.5)
        model.add(Dense(1))

        model.compile(loss=['mean_squared_error'], optimizer=Adam(lr=0.01))

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100, verbose=0)


        # # Lets Do the prediction and check performance metrics
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Transformback to original form
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # Calculate RMSE performance metrics
        print(sqrt(mean_squared_error(y_train, train_predict)))

        ### Test Data RMSE
        print(sqrt(mean_squared_error(y_test, test_predict)))

        ### Plotting
        # shift train predictions for plotting
        # look_back = 100
        # trainPredictPlot = np.empty_like(data)
        # trainPredictPlot[:, :] = np.nan
        # trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
        # # shift test predictions for plotting
        # testPredictPlot = np.empty_like(data)
        # testPredictPlot[:, :] = np.nan
        # testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict

        # # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(data))
        # plt.plot(trainPredictPlot)
        # plt.plot(testPredictPlot)
        # plt.show()

        # test = len(test_data) - 100

        x_input = test_data[-101:].reshape(1, -1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()

        # demonstrate prediction for next 30 days

        lst_output = []
        n_steps = 100
        i = 0
        while i < 30:

            if len(temp_input) > 100:
                # print(temp_input)
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i, x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                # print(x_input)
                yhat = model.predict(x_input, verbose=1)
                print("{} day output {}".format(i, yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                # print(temp_input)
                lst_output.extend(yhat.tolist())
                i = i + 1
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i + 1

        day_new = np.arange(1, 101)
        day_pred = np.arange(101, 131)

        value_points = len(data) - 100
        print(day_new.shape)
        print(scaler.inverse_transform(data[value_points:]).shape)

        # print(scaler.inverse_transform(data[value_points:]))

        print(scaler.inverse_transform(lst_output))

        st.set_option('deprecation.showPyplotGlobalUse', False)

        plt.plot(day_new, scaler.inverse_transform(data[value_points:]))
        plt.plot(day_pred, scaler.inverse_transform(lst_output))

        st.pyplot()
        keras.backend.clear_session()

if __name__ == '__main__':
    main()


