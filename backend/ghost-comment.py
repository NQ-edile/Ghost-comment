from flask import Flask, jsonify, request
import urllib.request 
from flask_cors import CORS
app = Flask(__name__)

CORS(app)

cors = CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})

@app.route("/", methods=['GET', 'POST'])
def index():
    if (request.method == 'POST'):
        import random
        import re, string
        from nltk.corpus import twitter_samples
        from nltk.stem.wordnet import WordNetLemmatizer
        from nltk.tag import pos_tag
        from nltk.corpus import stopwords
        from nltk import FreqDist
        from nltk import classify
        from nltk import NaiveBayesClassifier
        from nltk.tokenize import word_tokenize

        stop_words = stopwords.words('english')

        def remove_noise(tweet_tokens, stop_words = ()):

            cleaned_tokens = []

            for token, tag in pos_tag(tweet_tokens):
                token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                            '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
                token = re.sub("(@[A-Za-z0-9_]+)","", token)

                if tag.startswith("NN"):
                    pos = 'n'
                elif tag.startswith('VB'):
                    pos = 'v'
                else:
                    pos = 'a'

                lemmatizer = WordNetLemmatizer()
                token = lemmatizer.lemmatize(token, pos)

                if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                    cleaned_tokens.append(token.lower())
            return cleaned_tokens

        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

        positive_cleaned_tokens_list = []
        negative_cleaned_tokens_list = []

        for tokens in positive_tweet_tokens:
            positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_tweet_tokens:
            negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

        def get_tweets_for_model(cleaned_tokens_list):
            for tweet_tokens in cleaned_tokens_list:
                yield dict([token, True] for token in tweet_tokens)

        positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
        negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

        positive_dataset = [(tweet_dict, "Positive")
                            for tweet_dict in positive_tokens_for_model]

        negative_dataset = [(tweet_dict, "Negative")
                            for tweet_dict in negative_tokens_for_model]

        dataset = positive_dataset + negative_dataset

        random.shuffle(dataset)

        train_data = dataset[:7000]
        classifier = NaiveBayesClassifier.train(train_data)

        message = request.get_json()
        custom_tweet = message['message']
        custom_tokens = remove_noise(word_tokenize(custom_tweet))

        sentiment = classifier.classify(dict([token, True] for token in custom_tokens))


        return jsonify({'response': sentiment}), 201
    else:
        return jsonify({"hey":"Hey There!"})

if __name__ == '__main__':
    app.run(debug=True)