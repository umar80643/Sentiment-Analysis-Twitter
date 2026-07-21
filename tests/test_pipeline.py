from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import preprocessing
from sanderstwitter02 import getTweetsRawData
from sentiment import FeatureSettings, extract_features, getTrainingAndTestData, k_fold_cross_validation, predict, trainAndClassify, train_final_model
from stanfordcorpus import getNormalisedTweets

def test_preprocessing_preserves_feature_markers():
    result = preprocessing.processAll("@ana LOVES #Python!!! http://example.com :D")
    assert "__HNDL" in result and "__HASH_PYTHON" in result and "__URL" in result and "__EMOT_LAUGH" in result

def test_feature_extraction_adds_ngrams_and_negation():
    features = extract_features(["not", "good", "product"], FeatureSettings(ngram=3, negtn=True))
    assert "has(not,good)" in features and "has(not,good,product)" in features and features["neg_l(good)"] > 0

def test_sanders_loader(tmp_path):
    corpus = tmp_path / "sanders.csv"; corpus.write_text('apple,positive,1,date,"Great phone"\n', encoding="utf-8")
    assert getTweetsRawData(corpus) == [("Great phone", "pos", "apple", ["@apple"])]

def test_sentiment140_loader(tmp_path):
    source = tmp_path / "tweets.norm.csv"; source.write_text('good day,pos,NO_QUERY,0\n', encoding="utf-8")
    assert getNormalisedTweets(source) == [("good day", "pos", "NO_QUERY", [])]

def test_training_prediction_and_folds():
    tweets = [("I love this", "pos", "", []), ("great product", "pos", "", []), ("bad service", "neg", "", []), ("I hate this", "neg", "", []), ("fine okay", "neu", "", []), ("ordinary day", "neu", "", [])]
    assert len(list(k_fold_cross_validation(tweets, 3))) == 3
    model = trainAndClassify(tweets, folds=3)
    assert predict("I love it", model) in {"pos", "neg", "neu"}
    train, test = getTrainingAndTestData(tweets, 3, 0, "1step", {"ngram": 1, "negtn": False})
    assert train and test

def test_final_model_can_predict():
    tweets = [("love it", "pos", "", []), ("hate it", "neg", "", []), ("ordinary", "neu", "", [])]
    model = train_final_model(tweets)
    assert predict("love it", model) in {"pos", "neg", "neu"}
