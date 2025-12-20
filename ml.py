#import packages
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
features_list = ["followers_count", "friends_count", "follower_to_following", "post_count", "has_location",
                     "default_profile_image", "verified", 'bio_length',
                     "account_age_days","post_frequency"]

def data_unloading(data):
    data['bio_length'] = data['description'].str.len()
    data["created_at"] = pd.to_datetime(data["created_at"])
    data['account_age_days'] = (pd.to_datetime('today') - data['created_at']).dt.days
    data["follower_to_following"] = data['followers_count'] / (data['friends_count'] + 1)
    data['has_location'] = data['location'].notnull().astype(int)
    data["post_frequency"] = data['post_count'] / (data['account_age_days'] + 1)
    X_param = data[features_list]
    return X_param

# csv reading and division
bot_data = pd.read_csv("./twitter_profiles.csv")
X = data_unloading(bot_data)
y=bot_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#XGB CLassifier
XGB_model = XGBClassifier(n_estimators=1000, random_state=0,early_stopping_rounds=10)
XGB_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
predictions = XGB_model.predict(X_test)

scores = mean_absolute_error(y_test,predictions)
# print(scores)



#Custom input to make sure it is working with split
#
# custom_val = pd.DataFrame(
#     [{"name": ";saonidoa", "screen_name": "fojknedmsla", "followers_count": 113, "friends_count": 39,
#       "post_count": 266,
#       "lang": "en", "location": "", "default_profile_image": 0, "profile_use_background_image": 1, "verified": 0,
#       "description": "stay away from bird", "created_at": "2025-04-18"}])
# #cleaning
# X_check = data_unloading(custom_val)
# X_check = custom_val[features_list]
# # print(X_check)
#
# # Prediction
# prediction_main = XGB_model.predict(X_check)
# probability_main = XGB_model.predict_proba(X_check)
#
# # print(prediction_main)
# # print(probability_main)
#
#
# human_proba=probability_main[0][0]*100
# bot_proba=probability_main[0][1]*100
#
# # if prediction_main==0:
# #     print("the percentage of the data to be human is "+ str(human_proba*100))
# # else:
# #     print("the percentage of the data to be botted is " + str(bot_proba*100))


# #saving for web below code is ai

model_data = {
    'model': XGB_model,
    'features': features_list,
}
joblib.dump(model_data, 'bot_detector.pkl')