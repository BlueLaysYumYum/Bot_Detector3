#import packages
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
features_list = ["followers_count", "friends_count", "follower_to_following", "post_count", "has_location",
                     "default_profile_image", "profile_use_background_image", "verified", 'bio_length',
                     "account_age_days","post_frequency"]
#TODO add function
# func for division was messing up at later parts
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
#TODO add tweets frequency

# now = pd.to_datetime('today')
bot_data = pd.read_csv("./twitter_profiles.csv")
# bot_data['bio_length']=bot_data['description'].str.len()
# bot_data["created_at"]=pd.to_datetime(bot_data["created_at"])
# bot_data['account_age_days'] = (pd.to_datetime('today') - bot_data['created_at']).dt.days
# bot_data["follower_to_following"]=bot_data['followers_count']/(bot_data['friends_count']+1)
# bot_data['has_location'] = bot_data['location'].notnull().astype(int)
# features_list=["followers_count","friends_count","follower_to_following","post_count","has_location","default_profile_image","profile_use_background_image","verified",'bio_length',"account_age_days"]
# X = bot_data[features_list]
X = data_unloading(bot_data)
y=bot_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#XGB regression
XGB_model = XGBClassifier(n_estimators=1000, random_state=0,early_stopping_rounds=10)
XGB_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
predictions = XGB_model.predict(X_test)
scores = mean_absolute_error(y_test,predictions)
print(scores)

#Custom input to make sure it is working with split

# custom_val = pd.DataFrame(
#     [{"name": ";saonidoa", "screen_name": "fojknedmsla", "followers_count": 113, "friends_count": 39,
#       "post_count": 266,
#       "lang": "en", "location": "", "default_profile_image": 0, "profile_use_background_image": 1, "verified": 0,
#       "description": "stay away from bird", "created_at": "2025-04-18"}])
# custom_val['bio_length']=custom_val['description'].str.len()
# custom_val["created_at"]=pd.to_datetime(custom_val["created_at"])
# print(custom_val['created_at'].max())
# custom_val['account_age_days'] = (pd.to_datetime('today') - custom_val['created_at']).dt.days
# custom_val["follower_to_following"]=custom_val['followers_count']/(custom_val['friends_count']+1)
# custom_val['has_location'] = custom_val['location'].notnull().astype(int)
# # X_check= data_unloading(custom_val)
#
#
# X_check = custom_val[features_list]
# print(X_check)
# prediction_main = XGB_model.predict(X_check)
# print(prediction_main)
# # if prediction_main[0] > 0.5:
# #     print("Bot Detected")
# # else:
# #     print("Human Detected")
#
# #saving for web below code is ai
#
model_data = {
    'model': XGB_model,
    'features': features_list,
}
joblib.dump(model_data, 'bot_detector.pkl')