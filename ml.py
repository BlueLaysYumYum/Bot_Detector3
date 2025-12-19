#import packages
import numpy as np
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
features_list = ["followers_count", "friends_count", "follower_to_following", "post_count", "has_location",
                     "default_profile_image", "profile_use_background_image", "verified", 'bio_length',
                     "account_age_days"]
#func for division was messing up at later parts
# def data_unloading(data):
#     data['bio_length'] = data['description'].str.len()
#     data["created_at"] = pd.to_datetime(data["created_at"])
#     data['account_age_days'] = (data['created_at'].max() - data['created_at']).dt.days
#     data["follower_to_following"] = data['followers_count'] / (data['friends_count'] + 1)
#     data['has_location'] = data['location'].notnull().astype(int)
#
#     X = bot_data[features_list]
#     return X
#csv reading and division
bot_data = pd.read_csv("./twitter_profiles.csv")
bot_data['bio_length']=bot_data['description'].str.len()
bot_data["created_at"]=pd.to_datetime(bot_data["created_at"])
bot_data['account_age_days'] = (bot_data['created_at'].max() - bot_data['created_at']).dt.days
bot_data["follower_to_following"]=bot_data['followers_count']/(bot_data['friends_count']+1)
bot_data['has_location'] = bot_data['location'].notnull().astype(int)
features_list=["followers_count","friends_count","follower_to_following","post_count","has_location","default_profile_image","profile_use_background_image","verified",'bio_length',"account_age_days"]
X = bot_data[features_list]
y=bot_data["label"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#XGB regression
XGB_model = XGBRegressor(n_estimators=1000, random_state=0,early_stopping_rounds=10)
XGB_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
predictions = XGB_model.predict(X_test)
scores = mean_absolute_error(y_test,predictions)
print(scores)

#Custom input to make sure it is working with split

custom_val = pd.DataFrame(
    [{"name": ";saonidoa", "screen_name": "fojknedmsla", "followers_count": 43249, "friends_count": 23291,
      "post_count": 21312,
      "lang": "en", "location": "", "default_profile_image": 1, "profile_use_background_image": 0, "verified": 0,
      "description": "", "created_at": "2025-04-18"}])
custom_val['bio_length']=custom_val['description'].str.len()
custom_val["created_at"]=pd.to_datetime(custom_val["created_at"])
custom_val['account_age_days'] = (custom_val['created_at'].max() - custom_val['created_at']).dt.days
custom_val["follower_to_following"]=custom_val['followers_count']/(custom_val['friends_count']+1)
custom_val['has_location'] = custom_val['location'].notnull().astype(int)
# X_check= data_unloading(custom_val)


X_check = custom_val[features_list]
print(X_check)

#forest model to compare

from sklearn.ensemble import RandomForestRegressor
forest= RandomForestRegressor(random_state=1,)
forest.fit(X,y)
predictions_forest = forest.predict(X_check)
prediction_main = XGB_model.predict(X_check)
print(predictions_forest)
print(prediction_main)
if prediction_main[0] > 0.5:
    print("Bot Detected")
else:
    print("Human Detected")

#saving for web below code is ai

model_data = {
    'model': XGB_model,
    'features': features_list,
}
joblib.dump(model_data, 'bot_detector.pkl')