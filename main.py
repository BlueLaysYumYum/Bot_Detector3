import tweepy
import pandas as pd
import joblib
from datetime import datetime,timezone
from flask import Flask,jsonify,request
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)
CORS(app)

client = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAAHEU6QEAAAAALRxG9f1JcG3rN7Ln0tXV87HYGVk%3DAY4YxTQLsEWTA52kxDizSXxH2W5HYZRTbLKjZYa3aDyySsfuva",
    wait_on_rate_limit=True
)

@app.route("/getusername",methods=["POST"] )
def getusername():
    data=request.json
    username=data["username"].strip()

    user = client.get_user(
    username=username,
    user_fields=[
        "created_at",
        "public_metrics",
        "verified",
        "description",
        "profile_image_url",
        "protected",
        "location",
        # "profile_banner_url"
    ]
    )
    user_data = user.data
    
    if user_data is None:
        return jsonify({
            "error": "User not found",
            "result": -1
        })
    
    #user profile phot
    profile_link=user_data.profile_image_url
    is_defaults=1
    # link="https://abs.twimg.com/sticky/default_profile_images/default_profile_400x400.png"
    if "default_profile" in profile_link:
        is_defaults=1
    else:
        is_defaults=0

    #account age
    today=datetime.now(timezone.utc)
    account_age=user_data.created_at
    account_age_days=(today-account_age).days

    #posts count
    posts=user_data.public_metrics["tweet_count"]

    #bio length
    if user_data.description:
        bio_length=len(user_data.description)
    else:
        bio_length=0


    #follow ratio
    followers=user_data.public_metrics["followers_count"]
    following=user_data.public_metrics["following_count"]
    if following==0:
        follower_following_ratio=followers
    else:
        follower_following_ratio=followers/following


    #verified
    verification=user_data.verified
    verified=0
    if verification==True:
        verified=1

    #location
    location=user_data.location

    # profile background
    # profile_background_url=user_data.profile_banner_url

    # print("Username: @",username.strip())
    # print("Profile:", is_defaults)
    # print("followers:",followers)
    # print("following:",following)
    # print("follower ratio:",follower_following_ratio)
    # print("Post count:",posts)
    # print("account days created:",account_age_days)
    # print("bio length:",bio_length)
    # print("verified:",verified)
    # print("location:",location)
    # print("profile bg link:",profile_background_url)

    #Integration

    model_data = joblib.load("./bot_detector.pkl")
    model = model_data['model']
    features_list = model_data['features']
    # print(features_list)
    user_value = pd.DataFrame(
        [{"name":username.strip(),
          "default_profile_image":is_defaults,
          "followers_count":followers,
          "friends_count":following,
          "post_count":posts,
          "account_age_days":account_age_days,
          "bio_length":bio_length,
          "verified":verified,
          "profile_image_url":profile_link,
          "location":location,
          # 'profile_background_url':profile_background_url
        }]
        )
    # print(user_value)
    def data_unloading(data):
        data["follower_to_following"] = data['followers_count'] / (data['friends_count'] + 1)
        data['has_location'] = data['location'].notnull().astype(int)
        # data['profile_use_background_image'] = (data['profile_background_url']).astype(int)
        data["post_frequency"] = data['post_count'] / (data['account_age_days'] + 1)
        X_param = data[features_list]
        return X_param
    input_data = data_unloading(user_value)
    prediction = model.predict(input_data)
    probability_main = XGB_model.predict_proba(X_check)
    #percentage calc
    human_proba=probability_main[0][0]*100
    bot_proba=probability_main[0][1]*100
    result=int(prediction[0])
    return jsonify({
        "username":username,
        "result":result
    })

if __name__ == "__main__":
    app.run(debug=True)


