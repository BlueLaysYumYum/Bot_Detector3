import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from ml import custom_val

# 1. LOAD & CLEAN DATA
df = pd.read_csv("twitter_profiles.csv")

# Create numeric features from text and dates
df['bio_length'] = df['description'].astype(str).str.len()
df['name_length'] = df['screen_name'].astype(str).str.len()
df['created_at'] = pd.to_datetime(df['created_at'])
df['account_age_days'] = (df['created_at'].max() - df['created_at']).dt.days

# Select the features the model will use
features = [
    'followers_count', 'friends_count', 'post_count',
    'default_profile_image', 'profile_use_background_image',
    'verified', 'bio_length', 'name_length', 'account_age_days'
]

X = df[features]
y = df['label']
print(X.head())
# 2. INITIALIZE MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 3. BASIC CROSS VALIDATION
# cv=5 means it will split the data into 5 parts and test 5 times
scores = cross_val_score(model, X, y, cv=5)

# 4. OUTPUT RESULT
print(f"Scores for each fold: {scores}")
print(f"Average Accuracy: {scores.mean():.2%}")
bot_data = pd.read_csv("./twitter_profiles.csv")
bot_data['bio_length']=bot_data['description'].str.len()
bot_data["created_at"]=pd.to_datetime(bot_data["created_at"])
bot_data['account_age_days'] = (bot_data['created_at'].max() - bot_data['created_at']).dt.days
bot_data["follower_to_following"]=bot_data['followers_count']/(bot_data['friends_count']+1)
bot_data['has_location'] = bot_data['location'].notnull().astype(int)