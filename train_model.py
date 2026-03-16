import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# load data
df = pd.read_csv("data/sales_data.csv")

df = df.dropna()

# convert day to number
day_map = {
    "monday":1,"tuesday":2,"wednesday":3,
    "thursday":4,"friday":5,"saturday":6,"sunday":7
}

df["day_num"] = df["day"].map(day_map)

# encode string features
weather_encoder = LabelEncoder()
season_encoder = LabelEncoder()
festival_encoder = LabelEncoder()

df["weather_enc"] = weather_encoder.fit_transform(df["weather"])
df["season_enc"] = season_encoder.fit_transform(df["season"])
df["festival_enc"] = festival_encoder.fit_transform(df["festival"])

# features
X = df[["hour","day_num","stock_start","price","weather_enc","season_enc","festival_enc"]]

# target
y = df["sold"]

# train model
model = RandomForestRegressor()
model.fit(X,y)

# save model and encoders
model_data = {
    "model": model,
    "weather_encoder": weather_encoder,
    "season_encoder": season_encoder,
    "festival_encoder": festival_encoder
}
pickle.dump(model_data, open("demand_model.pkl","wb"))

print("Model trained successfully")