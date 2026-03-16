import pickle
import numpy as np

model_data = pickle.load(open("demand_model.pkl","rb"))

model = model_data["model"]
weather_encoder = model_data["weather_encoder"]
season_encoder = model_data["season_encoder"]
festival_encoder = model_data["festival_encoder"]

def predict_demand(hour, day_num, stock_start, price, weather, season, festival):
    # Helper to safely transform, defaulting to first class if unseen
    def safe_transform(encoder, val, fallback=None):
        try:
            return encoder.transform([val])[0]
        except ValueError:
            # If the category wasn't seen during training, fallback
            return encoder.transform([fallback or encoder.classes_[0]])[0]

    weather_enc = safe_transform(weather_encoder, weather, "sunny")
    season_enc = safe_transform(season_encoder, season, "winter")
    festival_enc = safe_transform(festival_encoder, festival, "none")

    features = np.array([[hour, day_num, stock_start, price, weather_enc, season_enc, festival_enc]])

    prediction = model.predict(features)

    return prediction[0]