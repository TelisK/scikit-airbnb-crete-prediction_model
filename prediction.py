import joblib
import pandas as pd

model = joblib.load('model.pkl')
transformer = joblib.load('transformer.pkl')

def predict_price(
    neighbourhood, latitude, longitude,
    property_type, accommodates, beds,
    minimum_nights, has_pool, has_air_condition,
    has_free_parking, has_kitchen, host_is_superhost=0,
    instant_bookable='t', host_response_time='within an hour',
    number_of_reviews=0, review_scores_rating=4.5,
    estimated_occupancy_l365d=100,
    amenities_has_seaview=0, description_has_seaview=0,
    has_safe=0, has_hot_water=1, has_hair_dryer=0,
    has_mountain_view=0, has_fireplace=0
):
    input_data = pd.DataFrame([{
        'host_is_superhost': host_is_superhost,
        'latitude': latitude,
        'longitude': longitude,
        'beds': beds,
        'accommodates': accommodates,
        'minimum_nights': minimum_nights,
        'number_of_reviews': number_of_reviews,
        'review_scores_rating': review_scores_rating,
        'estimated_occupancy_l365d': estimated_occupancy_l365d,
        'instant_bookable': instant_bookable,
        'property_type': property_type,
        'neighbourhood_cleansed': neighbourhood,
        'host_response_time': host_response_time,
        'has_pool': has_pool,
        'amenities_has_seaview': amenities_has_seaview,
        'has_air_condition': has_air_condition,
        'has_free_parking': has_free_parking,
        'has_kitchen': has_kitchen,
        'has_safe': has_safe,
        'has_hot_water': has_hot_water,
        'has_hair_dryer': has_hair_dryer,
        'has_mountain_view': has_mountain_view,
        'has_fireplace': has_fireplace,
        'description_has_seaview': description_has_seaview,
    }])

    input_encoded = transformer.transform(input_data)
    prediction = model.predict(input_encoded)[0]
    
    return prediction



price = predict_price(
    host_is_superhost=1,
    neighbourhood='Chania Old Town',
    latitude=35.515305,
    longitude=24.019071,
    property_type='Entire rental unit',
    accommodates=4,
    beds=3,
    minimum_nights=2,
    has_pool=1,
    has_air_condition=1,
    has_free_parking=1,
    has_kitchen=1,
    description_has_seaview=1,
)

print(f'Price Prediction: {price}€/night')