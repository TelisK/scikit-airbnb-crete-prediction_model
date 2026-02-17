import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('listings.csv.gz')
df = pd.DataFrame(data)

# keeping the data we are interested in
df_for_model = df[['amenities','description','host_is_superhost',
                   'latitude','longitude','beds','accommodates','price','minimum_nights',
                   'number_of_reviews','review_scores_rating','estimated_occupancy_l365d',
                   'instant_bookable', 'property_type','neighbourhood_cleansed',
                   'host_response_time']].copy()

# making new columns automated, with the information inside these columns. A way
# of making one hot encoder with pandas

# df_for_model = pd.get_dummies(df_for_model, columns=['instant_bookable', 'property_type',
#                                                       'neighbourhood_cleansed', 
#                                                       'host_response_time'])


# amenities categorize
df_for_model['amenities'] = df_for_model['amenities'].str.lower()
df_for_model['has_pool'] = df_for_model['amenities'].str.contains('pool').astype(int)
df_for_model['amenities_has_seaview'] = df_for_model['amenities'].str.contains('sea view|seaview|view at the sea|view at the beach|ocean view|beachfront|waterfront', na=False).astype(int)
df_for_model['has_air_condition'] = df_for_model['amenities'].str.contains('air condition').astype(int)
df_for_model['has_free_parking'] = df_for_model['amenities'].str.contains('free parking').astype(int)
df_for_model['has_kitchen'] = df_for_model['amenities'].str.contains('kitchen|cooking|toaster', na=False).astype(int)
df_for_model['has_safe'] = df_for_model['amenities'].str.contains('safe').astype(int)
df_for_model['has_hot_water'] = df_for_model['amenities'].str.contains('hot water').astype(int)
df_for_model['has_hair_dryer'] = df_for_model['amenities'].str.contains('hair dryer').astype(int)
df_for_model['has_mountain_view'] = df_for_model['amenities'].str.contains('mountain view').astype(int)
df_for_model['has_fireplace'] = df_for_model['amenities'].str.contains('fireplace').astype(int)


# drop amenities
df_for_model = df_for_model.drop('amenities', axis=1)

# description categorize
df_for_model['description'] = df_for_model['description'].str.lower()
df_for_model['description_has_seaview'] = df_for_model['description'].str.contains('sea view|seaview|view at the sea|view at the beach|ocean view|beachfront|waterfront', na=False).astype(int)

# drop description
df_for_model = df_for_model.drop('description', axis=1)



# have to remove the data without price
df_for_model = df_for_model[df_for_model['price'].notna()]

# filling NaN with data
# filling the beds column the median of beds
df_for_model['beds'] = df_for_model['beds'].fillna(df_for_model['beds'].median())
# filling the 'host_is_superhost' column with false
df_for_model['host_is_superhost'] = df_for_model['host_is_superhost'].fillna('f')
# filling the 'review_scores_rating' with 0 if the listing has no reviews at all
df_for_model.loc[(df_for_model['number_of_reviews'] == 0 & df_for_model['review_scores_rating'].isna()), 'review_scores_rating'] = 0

# replacing t with 1 and f with 0 (true and false) on column 'host_is_superhost'
df_for_model['host_is_superhost'] = df_for_model['host_is_superhost'].map({'t': 1, 'f': 0})

# corrections on the price column
df_for_model['price'] = df_for_model['price'].str.replace('$','')
df_for_model['price'] = df_for_model['price'].str.replace(',','')
df_for_model['price'] = df_for_model['price'].astype(float)

# work with prices lower than 400€
df_for_model = df_for_model.loc[df_for_model['price'] <= 400]

# WE HAVE ZEROS
#print(df_for_model.isna().sum())

# splitting the data
X = df_for_model.drop('price',axis=1)
y = df_for_model['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# One Hot Encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

hot_features = ['instant_bookable','property_type','neighbourhood_cleansed','host_response_time']

one_hot = OneHotEncoder(handle_unknown='ignore')
transformer = ColumnTransformer(transformers=[('one_hot',one_hot,hot_features)], remainder='passthrough')

X_train_encoded = transformer.fit_transform(X_train)
X_test_encoded = transformer.transform(X_test)

# Training
model = RandomForestRegressor(random_state=69)
model.fit(X_train_encoded,y_train)

print('Model Trained!')

# Prediction
y_predict = model.predict(X_test_encoded)

# Checking
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'Mean Absolute Error: {mae:.2f} €')
print(f'R2 Score: {r2:.2f}')

# Save trained model
import joblib

joblib.dump(model, 'model.pkl')
joblib.dump(transformer, 'transformer.pkl')
print('Model Saved!')