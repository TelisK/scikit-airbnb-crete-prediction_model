import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv('listings.csv.gz')
df = pd.DataFrame(data)

# keeping the data we are interested in
df_for_model = df[['id','host_is_superhost','latitude','longitude','beds','price','minimum_nights','number_of_reviews','review_scores_rating']]

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

# WE HAVE ZEROS
#print(df_for_model.isna().sum())

# splitting the data
X = df_for_model.drop('price',axis=1)
y = df_for_model['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Training
model = RandomForestRegressor(random_state=69)
model.fit(X_train,y_train)

print('Model Trained!')

# Prediction
y_predict = model.predict(X_test)

# Checking
mae = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f'Mean Absolute Error: {mae:.2f} €')
print(f'R2 Score: {r2:.2f}')