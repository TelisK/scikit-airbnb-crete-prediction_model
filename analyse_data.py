import pandas as pd


data = pd.read_csv('listings.csv.gz')
df = pd.DataFrame(data)

# keeping the data we are interested in
df_for_model = df[['id','host_is_superhost','latitude','longitude','property_type','beds','price','minimum_nights','number_of_reviews','review_scores_rating']]

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


# WE HAVE ZEROS
print(df_for_model.isna().sum())

# splitting the data
X = df_for_model.drop('price',axis=1)
y = df_for_model['price']

