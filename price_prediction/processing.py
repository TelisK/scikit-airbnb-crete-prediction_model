import pandas as pd
import gzip


def clean_data(uploaded_file):
    # use of rt because is text not binary
    file = gzip.open(uploaded_file, mode='rt', encoding='utf-8')

    data = pd.read_csv(file)
    df = pd.DataFrame(data)

    # keeping the data we are interested in
    df_for_model = df[['amenities','description',
                    'latitude','longitude','beds','accommodates','price','minimum_nights',                    
                    'property_type']].copy()

    #removed:
    #'host_is_superhost','number_of_reviews','review_scores_rating','estimated_occupancy_l365d','instant_bookable','neighbourhood_cleansed','host_response_time'
    
    file_listings = df.count().id

    # remove data without price
    df_for_model = df_for_model[df_for_model['price'].notna()]

    # use of case=False can handle capital letters, and na=False writes 0 instead of NA.
    df_for_model['has_pool'] = df_for_model['amenities'].str.contains('pool', na=False, case=False).astype(int)
    df_for_model['amenities_has_seaview'] = df_for_model['amenities'].str.contains('sea view|seaview|view at the sea|' \
    'view at the beach|ocean view|beachfront|waterfront', na=False, case=False).astype(int)
    df_for_model['has_air_condition'] = df_for_model['amenities'].str.contains('air condition', na=False, case=False).astype(int)
    df_for_model['has_free_parking'] = df_for_model['amenities'].str.contains('free parking', na=False, case=False).astype(int)
    df_for_model['has_kitchen'] = df_for_model['amenities'].str.contains('kitchen|cooking|toaster', na=False, case=False).astype(int)
    df_for_model['has_safe'] = df_for_model['amenities'].str.contains('safe', na=False, case=False).astype(int)
    df_for_model['has_hot_water'] = df_for_model['amenities'].str.contains('hot water', na=False, case=False).astype(int)
    df_for_model['has_hair_dryer'] = df_for_model['amenities'].str.contains('hair dryer', na=False, case=False).astype(int)
    df_for_model['has_mountain_view'] = df_for_model['amenities'].str.contains('mountain view', na=False, case=False).astype(int)
    df_for_model['has_fireplace'] = df_for_model['amenities'].str.contains('fireplace', na=False, case=False).astype(int)
    df_for_model['is_villa'] = df_for_model['description'].str.contains('Villa', na=False, case=False).astype(int)
    df_for_model['is_in_old_town'] = df_for_model['description'].str.contains('old town', na=False, case=False).astype(int)
    df_for_model['description_has_seaview'] = df_for_model['description'].str.contains('sea view|seaview|view at the sea|' \
    'view at the beach|ocean view|beachfront|waterfront', na=False, case=False).astype(int)


    # filter for property type
    df_for_model['entire_place'] = df_for_model['property_type'].str.contains('entire', na=False, case=False).astype(int)
    df_for_model['private_room'] = df_for_model['property_type'].str.contains('private room', na=False, case=False).astype(int)
    df_for_model['shared_room'] = df_for_model['property_type'].str.contains('shared room|hostel', na=False, case=False).astype(int)
    df_for_model['alternative'] = df_for_model['property_type'].str.contains('camper|boat|' \
    'campsite|tent|hut|yurt|treehouse|bus|container', na=False, case=False).astype(int)
    df_for_model['luxury'] = df_for_model['property_type'].str.contains('villa|castle|island', na=False, case=False).astype(int)
    df_for_model['hotel_room'] = df_for_model['property_type'].str.contains('hotel|aparthotel|pension|resort', na=False, case=False).astype(int)



    df_for_model = df_for_model.drop(['amenities','description','property_type'], axis=1)

    # fill nan cells on beds with the median.
    df_for_model['beds'] = df_for_model['beds'].fillna(df_for_model['beds'].median())
    df_for_model['minimum_nights'] = df_for_model['minimum_nights'].fillna(df_for_model['minimum_nights'].median())


    # corrections on the price column
    df_for_model['price'] = df_for_model['price'].str.replace('$','')
    df_for_model['price'] = df_for_model['price'].str.replace(',','')
    df_for_model['price'] = df_for_model['price'].astype(float)

    active_listings = df_for_model.count().price

    return file_listings, active_listings, df_for_model