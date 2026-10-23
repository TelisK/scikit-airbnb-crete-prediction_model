from django.shortcuts import render
from .forms import DataUploadForm
import pandas as pd

# Create your views here.
@login_required
def import_data(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']

            data = pd.read_csv(file)
            df = pd.DataFrame(data)

            # keeping the data we are interested in
            df_for_model = df[['amenities','description','host_is_superhost',
                            'latitude','longitude','beds','accommodates','price','minimum_nights',
                            'number_of_reviews','review_scores_rating','estimated_occupancy_l365d',
                            'instant_bookable', 'property_type','neighbourhood_cleansed',
                            'host_response_time']].copy()
            
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
            df_for_model['shared room'] = df_for_model['property_type'].str.contains('shared room|hostel', na=False, case=False).astype(int)
            df_for_model['alternative'] = df_for_model['property_type'].str.contains('camper|rv|boat|' \
            'campsite|tent|hut|yurt|treehouse|bus|container', na=False, case=False).astype(int)
            # the entire serviced apartment gets into alternative . why???
            df_for_model['luxury'] = df_for_model['property_type'].str.contains('villa|castle|island', na=False, case=False).astype(int)
            df_for_model['hotel_room'] = df_for_model['property_type'].str.contains('hotel|aparthotel|pension|resort', na=False, case=False).astype(int)



            df_for_model = df_for_model.drop(['amenities','description','property_type'], axis=1)

            # fill nan cells on beds with the median.
            df_for_model['beds'] = df_for_model['beds'].fillna(df_for_model['beds'].median())

            df_for_model['host_is_superhost'] = df_for_model['host_is_superhost'].fillna('f')

            df_for_model.loc[(df_for_model['number_of_reviews'] == 0 & df_for_model['review_scores_rating'].isna()), 'review_scores_rating'] = 0

            df_for_model['host_is_superhost'] = df_for_model['host_is_superhost'].map({'t': 1, 'f': 0})
            df_for_model['instant_bookable'] = df_for_model['instant_bookable'].map({'t': 1, 'f': 0})
            df_for_model['host_response_time'] = df_for_model['host_response_time'].map({
                'a few days or more' : 1,
                'within a day' : 2,
                'within a few hours' : 3,
                'within an hour' : 4,
            }).fillna(0)

            # corrections on the price column
            df_for_model['price'] = df_for_model['price'].str.replace('$','')
            df_for_model['price'] = df_for_model['price'].str.replace(',','')
            df_for_model['price'] = df_for_model['price'].astype(float)

            # work with prices lower than 400€
            #df_for_model = df_for_model.loc[df_for_model['price'] <= 400]

            active_listings = df_for_model.count().price


def prediction():
    pass


# το prediction model θα δημιουργεί το transformer.pkl. Να το αποθηκεύω στη βάση; Ή να το δίνω με import απο εξωτερικό αρχείο;
# Ε΄άν θέλω στο μέλλον διαδραστικό χάρτη με περι΄΄οχές πώς θα εξυπηρετούσε να γίνει;
# Μπορεί να εμφανίζει στο χάρτη τα δεδομένα της βάσης. Τη πρόβλεψη καθώς και το σημείο του ενδιαφερόμενου θα δίνεται
# απο το prediction view ;;;