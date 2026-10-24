from django.db import models

# Create your models here.
class DataInformation(models.Model):
    # data import information
    created_at = models.DateTimeField(auto_now_add=True)
    data_country = models.CharField()
    data_area = models.CharField()
    data_town = models.CharField()
    data_date = models.DateField()
    file_listings = models.IntegerField()
    active_listings = models.IntegerField()

    def __str__(self):
        return f'{self.data_country} - {self.data_town} : Created at {self.created_at}, Listings {self.active_listings}/{self.file_listings}'

    
class DataSet(models.Model):
    data_information = models.ForeignKey(DataInformation, on_delete=models.CASCADE)
    amenities = models.TextField(blank=True)
    description = models.TextField(blank=True)
    host_is_superhost = models.BooleanField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    beds = models.FloatField()
    accommodates = models.IntegerField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    minimum_nights = models.IntegerField()
    number_of_reviews = models.IntegerField()
    review_scores_rating = models.FloatField(null=True)
    estimated_occupancy_l365d = models.IntegerField()
    instant_bookable = models.BooleanField()
    property_type = models.CharField()
    neighbourhood_cleansed = models.CharField()
    host_response_time = models.IntegerField(min=0, max=4)
    has_pool = models.BooleanField()
    amenities_has_seaview = models.BooleanField()
    has_air_condition = models.BooleanField()
    has_free_parking = models.BooleanField()
    has_kitchen = models.BooleanField()
    has_safe = models.BooleanField()
    has_hot_water = models.BooleanField()
    has_hair_dryer = models.BooleanField()
    has_mountain_view = models.BooleanField()
    has_fireplace = models.BooleanField()
    is_villa = models.BooleanField()
    is_in_old_town = models.BooleanField()
    description_has_seaview = models.BooleanField()
    entire_place = models.BooleanField()
    private_room = models.BooleanField()
    shared_room = models.BooleanField()
    alternative = models.BooleanField()
    luxury = models.BooleanField()
    hotel_room = models.BooleanField()