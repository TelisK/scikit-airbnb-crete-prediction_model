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
    amenities = models.CharField()
    description = models.CharField()
    host_is_superhost = models.BooleanField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    beds = models.FloatField()
    accommodates = models.FloatField()
    price = models.DecimalField(max_digits=10, decimal_places=5)
    minimum_nights = models.IntegerField()
    number_of_reviews = models.IntegerField()
    review_scores_rating = models.FloatField()
    estimated_occupancy_l365d = models.IntegerField()
    instant_bookable = models.BooleanField()
    property_type = models.CharField()
    neighbourhood_cleansed = models.CharField()
    host_response_time = models.TimeField()

