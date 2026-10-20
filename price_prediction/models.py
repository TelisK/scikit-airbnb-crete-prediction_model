from django.db import models

# Create your models here.
class DataSet(models.Model):
    amenities = models.CharField()
    description = models.CharField()
    host_is_superhost = models.BooleanField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    beds = models.FloatField()
    accommodates = models.CharField()
    price = models.DecimalField(max_digits=10, decimal_places=5)
    minimum_nights = models.IntegerField()
    number_of_reviews = models.IntegerField()
    review_scores_rating = models.FloatField()
    estimated_occupancy_l365d = models.IntegerField()
    instant_bookable = models.BooleanField()
    property_type = models.CharField()
    neighbourhood_cleansed = models.CharField()
    host_response_time = models.TimeField()
