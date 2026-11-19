# Airbnb Income Estimator

This project was created to estimate the price / night for an AirBnB listing, based on the location and the house features.

This is the new version of the application, running in Django.
The original backend version and first approach can be found in the [legacy](legacy/) folder.

## About

The admin can upload listing files from [insideairbnb.com/](https://insideairbnb.com) of *any location* to the application, and train the model. Based on these data, it can make price predictions for accommodation.

Map : [OpenFreeMap](https://openfreemap.org/)

Deployed application for Crete : 

**Note:** The application interface is in Greek, designed for Greek-speaking users.

## Features

- Simple web interface
- Machine learning predictions (scikit-learn RandomForestRegressor)
- User authentication (only admin has access to load files and train the model)

## Future Features

- Prediction of estimated occupancy.
- Prediction of estimated annual income.
- More efficient ways to check when the user is searching away from areas with available data.

## Tech Stack

- **Python 3.12.3**
- **Backend:** Django 6.1.1
- **Database:** Sqlite
- **Frontend:** HTML, CSS
- **Machine Learning:** scikit-learn, pandas
- **Server:** Ubuntu, Nginx + Gunicorn
- **Deployment:** Docker Compose, Cloudflare Tunnel

## Setup

- clone και virtual environment
- pip install -r requirements.txt


### Local Development

- python manage.py migrate 
- python manage.py createsuperuser
- python manage.py runserver

## Live





## Author

Tilemachos Koutsogiannakis  
GitHub: [@TelisK](https://github.com/TelisK)

## License

This project is licensed under the MIT License.