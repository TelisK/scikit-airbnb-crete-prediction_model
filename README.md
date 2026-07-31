This project was created to help people who are thinking about renting their property on Airbnb estimate a suitable nightly price.

It uses a Random Forest Regressor to predict the nightly price of Airbnb listings based on property features and location.

The dataset used in this project is provided by http://insideairbnb.com.

## Files

- `analyse_data.py` - Loads and cleans the dataset, trains the machine learning model, evaluates it, and saves the trained model.
- `prediction.py` - Loads the saved model and predicts the price of a new Airbnb listing.
- `listings.csv.gz` - Dataset used for training.
- `model.pkl` - Saved trained model.
- `transformer.pkl` - Saved One-Hot Encoder transformer.

## Requirements

Install the required libraries:

```bash
pip install pandas scikit-learn joblib
```

## Train the Model

Run:

```bash
python analyse_data.py
```

This will:
- Load and clean the dataset.
- Create new features from amenities and descriptions.
- Train a Random Forest model.
- Evaluate the model.
- Save the trained model as `model.pkl` and `transformer.pkl`.

## Make Predictions

Run:

```bash
python prediction.py
```

The script loads the saved model and predicts the nightly price for a sample Airbnb listing.

## Model Features

The model uses information such as:

- Location (latitude, longitude, neighbourhood)
- Property type
- Number of beds
- Number of guests accommodated
- Minimum nights
- Superhost status
- Instant booking
- Host response time
- Review score
- Estimated occupancy
- Amenities (pool, kitchen, air conditioning, parking, fireplace, etc.)
- Sea view and mountain view

## Output

Example:

```text
Price Prediction: 145.32€/night
```