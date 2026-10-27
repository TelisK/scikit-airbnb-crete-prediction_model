from django.shortcuts import render, redirect
from .forms import DataUploadForm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from django.contrib import messages
from . import processing
import hashlib
from . import models

# Create your views here.
# @login_required
def analyse_data(request):
    if request.method == 'POST':
        form = DataUploadForm(request.POST, request.FILES)

        if form.is_valid():

            try:
                file = request.FILES['file']
                print('app read the file')
                print(form.cleaned_data['file'].name)

                # Check the hash of the file to protect from double register.
                digest = hashlib.file_digest(file, "sha256").hexdigest()
                file.seek(0)
                print('hashlib is completed')

                if models.UploadedFile.objects.filter(file_hash=digest).exists():
                    print('inside if check for existing file')
                    messages.error(request, 'Το αρχείο είναι ήδη καταχωρημένο')
                    return redirect('price_prediction:analyse_data')

                try:
                    print('inside try go to the function to analyse')
                    file_listings, active_listings, df_for_model = processing.clean_data(file)
                    print(f'function completed.{file_listings}')
                    print('create record on db')
                    models.UploadedFile.objects.create(uploaded_file=file, file_hash=digest)

                except Exception as e:
                    messages.error(request, f'Σφάλμα στην επεξεργασία δεδομένων. {e}')
                    return redirect('price_prediction:analyse_data')

                context_to_html = {
                    'file_listings':file_listings,
                    'active_listings':active_listings,
                    'result':True,
                }

                return render(request, 'price_prediction/analyse_data.html', context=context_to_html)
            except Exception as e:
                messages.error(request, e)

        else:
            messages.error(request, 'Συνέβη σφάλμα στη φόρμα')
            return render(request, 'price_prediction/analyse_data.html', context={'form':form})


    else:
        form = DataUploadForm()

    return render(request, 'price_prediction/analyse_data.html', context={'form':form})


# @login_required
def import_data(request):
    pass

            

        # add to the database
    # try:
    
    # else:
    #     messages.error(request, f'{form.errors}')


def model_train(request, df_for_model):

    # splitting the data
    X = df_for_model.drop('price',axis=1)
    y = df_for_model['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

    # Training
    model = RandomForestRegressor(n_estimators=50, random_state=69)
    model.fit(X_train,y_train)
    messages.success(request, 'Model Trained!')

    joblib.dump(model, 'model.pkl')
    messages.success(request, f'Model Saved!')


def prediction():
    pass


# το prediction model θα δημιουργεί το transformer.pkl. Να το αποθηκεύω στη βάση; Ή να το δίνω με import απο εξωτερικό αρχείο;
# Ε΄άν θέλω στο μέλλον διαδραστικό χάρτη με περι΄΄οχές πώς θα εξυπηρετούσε να γίνει;
# Μπορεί να εμφανίζει στο χάρτη τα δεδομένα της βάσης. Τη πρόβλεψη καθώς και το σημείο του ενδιαφερόμενου θα δίνεται
# απο το prediction view ;;;