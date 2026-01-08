import shutil
from django.shortcuts import render
from sklearn.model_selection import train_test_split



from .models import userRegisteredTable
from django.core.exceptions import ValidationError
from django.contrib import messages


def userRegisterCheck(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        username = request.POST.get("loginId")
        mobile = request.POST.get("mobile")
        password = request.POST.get("password")
        

        # Create an instance of the model
        user = userRegisteredTable(
            name=name,
            email=email,
            loginid=username,
            mobile=mobile,
            password=password,
            
        )

        try:
            # Validate using model field validators
            user.full_clean()
            
            # Save to DB
            user.save()
            messages.success(request,'registration Successfully done,please wait for admin APPROVAL')
            return render(request, "userRegisterForm.html")


        except ValidationError as ve:
            # Get a list of error messages to display
            error_messages = []
            for field, errors in ve.message_dict.items():
                for error in errors:
                    error_messages.append(f"{field.capitalize()}: {error}")
            return render(request, "userRegisterForm.html", {"messages": error_messages})

        except Exception as e:
            # Handle other exceptions (like unique constraint fails)
            return render(request, "userRegisterForm.html", {"messages": [str(e)]})

    return render(request, "userRegisterForm.html")


def userLoginCheck(request):
    if request.method=='POST':
        username=request.POST['userUsername']
        password=request.POST['userPassword']

        try:
            user=userRegisteredTable.objects.get(loginid=username,password=password)

            if user.status=='Active':
                request.session['id']=user.id
                request.session['name']=user.name
                request.session['email']=user.email
                
                return render(request,'users/userHome.html')
            else:
                messages.error(request,'Status not activated please wait for admin approval')
                return render(request,'userLoginForm.html')
        except:
            messages.error(request,'Invalid details please enter details carefully or Please Register')
            return render(request,'userLoginForm.html')
    return render(request,'userLoginForm.html')


def userHome(request):
    if not request.session.get('id'):
        return render(request,'userLoginForm.html')
    return render(request,'users/userHome.html')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

def training(request):
    
    # # Load dataset
    # data = pd.read_csv('media/balanced_30000_transaction_data.csv')

    # # Features and target
    # X = data.drop('label', axis=1)
    # y = data['label']

    # # Split data into train, validation, and test sets
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # # Standardize features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    # # Save the scaler
    # joblib.dump(scaler, 'media/scaler.pkl')

    # # Apply Random Oversampling for _ros variants
    # ros = RandomOverSampler(random_state=42)
    # X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    # # Dictionary to store metrics and models
    # results = {}
    # models = {}

    # # K-Nearest Neighbors
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(X_train, y_train)
    # y_pred_knn = knn.predict(X_val)
    # models['KNN'] = knn
    # results['KNN'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_knn),
    #     'precision': precision_score(y_val, y_pred_knn, average='binary'),
    #     'recall': recall_score(y_val, y_pred_knn, average='binary'),
    #     'f1': f1_score(y_val, y_pred_knn, average='binary')
    # }

    # # KNN with Random Oversampling
    # knn_ros = KNeighborsClassifier(n_neighbors=5)
    # knn_ros.fit(X_train_ros, y_train_ros)
    # y_pred_knn_ros = knn_ros.predict(X_val)
    # models['KNN_ros'] = knn_ros
    # results['KNN_ros'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_knn_ros),
    #     'precision': precision_score(y_val, y_pred_knn_ros, average='binary'),
    #     'recall': recall_score(y_val, y_pred_knn_ros, average='binary'),
    #     'f1': f1_score(y_val, y_pred_knn_ros, average='binary')
    # }

    # # Random Forest
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    # y_pred_rf = rf.predict(X_val)
    # models['RF'] = rf
    # results['RF'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_rf),
    #     'precision': precision_score(y_val, y_pred_rf, average='binary'),
    #     'recall': recall_score(y_val, y_pred_rf, average='binary'),
    #     'f1': f1_score(y_val, y_pred_rf, average='binary')
    # }

    # # Random Forest with Random Oversampling
    # rf_ros = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_ros.fit(X_train_ros, y_train_ros)
    # y_pred_rf_ros = rf_ros.predict(X_val)
    # models['RF_ros'] = rf_ros
    # results['RF_ros'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_rf_ros),
    #     'precision': precision_score(y_val, y_pred_rf_ros, average='binary'),
    #     'recall': recall_score(y_val, y_pred_rf_ros, average='binary'),
    #     'f1': f1_score(y_val, y_pred_rf_ros, average='binary')
    # }

    # # Neural Network (MLP)
    # mlp = Sequential([
    #     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    # mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # checkpoint_mlp = ModelCheckpoint('best_mlp_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    # mlp.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, 
    #         callbacks=[checkpoint_mlp], verbose=0)
    # y_pred_mlp = (mlp.predict(X_val) > 0.5).astype(int).flatten()
    # models['MLP'] = mlp
    # results['MLP'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_mlp),
    #     'precision': precision_score(y_val, y_pred_mlp, average='binary'),
    #     'recall': recall_score(y_val, y_pred_mlp, average='binary'),
    #     'f1': f1_score(y_val, y_pred_mlp, average='binary')
    # }

    # # Neural Network (MLP) with Random Oversampling
    # mlp_ros = Sequential([
    #     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    # mlp_ros.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # checkpoint_mlp_ros = ModelCheckpoint('best_mlp_ros_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    # mlp_ros.fit(X_train_ros, y_train_ros, validation_data=(X_val, y_val), epochs=20, batch_size=32, 
    #             callbacks=[checkpoint_mlp_ros], verbose=0)
    # y_pred_mlp_ros = (mlp_ros.predict(X_val) > 0.5).astype(int).flatten()
    # models['MLP_ros'] = mlp_ros
    # results['MLP_ros'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_mlp_ros),
    #     'precision': precision_score(y_val, y_pred_mlp_ros, average='binary'),
    #     'recall': recall_score(y_val, y_pred_mlp_ros, average='binary'),
    #     'f1': f1_score(y_val, y_pred_mlp_ros, average='binary')
    # }

    # # Support Vector Machine
    # svm = SVC(kernel='rbf', probability=True, random_state=42)
    # svm.fit(X_train, y_train)
    # y_pred_svm = svm.predict(X_val)
    # models['SVM'] = svm
    # results['SVM'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_svm),
    #     'precision': precision_score(y_val, y_pred_svm, average='binary'),
    #     'recall': recall_score(y_val, y_pred_svm, average='binary'),
    #     'f1': f1_score(y_val, y_pred_svm, average='binary')
    # }

    # # Support Vector Machine with Random Oversampling
    # svm_ros = SVC(kernel='rbf', probability=True, random_state=42)
    # svm_ros.fit(X_train_ros, y_train_ros)
    # y_pred_svm_ros = svm_ros.predict(X_val)
    # models['SVM_ros'] = svm_ros
    # results['SVM_ros'] = {
    #     'accuracy': accuracy_score(y_val, y_pred_svm_ros),
    #     'precision': precision_score(y_val, y_pred_svm_ros, average='binary'),
    #     'recall': recall_score(y_val, y_pred_svm_ros, average='binary'),
    #     'f1': f1_score(y_val, y_pred_svm_ros, average='binary')
    # }

    # # Save all models
    # os.makedirs('models', exist_ok=True)
    # joblib.dump(models['KNN'], 'models/knn.pkl')
    # joblib.dump(models['KNN_ros'], 'models/knn_ros.pkl')
    # joblib.dump(models['RF'], 'models/random_forest.pkl')
    # joblib.dump(models['RF_ros'], 'models/random_forest_ros.pkl')
    # models['MLP'].save('models/mlp.keras')
    # models['MLP_ros'].save('models/mlp_ros.keras')
    # joblib.dump(models['SVM'], 'models/svm.pkl')
    # joblib.dump(models['SVM_ros'], 'models/svm_ros.pkl')

    # # Find the best model based on validation F1-score
    # best_model_name = max(results, key=lambda x: results[x]['f1'])
    # print(f"Best model: {best_model_name}")

    # # Store validation metrics in a DataFrame
    # validation_metrics = []
    # for model_name, metrics in results.items():
    #     validation_metrics.append({
    #         'Model': model_name,
    #         'Accuracy': metrics['accuracy'],
    #         'Precision': metrics['precision'],
    #         'Recall': metrics['recall'],
    #         'F1_Score': metrics['f1']
    #     })
    # validation_df = pd.DataFrame(validation_metrics)
    # validation_df.to_csv('validation_metrics.csv', index=False)

    # # Print validation metrics
    # print("Validation Metrics:")
    # for model_name, metrics in results.items():
    #     print(f"\n{model_name}:")
    #     for metric, value in metrics.items():
    #         print(f"  {metric}: {value:.4f}")

    # # Save the best model separately
    # if 'MLP' in best_model_name:
    #     os.system(f'cp models/{best_model_name.lower()}.keras models/best_model.keras')
    # else:
    #     joblib.dump(models[best_model_name], 'models/best_model.pkl')

    # # Evaluate best model on test set
    # if 'MLP' in best_model_name:
    #     best_model = models[best_model_name]
    #     y_pred_test = (best_model.predict(X_test) > 0.5).astype(int).flatten()
    # else:
    #     best_model = models[best_model_name]
    #     y_pred_test = best_model.predict(X_test)

    # test_metrics = {
    #     'accuracy': accuracy_score(y_test, y_pred_test),
    #     'precision': precision_score(y_test, y_pred_test, average='binary'),
    #     'recall': recall_score(y_test, y_pred_test, average='binary'),
    #     'f1': f1_score(y_test, y_pred_test, average='binary')
    # }

    # # Store test metrics in a DataFrame
    # test_metrics_df = pd.DataFrame([{
    #     'Model': best_model_name,
    #     'Accuracy': test_metrics['accuracy'],
    #     'Precision': test_metrics['precision'],
    #     'Recall': test_metrics['recall'],
    #     'F1_Score': test_metrics['f1']
    # }])
    # test_metrics_df.to_csv('media/test_metrics.csv', index=False)

    # # Print test metrics
    # print("\nTest Metrics for Best Model:")
    # for metric, value in test_metrics.items():
    #     print(f"  {metric}: {value:.4f}")
    results1=pd.read_csv(r'validation_metrics.csv')
    dff=results1.to_html()
    # Pass DataFrame to template (convert to dict for easier rendering)
    return render(request, 'users/training.html', {
         
        'results_df':dff  # Convert DataFrame to list of dictionaries
    })

import numpy as np
import joblib
import os
from django.shortcuts import render
from tensorflow.keras.models import load_model

def prediction(request):
    if request.method == 'POST':
        try:
            # Load scaler and model
            scaler = joblib.load('media/scaler.pkl')
            if os.path.exists('models/best_model.keras'):
                model = load_model('models/best_model.keras')
                is_keras_model = True
            else:
                model = joblib.load('models/best_model.pkl')
                is_keras_model = False

            # Extract features
            features = [
                float(request.POST['mean_amount']),
                float(request.POST['std_amount']),
                float(request.POST['skewness']),
                float(request.POST['kurtosis']),
                float(request.POST['transaction_count']),
                float(request.POST['median_inter_time']),
                float(request.POST['std_inter_time']),
                float(request.POST['autocorr_lag1']),
                float(request.POST['dominant_freq'])
            ]
            input_data = np.array(features).reshape(1, -1)

            # Scale and predict
            input_scaled = scaler.transform(input_data)
            if is_keras_model:
                prediction = (model.predict(input_scaled, verbose=0) > 0.5).astype(int).flatten()[0]
            else:
                prediction = model.predict(input_scaled)[0]

            result_label = 'Fraudulent' if prediction == 1 else 'Non-fraudulent'
            return render(request, 'users/prediction.html', {
                'result': prediction,
                'result_label': result_label
            })
        except Exception as e:
            return render(request, 'users/prediction.html', {'error': str(e)})
    return render(request, 'users/prediction.html')