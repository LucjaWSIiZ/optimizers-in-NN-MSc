import os
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Lion, Optimizer
from openpyxl import Workbook

# Funkcja do budowania i trenowania modelu,
# zwraca history, accuracy, class_report, training_time, testing_time, conf_matrix
def run_model(X_train, y_train, X_val, y_val, X_test, y_test, optimizer_name, epochs=24, iteration=0):
    model = Sequential()
    model.add(Dense(8, input_dim=X_train.shape[1]))
    model.add(Dense(17, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))

    optimizer = eval(optimizer_name)()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    start_training_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32)
    end_training_time = time.time()

    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    model.save(f'models/{optimizer_name}/nn_{iteration}.h5')

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    class_report = classification_report(y_test_classes, y_pred_classes, target_names=species, output_dict=True)

    training_time = end_training_time - start_training_time
    testing_time = end_time - start_time

    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    return history, accuracy, class_report, training_time, testing_time, conf_matrix

# Załadowanie zbioru danych
df = pd.read_excel('../Dry_Bean_Dataset.xlsx')

# Zakodowanie klas na numeryczne etykiety
species = df['Class'].unique()
label_mapping = {species[i]: i for i in range(len(species))}
df['Class'] = df['Class'].map(label_mapping)

# Kodowanie one-hot
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(df[['Class']])

# Wybór cech
selected_features = ['MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Extent', 'Solidity', 'roundness', 'ShapeFactor2', 'ShapeFactor4']
X = df[selected_features]

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n = 200 # Ilosc powtorzen
optimizers = ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Lion'] # Lista algorytmów optimizacji

# Utworzenie folderu 'stale-zbiory'
output_dir = 'stale-zbiory'
os.makedirs(output_dir, exist_ok=True)

# Utworzenie podziału danych na zbiór treningowy, walidacyjny i testowy
for i in range(n):
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2)
    split_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    # Zapis podzielonych danych do pliku .xlsx
    with pd.ExcelWriter(os.path.join(output_dir, f'data_split_{i + 1}.xlsx'), engine='openpyxl') as writer:
        for key, value in split_data.items():
            pd.DataFrame(value).to_excel(writer, sheet_name=key, index=False)

# Trenowanie i ewaluacja modelu dla poszczególnych algorytmów
for optimizer_name in optimizers:
    summary_data = []
    with pd.ExcelWriter(f'etap-2-{optimizer_name}.xlsx', engine='openpyxl') as writer:
        for i in range(n):
            print()
            print('-' * 50)
            print(f'Iteration {i + 1}/{n} using {optimizer_name}')
            print('Dat file: ', os.path.join(output_dir, f'data_split_{i + 1}.xlsx'))

            # Załadowanie danych z odpowiedniego pliku .xlsx
            with pd.ExcelFile(os.path.join(output_dir, f'data_split_{i + 1}.xlsx')) as split_file:
                X_train = pd.read_excel(split_file, sheet_name='X_train').values
                X_val = pd.read_excel(split_file, sheet_name='X_val').values
                X_test = pd.read_excel(split_file, sheet_name='X_test').values
                y_train = pd.read_excel(split_file, sheet_name='y_train').values
                y_val = pd.read_excel(split_file, sheet_name='y_val').values
                y_test = pd.read_excel(split_file, sheet_name='y_test').values

            # Urtworzenie modelu
            history, accuracy, class_report, training_time, testing_time, conf_matrix = run_model(X_train, y_train, X_val, y_val, X_test, y_test, optimizer_name, epochs=24, iteration=i+1)

            # Przygotowanie danych do podsumowania dla każdej iteracji
            iteration_data = {
                'Iteration': i + 1,
                'Overall Accuracy': accuracy,
                'Training Time [s]': training_time,
                'Testing Time [s]': testing_time,
            }

            # Dodanie metryk dla każdej klasy
            for cls in species:
                iteration_data[f'{cls} Precision'] = class_report[cls]['precision']
                iteration_data[f'{cls} Recall'] = class_report[cls]['recall']
                iteration_data[f'{cls} F1-score'] = class_report[cls]['f1-score']

            # Obliczenie średnich wartości metryk
            precision_vals = [class_report[cls]['precision'] for cls in species]
            recall_vals = [class_report[cls]['recall'] for cls in species]
            f1_vals = [class_report[cls]['f1-score'] for cls in species]

            iteration_data['Average Precision'] = np.mean(precision_vals)
            iteration_data['Average Recall'] = np.mean(recall_vals)
            iteration_data['Average F1-score'] = np.mean(f1_vals)

            summary_data.append(iteration_data)

            # Zapis wyników epoki do osobnych arkuszy
            epoch_results = pd.DataFrame({
                'Epoch': list(range(1, 25)),
                'Loss': history.history['loss'],
                'Accuracy': history.history['accuracy'],
                'Val_Loss': history.history['val_loss'],
                'Val_Accuracy': history.history['val_accuracy']
            })

            epoch_results.to_excel(writer, sheet_name=f'Run_{i + 1}', index=False)

            # Zapis macierzy konfuzji do osobnych arkuszy
            conf_matrix_df = pd.DataFrame(conf_matrix, index=species, columns=species)
            conf_matrix_df.to_excel(writer, sheet_name=f'Conf_Matrix_{i + 1}', index=True)

        # Zapis podsumowania do arkusza
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"Training with {optimizer_name} completed and results saved to 'etap-2-{optimizer_name}.xlsx'")

print("All training processes with different optimizers are completed.")
