import os
import time

import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Lion, Optimizer
from openpyxl import Workbook


# Zdefiniowanie min_delta dla poszczególnych optimizatorów
min_delta_values = {
    'SGD': 0.0005,
    'RMSprop': 0.001,
    'Adam': 0.0005,
    'AdamW': 0.0005,
    'Adadelta': 0.0009,
    'Adagrad': 0.005,
    'Adamax': 0.0005,
    'Nadam': 0.0001,
    'Lion': 0.0005
}

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

    min_delta = min_delta_values[optimizer_name]
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=min_delta, restore_best_weights=True)

    start_training_time = time.time()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=32, callbacks=[early_stopping])
    end_training_time = time.time()

    start_testing_time = time.time()
    y_pred = model.predict(X_test)
    end_testing_time = time.time()
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    model.save(f'models/{optimizer_name}/nn_{iteration}.h5')

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    class_report = classification_report(y_test_classes, y_pred_classes, target_names=species, output_dict=True)

    training_time = end_training_time - start_training_time
    testing_time = end_testing_time - start_testing_time

    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

    return history, accuracy, class_report, training_time, testing_time, conf_matrix

# Załadowanie zbioru danych
df = pd.read_excel('../Dry_Bean_Dataset.xlsx')
species = df['Class'].unique()

n = 200  # Ilosc powtorzen
epochs = 150 # Maksymalna ilość epok
optimizers = ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Lion']  # Lista algorytmów optimizacji

# Pobranie danych z podzielonych plików w poprzednim etapie
input_dir = '../etap-2/stale-zbiory'

# Trenowanie i ewaluacja modelu dla poszczególnych algorytmów
for optimizer_name in optimizers:
    summary_data = []
    with pd.ExcelWriter(f'etap-3-{optimizer_name}.xlsx', engine='openpyxl') as writer:
        for i in range(n):
            print()
            print('-' * 50)
            print(f'Iteration {i + 1}/{n} using {optimizer_name}')
            print('Dat file: ', os.path.join(input_dir, f'data_split_{i + 1}.xlsx'))

            # Załadowanie danych z odpowiedniego pliku .xlsx
            with pd.ExcelFile(os.path.join(input_dir, f'data_split_{i + 1}.xlsx')) as split_file:
                X_train = pd.read_excel(split_file, sheet_name='X_train').values
                X_val = pd.read_excel(split_file, sheet_name='X_val').values
                X_test = pd.read_excel(split_file, sheet_name='X_test').values
                y_train = pd.read_excel(split_file, sheet_name='y_train').values
                y_val = pd.read_excel(split_file, sheet_name='y_val').values
                y_test = pd.read_excel(split_file, sheet_name='y_test').values

            # Utworzenie modelu
            history, accuracy, class_report, training_time, testing_time, conf_matrix = run_model(X_train, y_train, X_val, y_val, X_test, y_test, optimizer_name, epochs=epochs, iteration=i+1)

            # Przygotowanie danych do podsumowania dla każdej iteracji
            iteration_data = {
                'Iteration': i + 1,
                'Overall Accuracy': accuracy,
                'Tranining Time [s]': training_time,
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
                'Epoch': list(range(1, len(history.history['loss']) +1)),
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
