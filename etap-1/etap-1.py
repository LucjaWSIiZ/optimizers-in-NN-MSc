import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.losses import CategoricalCrossentropy
from openpyxl import Workbook


# Funkcja do budowania i trenowania modelu, zwraca history, accuracy, class_report
def run_model(X_train, y_train, X_test, y_test, epochs=24):
    model = Sequential()
    model.add(Dense(8, input_dim=X_train.shape[1]))
    model.add(Dense(17, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(7, activation='sigmoid'))

    model.compile(optimizer=RMSprop(), loss=CategoricalCrossentropy(), metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=32)

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    class_report = classification_report(y_test_classes, y_pred_classes, target_names=species, output_dict=True)

    return history, accuracy, class_report

# Załadowanie zbioru danych
df = pd.read_excel('../Dry_Bean_Dataset.xlsx')

# Zakodowanie klas na numeryczne etykiety
species = df['Class'].unique()
label_mapping = {species[i]: i for i in range(len(species))}
df['Class'] = df['Class'].map(label_mapping)

# Kodowanie one-hot
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(df[['Class']])

# Wybór cech
selected_features = ['MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Extent', 'Solidity', 'roundness', 'ShapeFactor2', 'ShapeFactor4']
X = df[selected_features]

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n = 200  # Ilosc powtorzen
summary_data = []

with pd.ExcelWriter('etap-1.xlsx', engine='openpyxl') as writer:
    for i in range(n):
        print()
        print('-' * 20)
        print(f'Iteration {i + 1}/{n}')
        # Podział danych na zbiory uczący i testowy
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        history, accuracy, class_report = run_model(X_train, y_train, X_test, y_test, epochs=24)

        # Przygotowanie danych do podsumowania dla każdej iteracji
        iteration_data = {
            'Iteration': i + 1,
            'Overall Accuracy': accuracy,
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

    # Zapis wyników podsumowania do arkusza
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("All training processes are completed and results are saved to 'etap-1.xlsx'")
