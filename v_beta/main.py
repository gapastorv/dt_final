import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, auc,
                             matthews_corrcoef, RocCurveDisplay, PrecisionRecallDisplay, log_loss, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score, roc_curve, precision_recall_curve)
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import Callback
import seaborn as sns
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
def dataframe_no_null(df):
    # Suponiendo que tienes un DataFrame llamado 'data'
    for columna in df.columns:
        datos_no_nulos = df[columna].count()
        print(f"Columna '{columna}': {datos_no_nulos} datos no nulos")


def transform_variable_types(dataset, type_of_variable, *args):
    for arg in args:
        if type_of_variable == 'dt':
            dataset[arg] = pd.to_datetime(dataset[arg])
        elif type_of_variable == 'datetime-special':
            # Eliminar la zona horaria de la cadena original
            dataset[arg] = dataset[arg].str[:-9]

            # Convertir la cadena a datetime utilizando un formato sin la zona horaria
            dataset[arg] = pd.to_datetime(dataset[arg], errors='coerce')

        elif type_of_variable == 'binary':
            # Suponiendo que tienes un DataFrame llamado 'df' y la columna que quieres convertir se llama 'is_yes'
            dataset[arg] = dataset[arg].replace({'Yes': 1, 'No': 0})

        else:
            dataset[arg] = dataset[arg].astype(type_of_variable)


def imputation(dataset, class_column, type, *args):
    if type == 'datetime-related':
        # Ordenar el DataFrame por la columna de fechas
        dataset.sort_values(by=args[0], inplace=True)

        # Imputar los valores faltantes en la columna de fecha con la fecha anterior válida
        dataset[args[0]] = dataset[args[0]].fillna(method='ffill')

        # Garantizar que los valores imputados sean menores que los valores de la otra columna
        mask = dataset[args[0]] > dataset[args[1]]
        dataset.loc[mask, args[0]] = dataset.loc[mask, args[1]]
    if type == 'str-related':
        # Convertir columnas a tipos de datos especificados
        dataset[args[0]] = dataset.apply(
            lambda row: 'Unknown details' if pd.isna(row[args[0]]) or row[args[1]] in ['To be determined'] else row[
                args[0]],
            axis=1)

    else:
        for arg in args:
            metric_by_class = None
            if type == 'category' or 'str':
                mode_by_class = dataset.groupby(class_column)[(arg)].apply(
                    lambda x: x.mode().iloc[0])
                metric_by_class = mode_by_class

                for class_value, mode in metric_by_class.items():
                    dataset.loc[
                        dataset[class_column] == class_value, arg] = dataset.loc[df_sin_columnas_nulas[class_column] ==
                                                                                 class_value, arg].fillna(mode)
            elif type == 'float':
                # Paso 1: Dividir el DataFrame en dos subconjuntos basados en la clase de interés (clase positiva y clase negativa)
                clase_positiva = dataset[dataset[class_column] == 1]
                clase_negativa = dataset[dataset[class_column] == 0]

                # Paso 2: Calcular la media de cada subconjunto para la columna de interés
                media_clase_positiva = clase_positiva[arg].mean()
                media_clase_negativa = clase_negativa[arg].mean()

                # Paso 3: Imputar los valores faltantes en 'columna' con las medias calculadas para cada clase
                dataset.loc[dataset[class_column] == 1, arg] = dataset.loc[dataset[class_column] == 1, arg].fillna(
                    media_clase_positiva)
                dataset.loc[dataset[class_column] == 0, arg] = dataset.loc[dataset[class_column] == 0, arg].fillna(
                    media_clase_negativa)

            elif type == 'datetime':
                # Ordenar el DataFrame por la columna de fechas
                dataset.sort_values(by=arg, inplace=True)

                # Convertir la columna a tipo datetime si no lo está ya
                dataset[arg] = pd.to_datetime(dataset[arg])

                # Imputar los valores faltantes en la columna de fecha con la fecha anterior válida
                dataset[arg].fillna(method='ffill', inplace=True)


def encontrar_columnas_categoria(dataset, type):
    columnas_categoria = []
    for columna in dataset.columns:
        if dataset[columna].dtype.name == type:
            columnas_categoria.append(columna)
    return columnas_categoria


# Especifica el número de columnas a leer
num_columns = 100

label = 'Affects Company Property'

# Lee el archivo CSV y especifica las primeras 100 columnas
df = pd.read_csv('pipeline-incidents-comprehensive-data.csv', encoding='latin1', sep=',')
print(f"Dimensiones del dataset inicial: {df.shape}")
print("Información adicional - datos disponibles:")
print(df.info())
dataframe_no_null(df)
types = df.dtypes
# print(df.head())

# Suponiendo que tienes un DataFrame llamado 'df'
# Calcula el porcentaje de valores nulos en cada columna
porcentaje_nulos_por_columna = (df.isnull().sum() / len(df)) * 100

# Selecciona las columnas que tienen al menos un 40% de valores nulos
columnas_a_eliminar = porcentaje_nulos_por_columna[porcentaje_nulos_por_columna >= 60].index

# Elimina las columnas seleccionadas del DataFrame
df_sin_columnas_nulas = df.drop(columnas_a_eliminar, axis=1)
dataframe_no_null(df_sin_columnas_nulas)

# Convertir columnas a tipos de datos especificados
df_sin_columnas_nulas['Provided Volume Released'] = (df_sin_columnas_nulas['Approximate Volume Released (m3)']
.apply(
    lambda x: 0 if x in ['Not Provided', 'Not Applicable'] else 1))
df_sin_columnas_nulas['Approximate Volume Released (m3)'] = (df_sin_columnas_nulas['Approximate Volume Released (m3)']
.apply(
    lambda x: 0 if x in ['Not Provided', 'Not Applicable'] else float(x)))

transform_variable_types(df_sin_columnas_nulas, 'category', 'Incident Types', 'Status', 'Substance',
                         'Release Type', 'What happened category', 'Why it happened category',
                         'Duration of interruption of pipeline operations', 'Pipeline or Facility Type',
                         'Activity being performed at time of incident', 'How the incident was discovered',
                         'Incident type', 'Released substance type', 'Regulation', 'Substance carried', 'Land Use',
                         'Population Density', 'Emergency Level', 'Investigation Type')

transform_variable_types(df_sin_columnas_nulas, 'binary', 'Significant', 'Pipeline or facility equipment involved',
                         'Rupture', 'Pipe body release', 'Residual effects on the environment',
                         label, 'Off Company Property', 'Affects Pipeline right-of-way',
                         'Affects off Pipeline right-of-way', 'Was NEB Staff Deployed', 'Insulation installed',
                         'Equipment or component has never been inspected',
                         'Most recent inspection part of the routine inspection program',
                         'No maintenance done on this equipment or component',
                         'Most recent maintenance work part of the routine maintenance program')

transform_variable_types(df_sin_columnas_nulas, str, 'Incident Number', 'Nearest Populated Centre', 'Province',
                         'Company', 'Detailed what happened', 'Detailed why it happened', 'Pipeline Name', 'Country',
                         'Kilometre post', 'Equipment or component involved')

transform_variable_types(df_sin_columnas_nulas, float, 'Approximate Volume Released (m3)')

transform_variable_types(df_sin_columnas_nulas, 'dt', 'Reported Date', 'Closed Date')

transform_variable_types(df_sin_columnas_nulas, 'datetime-special', 'Discovered Date and Time',
                         'Occurrence Date and Time')

imputation(df_sin_columnas_nulas, label, 'category',
           'Duration of interruption of pipeline operations', 'Emergency Level', 'Investigation Type',
           'How the incident was discovered', 'Regulation', 'Pipeline or Facility Type', 'Substance carried',
           'Released substance type', 'Pipe body release')

imputation(df_sin_columnas_nulas, label, 'float', 'Pipeline length (km)',
           'Released volume (m3)')

imputation(df_sin_columnas_nulas, label, 'datetime', 'Closed Date',
           'Discovered Date and Time')

imputation(df_sin_columnas_nulas, label, 'datetime-related',
           'Occurrence Date and Time', 'Discovered Date and Time')

imputation(df_sin_columnas_nulas, label, 'str-related',
           'Detailed what happened', 'What happened category')

imputation(df_sin_columnas_nulas, label, 'str-related',
           'Detailed why it happened', 'Why it happened category')

df['Equipment or component involved'] = df['Equipment or component involved'].fillna('Unknown')

# Primero, asegúrate de que las columnas de fecha estén en el formato datetime
df_sin_columnas_nulas['Occurrence Date and Time'] = pd.to_datetime(df_sin_columnas_nulas['Occurrence Date and Time'])
df_sin_columnas_nulas['Discovered Date and Time'] = pd.to_datetime(df_sin_columnas_nulas['Discovered Date and Time'])

# Luego, calcula la diferencia en horas y crea una nueva columna con los resultados
df_sin_columnas_nulas['Time to discover incident (hours)'] = (df_sin_columnas_nulas['Discovered Date and Time'] -
                                                              df_sin_columnas_nulas[
                                                                  'Occurrence Date and Time']).dt.total_seconds() / 3600

# Reemplazar los valores NaN con '0.00000000' en la columna 'Pipeline outside diameter (NPS)'
df_sin_columnas_nulas['Pipeline outside diameter (NPS)'] = df_sin_columnas_nulas[
    'Pipeline outside diameter (NPS)'].fillna('0.00000')
df_sin_columnas_nulas['Pipeline Name'] = df_sin_columnas_nulas['Pipeline Name'].replace('nan', '0').fillna(0)

df_sin_columnas_nulas['Kilometre post'] = df_sin_columnas_nulas['Kilometre post'].replace('nan', np.nan)
moda_por_centro_poblado = df_sin_columnas_nulas.groupby('Nearest Populated Centre')['Kilometre post'].apply(
    lambda x: x.mode().dropna().iloc[0] if not x.mode().dropna().empty else str(0))
for centro_poblado, moda in moda_por_centro_poblado.items():
    df_sin_columnas_nulas.loc[(df_sin_columnas_nulas['Nearest Populated Centre'] == centro_poblado) & (
        df_sin_columnas_nulas['Kilometre post'].isnull()), 'Kilometre post'] = moda

# transform_variable_types(df_sin_columnas_nulas, 'category', 'Province', 'Company', 'Pipeline outside diameter (NPS)',
#                          'Equipment or component involved', 'Pipeline Name', 'Detailed why it happened',
#                          'Detailed what happened', 'Kilometre post', 'Nearest Populated Centre')

new_df = df_sin_columnas_nulas.drop(['Incident Number'], axis=1)
new_df = new_df.drop(['Country'], axis=1)

atributos_analisis = pd.DataFrame()
# Calcula el porcentaje de valores nulos en cada columna
porcentaje_nulos_por_columna = (new_df.isnull().sum() / len(df)) * 100
total_nulos = new_df.isnull().sum()
variable_types = new_df.dtypes
atributos_analisis['Null'] = total_nulos
atributos_analisis['Percentage'] = porcentaje_nulos_por_columna
atributos_analisis['Type'] = variable_types
atributos_analisis['Unique Values'] = new_df.nunique()

# Crear una lista para almacenar el nombre de las columnas que cumplen con el criterio
columnas_con_proporcion_deseada = []

min_proportion = 0.45
max_proportion = 0.55

# Iterar sobre todas las columnas del DataFrame new_df
for columna in new_df.columns:
    proportion_values = new_df[columna].value_counts(normalize=True)
    if any((proportion_values >= min_proportion) & (proportion_values <= max_proportion)):
        columnas_con_proporcion_deseada.append(columna)
        print(f"Columna: {columna}")
        print(proportion_values)
        print("\n")

# Imprimir las columnas que cumplen con el criterio
print(f"Columnas con al menos un valor entre {min_proportion} y {max_proportion} de proporción:")
print(columnas_con_proporcion_deseada)

# Obtener las listas de columnas de diferentes tipos
columnas_numericas = new_df.select_dtypes(include=['int64', 'float64']).columns
columnas_categoricas = encontrar_columnas_categoria(new_df, 'category')
columnas_datetime = encontrar_columnas_categoria(new_df, 'datetime64[ns]')
columnas_object = encontrar_columnas_categoria(new_df, 'object')

# Suponiendo que 'columna_1' y 'columna_2' son columnas categóricas en tu DataFrame new_df
new_df_encoded = pd.get_dummies(new_df, columns=columnas_categoricas)
label_encoder = LabelEncoder()

for column in columnas_object:
    new_df_encoded[column] = label_encoder.fit_transform(new_df_encoded[column])

for column in columnas_datetime:
    new_df_encoded[column + '_month'] = new_df_encoded[column].dt.month
    new_df_encoded[column + '_date'] = new_df_encoded[column].dt.day

reduced_df = new_df_encoded.drop(columnas_datetime, axis=1)

# Suponiendo que 'new_df' contiene tus datos
scaler = MinMaxScaler()
new_df_scaled = reduced_df.copy()
new_df_scaled = pd.DataFrame(scaler.fit_transform(reduced_df), columns=reduced_df.columns)

X = reduced_df.drop(label, axis=1)  # DataFrame con las características, excluyendo la columna de la clase
y = reduced_df[label]  # Serie que contiene las etiquetas, es decir, la columna de la clase

# Stratified 10-fold cross validation
stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

print(f"XGBoost: ")
# Inicializar las listas para almacenar las métricas
auc_scores = []
accuracy_scores = []
recall_scores = []
f1_scores = []
prec_scores = []
confusion_matrices = []
# Inicializar las listas para almacenar las métricas adicionales
mcc_scores = []
loss_scores = []
overall_confusion_matrix = np.zeros((2, 2))

# Inicializar las listas para almacenar las métricas de log_loss
train_log_loss_scores = []
val_log_loss_scores = []

model = XGBClassifier()

for train_index, test_index in stratified_k_fold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Entrenar el modelo
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric=["logloss"], eval_set=eval_set, verbose=False)

    # Predecir las etiquetas
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcular métricas
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    overall_confusion_matrix += cm

    # Imprimir las métricas
    print("AUC: %.3f" % auc)
    print("Accuracy: %.3f" % accuracy)
    print("Recall: %.3f" % recall)
    print("Precision: %.3f" % precision)
    print("F1-score: %.3f" % f1)
    print("Matthews Correlation Coefficient (MCC): %.3f" % mcc)
    print("Log Loss: %.3f" % loss)

    # Imprimir la matriz de confusión
    print("Confusion Matrix:")
    print(cm)

    # Almacenar las métricas en las listas
    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    recall_scores.append(recall)
    f1_scores.append(f1)
    prec_scores.append(precision)
    mcc_scores.append(mcc)
    loss_scores.append(loss)
    confusion_matrices.append(cm)

    # Obtener las métricas de log_loss
    train_log_loss = model.evals_result()['validation_0']['logloss']
    val_log_loss = model.evals_result()['validation_1']['logloss']
    train_log_loss_scores.append(train_log_loss)
    val_log_loss_scores.append(val_log_loss)

# Calcular el promedio y la desviación estándar de las métricas
avg_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
avg_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
avg_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
avg_prec = np.mean(prec_scores)
std_prec = np.std(prec_scores)
avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
avg_mcc = np.mean(mcc_scores)
std_mcc = np.std(mcc_scores)
avg_loss = np.mean(loss_scores)
std_loss = np.std(loss_scores)

# Imprimir las métricas promedio
print("Average AUC: %.3f (+/- %.3f)" % (avg_auc, std_auc))
print("Average Accuracy: %.3f (+/- %.3f)" % (avg_accuracy, std_accuracy))
print("Average Recall: %.3f (+/- %.3f)" % (avg_recall, std_recall))
print("Average Precision: %.3f (+/- %.3f)" % (avg_prec, std_prec))
print("Average F1-score: %.3f (+/- %.3f)" % (avg_f1, std_f1))
print("Average Matthews Correlation Coefficient (MCC): %.3f (+/- %.3f)" % (avg_mcc, std_mcc))
print("Average Log Loss: %.3f (+/- %.3f)" % (avg_loss, std_loss))

# Imprimir la matriz de confusión general
print("Overall Confusion Matrix:")
print(overall_confusion_matrix)

# Plot AUC
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title(f'ROC Curve - XGBoost')
plt.show()

# Plot Precision vs Recall
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
plt.title(f'Precision-Recall Curve - XGBoost')
plt.show()

results = model.evals_result()

plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss for XGBoost')
plt.legend()
plt.show()

# Suponiendo que 'new_df' contiene tus datos
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

EPOCHS = 50

if isinstance(y, pd.DataFrame):
    y_encoded = LabelEncoder().fit_transform(y.iloc[:, 0])
else:
    y_encoded = LabelEncoder().fit_transform(y)

# Stratified 10-fold cross validation
stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Deep Autoencoder model using Keras
input_dim = X_scaled.shape[1]  # Numero de caracteristicas

# Construyendo el modelo
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

# Compilando el modelo
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Listas para almacenar los resultados de evaluación
accuracy_scores = []
best_test_loss = []
auc_scores = []
recall_scores = []
prec_scores = []
f1_scores = []
loss_scores = []
mcc_scores = []
mse_scores = []
mae_scores = []
r2_scores = []
mape_scores = []
rmse_scores = []
nrmse_scores = []
wape_scores = []
wmape_scores = []
params_scores = []
CURRENT_FOLD = 0

best_model = None
best_val_loss = float('inf')

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        loss, _ = self.model.evaluate(x_test, y_test, verbose=0)
        self.test_loss.append(loss)

def create_classifier(encoding_dim):
    model = Sequential()
    model.add(Dense(int(encoding_dim * 9 / 10), activation='relu', input_dim=encoding_dim))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

for train_index, test_index in stratified_k_fold.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    autoencoder.fit(X_train, X_train, epochs=EPOCHS, batch_size=256, verbose=0)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-2].output)
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Visualización de Pérdida Media vs. Épocas para cada fold
    classifier = create_classifier(128)
    test_callback = TestCallback((X_test_encoded, y_test))
    history = classifier.fit(X_train_encoded, y_train, epochs=EPOCHS, batch_size=256, verbose=0,
                             callbacks=[test_callback])

    y_pred = classifier.predict(X_test_encoded)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    current_auc = roc_auc_score(y_test, y_pred)  # Calcular el Área Bajo la Curva ROC
    current_recall = recall_score(y_test, y_pred)  # Calcular la tasa de recuperación (Recall)
    current_precision = precision_score(y_test, y_pred)  # Calcular la precisión
    current_f1 = f1_score(y_test, y_pred)  # Calcular el F1-Score
    current_acc = accuracy_score(y_test, y_pred)  # Calcular la precisión
    current_mcc = matthews_corrcoef(y_test, y_pred)  # Calcular el coeficiente de correlación de Matthews
    current_loss_log = log_loss(y_test, y_pred)  # Calcular la pérdida logarítmica

    current_mse = mean_squared_error(y_test, y_pred)
    current_mae = mean_absolute_error(y_test, y_pred)
    current_r2 = r2_score(y_test, y_pred)

    abs_error = np.abs(y_test - y_pred)
    actual = np.abs(y_test)

    mape = np.mean(np.abs(abs_error / actual)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nrmse = rmse / (y_test.max() - y_test.min())
    wape = np.sum(abs_error) / np.sum(actual) * 100
    wmape = np.sum(abs_error) / np.sum(actual)

    cm = confusion_matrix(y_test, y_pred)

    print(f"Metrics for Fold {CURRENT_FOLD}:")
    print(f"AUC: {current_auc}")
    print(f"Accuracy: {current_acc}")
    print(f"Recall: {current_recall}")
    print(f"Precision: {current_precision}")
    print(f"F1-score: {current_f1}")
    print(f"Matthews Correlation Coefficient (MCC): {current_mcc}")
    print(f"Log Loss: {current_loss_log}")
    print(f"MSE: {current_mse}")
    print(f"MAE: {current_mae}")
    print(f"R2 Score: {current_r2}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")
    print(f"NRMSE: {nrmse}")
    print(f"WAPE: {wape}")
    print(f"WMAPE: {wmape}")

    # Imprimir la matriz de confusión
    print("Confusion Matrix:")
    print(cm)
    print("\n")

    auc_scores.append(current_auc)
    recall_scores.append(current_recall)
    prec_scores.append(current_precision)
    f1_scores.append(current_f1)
    accuracy_scores.append(current_acc)
    mcc_scores.append(current_mcc)
    loss_scores.append(current_loss_log)

    mse_scores.append(current_mse)
    mae_scores.append(current_mae)
    r2_scores.append(current_r2)

    mape_scores.append(mape)
    rmse_scores.append(rmse)
    nrmse_scores.append(nrmse)
    wape_scores.append(wape)
    wmape_scores.append(wmape)

    # Obtener la pérdida en el conjunto de validación del último epoch
    val_loss = history.history['loss'][-1]

    # Guardar el modelo si la pérdida en el conjunto de validación es la menor hasta ahora
    if val_loss < best_val_loss:
        best_auc = current_auc
        best_model = classifier
        best_val_loss = val_loss

    CURRENT_FOLD += 1

# Calcular el promedio y la desviación estándar de las métricas
avg_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)
avg_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
avg_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
avg_prec = np.mean(prec_scores)
std_prec = np.std(prec_scores)
avg_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
avg_mcc = np.mean(mcc_scores)
std_mcc = np.std(mcc_scores)
avg_loss = np.mean(loss_scores)
std_loss = np.std(loss_scores)
avg_mse = np.mean(mse_scores)
std_mse = np.std(mse_scores)
avg_mae = np.mean(mae_scores)
std_mae = np.std(mae_scores)
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
avg_mape = np.mean(mape_scores)
std_mape = np.std(mape_scores)
avg_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
avg_nrmse = np.mean(nrmse_scores)
std_nrmse = np.std(nrmse_scores)
avg_wape = np.mean(wape_scores)
std_wape = np.std(wape_scores)
avg_wmape = np.mean(wmape_scores)
std_wmape = np.std(wmape_scores)

print("Average AUC: %.3f (+/- %.3f)" % (avg_auc, std_auc))
print("Average Accuracy: %.3f (+/- %.3f)" % (avg_accuracy, std_accuracy))
print("Average Recall: %.3f (+/- %.3f)" % (avg_recall, std_recall))
print("Average Precision: %.3f (+/- %.3f)" % (avg_prec, std_prec))
print("Average F1-score: %.3f (+/- %.3f)" % (avg_f1, std_f1))
print("Average Matthews Correlation Coefficient (MCC): %.3f (+/- %.3f)" % (avg_mcc, std_mcc))
print("Average Log Loss: %.3f (+/- %.3f)" % (avg_loss, std_loss))
print("Average MSE: %.3f (+/- %.3f)" % (avg_mse, std_mse))
print("Average MAE: %.3f (+/- %.3f)" % (avg_mae, std_mae))
print("Average R2 Score: %.3f (+/- %.3f)" % (avg_r2, std_r2))
print("Average MAPE: %.3f (+/- %.3f)" % (avg_mape, std_mape))
print("Average RMSE: %.3f (+/- %.3f)" % (avg_rmse, std_rmse))
print("Average NRMSE: %.3f (+/- %.3f)" % (avg_nrmse, std_nrmse))
print("Average WAPE: %.3f (+/- %.3f)" % (avg_wape, std_wape))
print("Average WMAPE: %.3f (+/- %.3f)" % (avg_wmape, std_wmape))

# Visualización de la pérdida media vs. épocas
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(test_callback.test_loss, label='Test Loss')
plt.title('Loss vs. Epochs - Autoencoder')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


print()
