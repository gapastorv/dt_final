import pandas as pd
from keras import metrics
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score, auc,
                             matthews_corrcoef, RocCurveDisplay, PrecisionRecallDisplay, log_loss, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import Callback
import seaborn as sns
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import warnings
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# Ejemplo de implementación de un clasificador utilizando Random Forest
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def ver_no_nulos(dataset):
    # EN esta funcion se hace una verificación de las columnas y los componentes no nulos
    for columna in dataset.columns:
        #conteo de valores que no sean nulos
        datos_no_nulos = dataset[columna].count()
        print(f"Columna '{columna}': {datos_no_nulos} datos no nulos")


def cambio_tipo_variable(dataset, tipo_dato, *args):
    #En este proyecto se aplico esta funcion para transformar las variables a un tipo de dato especfico
    #Esto se aplico con el fin de poder luego impleemntar modificaciones sobre los valores para normalizar y vectorizar
    for arg in args:
        #conversión a fecha
        if tipo_dato == 'dt':
            dataset[arg] = pd.to_datetime(dataset[arg])
        #conversion de fecha que incluya horas
        elif tipo_dato == 'datetime-special':
            # Eliminar la zona horaria de la cadena original
            dataset[arg] = dataset[arg].str[:-9]

            # Convertir la cadena a datetime utilizando un formato sin la zona horaria
            dataset[arg] = pd.to_datetime(dataset[arg], errors='coerce')

        elif tipo_dato == 'binary':
            #esta se aplica para casos donde los valores sean de ambito binario (si, no)
            dataset[arg] = dataset[arg].replace({'Yes': 1, 'No': 0})

        else:
            #para transformar a float o int dependiendo el argumento de el tipo de variable
            dataset[arg] = dataset[arg].astype(tipo_dato)


def imputacion_variables(dataset, columna_clase, tipo_dato, *args):
    if tipo_dato == 'datetime-related':
        #Esto se aplica con el fin de rellenar los datos por medio de valores previos, y valores de otras variables
        # Ordenar el DataFrame por la columna de fechas
        dataset.sort_values(by=args[0], inplace=True)

        # Imputacion con la fecha anterior
        dataset[args[0]] = dataset[args[0]].fillna(method='ffill')

        # Verificar que valores sean menores a los de otra fila, se puede aplicar para features relacionadas
        # EJ: Fecha de incidente llenada con base en la fecha de detección dle incidente
        mask = dataset[args[0]] > dataset[args[1]]
        dataset.loc[mask, args[0]] = dataset.loc[mask, args[1]]
    if tipo_dato == 'str-related':
        # Convertir columnas a tipos de datos especificados
        # En este caso solo define una constante en casos de que los valores de una variable adyancente fuera nula
        # EJ: Kilometre post se aplica para ver cual fue la localidad cercana mas afectada
        # Esta variable depende de otra columna de la localidad o condado en el cual ocurrio el incidente
        dataset[args[0]] = dataset.apply(
            lambda row: 'Unknown details' if pd.isna(row[args[0]]) or row[args[1]] in ['To be determined'] else row[
                args[0]],
            axis=1)

    else:
        for arg in args:
            metric_by_class = None
            #Para relleno de datos con la moda de la clase en cuestión.
            if tipo_dato == 'category' or 'str':
                mode_by_class = dataset.groupby(columna_clase)[(arg)].apply(
                    lambda x: x.mode().iloc[0])
                metric_by_class = mode_by_class

                for class_value, mode in metric_by_class.items():
                    dataset.loc[
                        dataset[columna_clase] == class_value, arg] = dataset.loc[dataset_columnas_reducidas[columna_clase] ==
                                                                             class_value, arg].fillna(mode)
            #Para relleno de datos con la media de la clase en cuestion.
            elif tipo_dato == 'float':
                #Division del DataFrame en dos subconjuntos basados en la clase principal (clase positiva y clase negativa)
                clase_positiva = dataset[dataset[columna_clase] == 1]
                clase_negativa = dataset[dataset[columna_clase] == 0]

                #Calculo de la media de cada subconjunto para la clase
                media_clase_positiva = clase_positiva[arg].mean()
                media_clase_negativa = clase_negativa[arg].mean()

                #Imputacion de valores faltantes con las medias calculadas
                dataset.loc[dataset[columna_clase] == 1, arg] = dataset.loc[dataset[columna_clase] == 1, arg].fillna(
                    media_clase_positiva)
                dataset.loc[dataset[columna_clase] == 0, arg] = dataset.loc[dataset[columna_clase] == 0, arg].fillna(
                    media_clase_negativa)

            elif tipo_dato == 'datetime':
                #Ordenamiento de la columna por medio de fechas
                dataset.sort_values(by=arg, inplace=True)

                #En caso de que no haya sido convertida antes a datetime, se la convierte aqui
                dataset[arg] = pd.to_datetime(dataset[arg])

                #Imputacion de datos
                dataset[arg].fillna(method='ffill', inplace=True)


def encontrar_columnas_categoria(dataset, tipo_dato):
    #funcion de extraccion de variables de un tipo de dato dado
    columnas_categoria = []
    for columna in dataset.columns:
        if dataset[columna].dtype.name == tipo_dato:
            columnas_categoria.append(columna)
    return columnas_categoria


# Especifica el número de columnas a leer, esto ya no fue necesario para esta versión
num_columns = 100

#nombre de la clase principal del dataset
label = 'Affects Company Property'

#lectura del dataset y almacenamiento en una variable
dataset = pd.read_csv('pipeline-incidents-comprehensive-data.csv', encoding='latin1', sep=',')
print(f"Dimensiones del dataset inicial: {dataset.shape}")
print("Información adicional - datos disponibles:")
print(dataset.info())
#impresion de datos no nulos por variable
ver_no_nulos(dataset)
tipo_datos = dataset.dtypes
# print(dataset.head())

#uso de heatmap para ver valore snulas de las variables del dataset
plt.figure(figsize=(15, 10))
sns.heatmap(dataset.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

#Descarte de columnas de datos por porcentaje, en este caso, las que tengan más del 60% de datos nulos
porcentaje_nulos_por_columna = (dataset.isnull().sum() / len(dataset)) * 100

#seleccion de variables/columnas a eliminar
columnas_a_eliminar = porcentaje_nulos_por_columna[porcentaje_nulos_por_columna >= 60].index

# Eliminacion de las columnas seleccionadas del DataFrame
dataset_columnas_reducidas = dataset.drop(columnas_a_eliminar, axis=1)
ver_no_nulos(dataset_columnas_reducidas)

#vista previa con un heatmap de los valores nulos en el dataset tras el descarte inicial de columnas
plt.figure(figsize=(20, 20))
sns.heatmap(dataset_columnas_reducidas.isnull(), yticklabels=False,
            cbar=False, cmap='viridis')
plt.show()

#presentación con displot de porcentaje de datos nulos por columna
plt.figure(figsize=(10,6))
sns.displot(
    data=dataset_columnas_reducidas.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=2.25
)
# Modificar etiquetas del eje y para su visualizacion en el grafico
plt.yticks(ticks=range(len(dataset_columnas_reducidas.columns)),
           labels=[col[:10] + '...' if len(col) > 10 else col for col in dataset_columnas_reducidas.columns])

plt.subplots_adjust(hspace=0.5)

plt.show()

#relleno de datos inicial, se aplican valores de 0 y 1 para determinar si el registro contiene datos del liquido vertido
dataset_columnas_reducidas['Provided Volume Released'] = (dataset_columnas_reducidas['Approximate Volume Released (m3)']
.apply(
    lambda x: 0 if x in ['Not Provided', 'Not Applicable'] else 1))

#relleno de columna de cantidad de liquido vertido con valores de 0, en caso de que fueran vacios
dataset_columnas_reducidas['Approximate Volume Released (m3)'] = (dataset_columnas_reducidas['Approximate Volume Released (m3)']
.apply(
    lambda x: 0 if x in ['Not Provided', 'Not Applicable'] else float(x)))

# Convertir columnas a tipos de datos especificados
#conversion a categoricos
cambio_tipo_variable(dataset_columnas_reducidas, 'category', 'Incident Types', 'Status', 'Substance',
                         'Release Type', 'What happened category', 'Why it happened category',
                         'Duration of interruption of pipeline operations', 'Pipeline or Facility Type',
                         'Activity being performed at time of incident', 'How the incident was discovered',
                         'Incident type', 'Released substance type', 'Regulation', 'Substance carried', 'Land Use',
                         'Population Density', 'Emergency Level', 'Investigation Type')

#conversion a binarios
cambio_tipo_variable(dataset_columnas_reducidas, 'binary', 'Significant', 'Pipeline or facility equipment involved',
                         'Rupture', 'Pipe body release', 'Residual effects on the environment',
                         label, 'Off Company Property', 'Affects Pipeline right-of-way',
                         'Affects off Pipeline right-of-way', 'Was NEB Staff Deployed', 'Insulation installed',
                         'Equipment or component has never been inspected',
                         'Most recent inspection part of the routine inspection program',
                         'No maintenance done on this equipment or component',
                         'Most recent maintenance work part of the routine maintenance program')

#conversion a string (este de igual forma se vio representado como object
cambio_tipo_variable(dataset_columnas_reducidas, str, 'Incident Number', 'Nearest Populated Centre', 'Province',
                         'Company', 'Detailed what happened', 'Detailed why it happened', 'Pipeline Name', 'Country',
                         'Kilometre post', 'Equipment or component involved')

#conversion a float
cambio_tipo_variable(dataset_columnas_reducidas, float, 'Approximate Volume Released (m3)')

#conversion a datetime
cambio_tipo_variable(dataset_columnas_reducidas, 'dt', 'Reported Date', 'Closed Date')

#conversion a datetime, dado el caso qeu incluya hora
cambio_tipo_variable(dataset_columnas_reducidas, 'datetime-special', 'Discovered Date and Time',
                         'Occurrence Date and Time')

#imputacion de datos categoricos, se toma en cuenta la moda
imputacion_variables(dataset_columnas_reducidas, label, 'category',
           'Duration of interruption of pipeline operations', 'Emergency Level', 'Investigation Type',
           'How the incident was discovered', 'Regulation', 'Pipeline or Facility Type', 'Substance carried',
           'Released substance type', 'Pipe body release')

#imputacion de datos de tipo flotante, se toma en cuenta la media
imputacion_variables(dataset_columnas_reducidas, label, 'float', 'Pipeline length (km)',
           'Released volume (m3)')

#imputacion de datos de fecha, se aplica aca el relleno de datos con base en valores previos en la columna
imputacion_variables(dataset_columnas_reducidas, label, 'datetime', 'Closed Date',
           'Discovered Date and Time')

#imputacion de datos de fecha, con segunda columna para relleno de datos y que haya concordancia
imputacion_variables(dataset_columnas_reducidas, label, 'datetime-related',
           'Occurrence Date and Time', 'Discovered Date and Time')

#imputacion con base en variables relacionadas, en este caso por detalles que se correlacionan con las categorias
imputacion_variables(dataset_columnas_reducidas, label, 'str-related',
           'Detailed what happened', 'What happened category')

imputacion_variables(dataset_columnas_reducidas, label, 'str-related',
           'Detailed why it happened', 'Why it happened category')

#relleno con constante, esta implementacion se hizo debido a que era de las columnas que menos datos vacios contenia
dataset['Equipment or component involved'] = dataset['Equipment or component involved'].fillna('Unknown')

# SE aplica una segunda conversion a datetime en estas columnas, esto con el fin de evitar fallos en el proceso
dataset_columnas_reducidas['Occurrence Date and Time'] = pd.to_datetime(dataset_columnas_reducidas['Occurrence Date and Time'])
dataset_columnas_reducidas['Discovered Date and Time'] = pd.to_datetime(dataset_columnas_reducidas['Discovered Date and Time'])

# Calcyulo de la diferencia de horas entre la fecha del incidente y la fecha de descubrimiento de este
dataset_columnas_reducidas['Time to discover incident (hours)'] = (dataset_columnas_reducidas['Discovered Date and Time'] -
                                                              dataset_columnas_reducidas[
                                                                  'Occurrence Date and Time']).dt.total_seconds() / 3600

#NOTA: la implementacion de constantes en estas columnas se debe a que el proceso previo aun contenia valores nulos
#dichos valores dado que no se detemrinaban como vacios sino como "nan" (string)
# Reemplazar los valores NaN con '0.00000000' en la columna 'Pipeline outside diameter (NPS)'
dataset_columnas_reducidas['Pipeline outside diameter (NPS)'] = dataset_columnas_reducidas[
    'Pipeline outside diameter (NPS)'].fillna('0.00000')
dataset_columnas_reducidas['Pipeline Name'] = dataset_columnas_reducidas['Pipeline Name'].replace('nan', '0').fillna(0)

#relleno de datos con base en la moda de columna de condado para Kilometre post, que contiene aun vacios
dataset_columnas_reducidas['Kilometre post'] = dataset_columnas_reducidas['Kilometre post'].replace('nan', np.nan)
moda_por_centro_poblado = dataset_columnas_reducidas.groupby('Nearest Populated Centre')['Kilometre post'].apply(
    lambda x: x.mode().dropna().iloc[0] if not x.mode().dropna().empty else str(0))
for centro_poblado, moda in moda_por_centro_poblado.items():
    dataset_columnas_reducidas.loc[(dataset_columnas_reducidas['Nearest Populated Centre'] == centro_poblado) & (
        dataset_columnas_reducidas['Kilometre post'].isnull()), 'Kilometre post'] = moda

# cambio_tipo_variable(dataset_columnas_reducidas, 'category', 'Province', 'Company', 'Pipeline outside diameter (NPS)',
#                          'Equipment or component involved', 'Pipeline Name', 'Detailed why it happened',
#                          'Detailed what happened', 'Kilometre post', 'Nearest Populated Centre')

#estas columnas fueron descartadas al no tener valores categoricos de apoyo
new_df = dataset_columnas_reducidas.drop(['Incident Number'], axis=1)
#por solo tenrr un unico valor en pais (Canada)
new_df = new_df.drop(['Country'], axis=1)

#muestra de datos tras imputacion, en cuanto a si hay datos nulos y procentaje de estos
plt.figure(figsize=(20, 20))
sns.heatmap(new_df.isnull(), yticklabels=False,
            cbar=False, cmap='viridis')
plt.show()

plt.figure(figsize=(10,6))
sns.displot(
    data=new_df.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=2.25
)
# Modificar etiquetas del eje y para mostrarla
plt.yticks(ticks=range(len(new_df.columns)),
           labels=[col[:10] + '...' if len(col) > 10 else col for col in new_df.columns])

plt.subplots_adjust(hspace=0.5)

plt.show()

atributos_analisis = pd.DataFrame()
# Calcula el porcentaje de valores nulos en cada columna
porcentaje_nulos_por_columna = (new_df.isnull().sum() / len(dataset)) * 100
total_nulos = new_df.isnull().sum()
variable_tipo_datos = new_df.dtypes
atributos_analisis['Null'] = total_nulos
atributos_analisis['Percentage'] = porcentaje_nulos_por_columna
atributos_analisis['Type'] = variable_tipo_datos
atributos_analisis['Unique Values'] = new_df.nunique()

# Crear una lista para almacenar el nombre de las columnas que cumplen con el criterio
# en este caso, columnas que tengan una buena distribucion de datos
#esto con el fin de que se pueda encontrar variables que lleguen a aportar por su diversidad de informacion
columnas_con_proporcion_deseada = []

# se establecio como criterio que un valor contenga al menos 45% a 55% de los valores de la columna
#esto para determinar variables que puedan contener informacion clave y pasarse a clasificacion binaria (0 o 1)
min_proportion = 0.45
max_proportion = 0.55

# Iteracion sobre todas las columnas del DataFrame new_df
for columna in new_df.columns:
    proportion_values = new_df[columna].value_counts(normalize=True)
    if any((proportion_values >= min_proportion) & (proportion_values <= max_proportion)):
        columnas_con_proporcion_deseada.append(columna)
        print(f"Columna: {columna}")
        print(proportion_values)
        print("\n")

# Impresion de las columnas que cumplen con el criterio
print(f"Columnas con al menos un valor entre {min_proportion} y {max_proportion} de proporción:")
print(columnas_con_proporcion_deseada)

# Obtener las listas de columnas de diferentes tipos
columnas_numericas = new_df.select_dtypes(include=['int64', 'float64']).columns
columnas_categoricas = encontrar_columnas_categoria(new_df, 'category')
columnas_datetime = encontrar_columnas_categoria(new_df, 'datetime64[ns]')
columnas_object = encontrar_columnas_categoria(new_df, 'object')

# aplicacion de one hot bit encoding, esto para pasar a vector variables categoricas
new_df_encoded = pd.get_dummies(new_df, columns=columnas_categoricas)
# se aplico labelEncoder para transformar varaibles de tipo string a numerico, one-to-one encoding
label_encoder = LabelEncoder()

for column in columnas_object:
    new_df_encoded[column] = label_encoder.fit_transform(new_df_encoded[column])

#separacion de datos en columnas de tipo datetime, considerando solamente mes y fecha
for column in columnas_datetime:
    new_df_encoded[column + '_month'] = new_df_encoded[column].dt.month
    new_df_encoded[column + '_date'] = new_df_encoded[column].dt.day

#columnas de tipo datetime se descartan aqui al tenr los datos almacenados por mes y fecha
reduced_df = new_df_encoded.drop(columnas_datetime, axis=1)

# APlicación de MinMaxScaler para normalizar dataset
scaler = MinMaxScaler()
new_df_scaled = reduced_df.copy()
new_df_scaled = pd.DataFrame(scaler.fit_transform(reduced_df), columns=reduced_df.columns)

from sklearn.model_selection import StratifiedKFold

X = new_df_scaled.drop(label, axis=1)  # Características/variables del datatset, sin la clase
y = new_df_scaled[label]  # Clase del dataset

from keras.models import Model
from keras.layers import Input, Dense

input_dim = X.shape[1]#se analiza la dimension del dataset con variables, esto para que se aplique al modelo de autoencoder

print(f"XGBoost: ")
#listas para almacenar las métricas de XGBoost como modelo en solitario (se inicia verificando el modelo solo)
auc_scores = []
accuracy_scores = []
recall_scores = []
f1_scores = []
prec_scores = []
confusion_matrices = []
mcc_scores = []
loss_scores = []
overall_confusion_matrix = np.zeros((2, 2))
best_xgb = None
best_cm = None
best_accuracy = 0
#listas de perdida (aplicado en versiones previas que no usaban GridSearch)
train_log_loss_scores = []
val_log_loss_scores = []

#numero de epocas para autoencoder, se determino este valor tambien para las iteraciones en XGBoost
epochs = 200
#parametros de entreameinto de XGBoost, se aplicaron parametros mas reducidos dado overfitting al incrmentarlos
param_grid_xgb = {
        'n_estimators': [epochs],
        'learning_rate': [0.1],
        'max_depth': [3],
        'epochs': [epochs]
    }

#defincion del modelo, folds y el stratified k-fold
model = XGBClassifier()
folds = 10
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1)

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #gridsearch aplicado para definir hieprparametros para el modelo de arbol de decision
    grid_search_xgb = GridSearchCV(model, param_grid_xgb, cv=3, scoring='accuracy')

    # Entrenamiento GridSearch para encontrar la mejor combinación de hiperparámetros
    grid_search_xgb.fit(X_train, y_train, early_stopping_rounds=100,
                        eval_metric=['logloss', 'error', 'error@0.5'],
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        verbose=False)

    # Obtencion del mejor modelo entrenado, se lo considera por mejor resultado de parametros
    best_xgb_model = grid_search_xgb.best_estimator_

    # predicciones con el mejor modelo
    y_pred_proba = best_xgb_model.predict_proba(X_test)[:, 1]
    y_pred = best_xgb_model.predict(X_test)

    # Calculo de métricas, se toma en cuenta el resultado predicho y el real
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)
    overall_confusion_matrix += cm

    # Impresion de las métricas
    print("AUC: %.3f" % auc)
    print("Accuracy: %.3f" % accuracy)
    print("Recall: %.3f" % recall)
    print("Precision: %.3f" % precision)
    print("F1-score: %.3f" % f1)
    print("Matthews Correlation Coefficient (MCC): %.3f" % mcc)
    print("Log Loss: %.3f" % loss)

    # Impresion de la matriz de confusión
    print("Confusion Matrix:")
    print(cm)

    # Almacenamiento de las métricas en las listas
    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    recall_scores.append(recall)
    f1_scores.append(f1)
    prec_scores.append(precision)
    mcc_scores.append(mcc)
    loss_scores.append(loss)
    confusion_matrices.append(cm)

    #se alamcena el mejor modelo con base en el accuracy, este servira para los plots de roc, prec-recall y loss
    if best_accuracy < accuracy:
        best_accuracy = accuracy
        best_xgb = best_xgb_model
        best_cm = cm

# Calculo del promedio y la desviación estándar de las métricas
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

# Impresion de las métricas promedio y desviaciones
print("Average AUC: %.3f (+/- %.3f)" % (avg_auc, std_auc))
print("Average Accuracy: %.3f (+/- %.3f)" % (avg_accuracy, std_accuracy))
print("Average Recall: %.3f (+/- %.3f)" % (avg_recall, std_recall))
print("Average Precision: %.3f (+/- %.3f)" % (avg_prec, std_prec))
print("Average F1-score: %.3f (+/- %.3f)" % (avg_f1, std_f1))
print("Average Matthews Correlation Coefficient (MCC): %.3f (+/- %.3f)" % (avg_mcc, std_mcc))
print("Average Log Loss: %.3f (+/- %.3f)" % (avg_loss, std_loss))

# Impresion la matriz de confusión general
print("Overall Confusion Matrix:")
print(overall_confusion_matrix)

# plot ROC-AUC, con el mejor modelo obtenido
RocCurveDisplay.from_estimator(best_xgb, X_test, y_test)
plt.title(f'ROC Curve - XGBoost')
plt.show()

# plot Precision vs Recall, con el mejor modelo obtenido
PrecisionRecallDisplay.from_estimator(best_xgb, X_test, y_test)
plt.title(f'Precision-Recall Curve - XGBoost')
plt.show()

#calculo de los valores de perdida por iteraciones para el mejor modelo obtenido
results = best_xgb.evals_result()

#plot para los valores de peridad por iteraciones
plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss for XGBoost')
plt.legend()
plt.show()

#plot de matrices de confusion del modelo, para los resultados generales y resultados con una muestra y el mejor modelo
sns.heatmap(overall_confusion_matrix, annot=True)
plt.title("Matriz de Confusión - XGB")
plt.show()

sns.heatmap(overall_confusion_matrix / np.sum(overall_confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
plt.title("Matriz de Confusión - XGB (Porcentage)")
plt.show()

sns.heatmap(best_cm, annot=True)
plt.title("Matriz de Confusión - XGB")
plt.show()

sns.heatmap(best_cm / np.sum(best_cm), annot=True, fmt='.2%', cmap='Blues')
plt.title("Matriz de Confusión - XGB (Porcentage)")
plt.show()

# Codificador aplicado (autoencoder)
#este modelo implementa capas densas, en formato de cuello de botella y similar a un CNN
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
#NOTA: este modelo tuvo bueno resultados, aunque mas bajos tras tener menos capas pero sin casos de overfitting
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
#compilacion de modelo de autoencoder con metricas y optimizador adam, se toma un lr de 0.0001 por defecto
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=[
    metrics.AUC(), metrics.Accuracy(), metrics.F1Score(), metrics.Recall(),
    metrics.Precision(), metrics.R2Score(), metrics.MAPE, metrics.MAE])


#implementacion de una clase de callback para llevar el registro de perdida de un modelo de clasificacion binaria
#dicho modelo se generara para cada fold posteriormente
class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loss = []

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        loss, _ = self.model.evaluate(x_test, y_test, verbose=0)
        self.test_loss.append(loss)

#metricas de evaluacion de autoencoder (NN) y para sus mejores modelos y resultados
best_nn_model = None
best_nn_acc = 0
model = None
best_acc = 0
callback = None
fold = 0
history = None
model_nn_accuracies = []
roc_auc_scores_nn = []
precision_scores_nn = []
recall_scores_nn = []
f1_scores_nn = []
losses_nn = []
mcc_scores_nn = []
best_cm_nn = None
#metricas de evaluacion de autoencoder y XGB (XGB-NN) y para sus mejores modelos y resultados
model_xgb_accuracies = []
roc_auc_scores_xgb = []
precision_scores_xgb = []
recall_scores_xgb = []
f1_scores_xgb = []
losses_xgb = []
mcc_scores_xgb = []
best_cm_xgb = None

#variables para matrices de confusion
overall_cm_nn = np.zeros((2, 2))
overall_cm_xgb = np.zeros((2, 2))

y_test_all = []  # Almacenamiento de los valores reales de las etiquetas de clase para todos los folds
y_pred_prob_all_nn = []  # Almacenamiento de las probabilidades predichas por autoencoder (NN) para todos los folds
y_pred_prob_all_xgb = []  # Almacenamiento de las probabilidades predichas por XGB-NN para todos los folds

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    print(f"Current fold: {fold}")

    #entrenamiento de autoencoder
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
    #obtencion y decodificacion de datos en variables obtenidas tras el proceso de codificacion del autencoder
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[-2].output)
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Modelo de red neuronal para clasificación binaria determinado, secuencial y de capas densas
    model_nn = Sequential()
    model_nn.add(Input(shape=(X_train_encoded.shape[1],)))
    model_nn.add(Dense(32, activation='relu'))
    model_nn.add(Dense(1, activation='sigmoid'))
    model_nn.compile(loss='binary_crossentropy', optimizer='adam',
                     metrics=['accuracy'])

    #defincion de callback y variable para almacenar registros de perdida del modelo con autoencoder
    test_callback = TestCallback((X_test_encoded, y_test))
    history_nn = model_nn.fit(X_train_encoded, y_train, epochs=epochs, batch_size=256,
                              validation_data=(X_test_encoded, y_test), verbose=0,
                              callbacks=[test_callback])

    #prediccion de resultados para el autoencoder
    y_pred_nn = model_nn.predict(X_test_encoded)
    y_pred_nn = (y_pred_nn > 0.5).astype(int).flatten()

    y_pred_prob_nn = model_nn.predict(X_test_encoded, use_multiprocessing=True)
    y_pred_prob_all_nn.extend(y_pred_prob_nn)

    # registros de perdidad (loss) del autoencoder, sobre validation y training
    train_loss = history_nn.history['loss']
    val_loss = history_nn.history['val_loss']

    # Métricas para model_nn (variable del modelo de clasificacion binaria con el autoencoder)
    acc_nn = accuracy_score(y_test, y_pred_nn)
    precision_nn = precision_score(y_test, y_pred_nn)
    recall_nn = recall_score(y_test, y_pred_nn)
    f1_nn = f1_score(y_test, y_pred_nn)
    cm_nn = confusion_matrix(y_test, y_pred_nn)
    auc_nn = roc_auc_score(y_test, y_pred_nn)
    mcc_nn = matthews_corrcoef(y_test, y_pred_nn)
    loss_nn = log_loss(y_test, y_pred_nn)

    #almacenamiento de resultados de metricas en listas
    model_nn_accuracies.append(acc_nn)
    roc_auc_scores_nn.append(auc_nn)
    precision_scores_nn.append(precision_nn)
    recall_scores_nn.append(recall_nn)
    f1_scores_nn.append(f1_nn)
    losses_nn.append(loss_nn)
    mcc_scores_nn.append(mcc_nn)
    overall_cm_nn += cm_nn

    #calculo del accuracy (nuevamente, se puede simplemente considerar acc_nn sin problemas)
    current_acc_nn = accuracy_score(y_test, y_pred_nn)

    #verificacion de accuracies entre el modelo actual y el mejor, se alamcena un nuevo modelo en caso del actual ser mejor
    if best_nn_acc < current_acc_nn:
        best_nn_model = model_nn
        best_nn_acc = current_acc_nn
        history = history_nn
        best_cm_nn = cm_nn

    # Creacion del modelo de XGBoost
    xgb_model = XGBClassifier()

    # GridSearch con el modelo con sus hiperparametros, tal como en el modelo de XGBoost anterior
    grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='accuracy')

    # Entrenamiento del modelo, pero en este caso con las variables obtenidas del autoencoder codificadas
    grid_search_xgb.fit(X_train_encoded, y_train, early_stopping_rounds=5,
                        eval_metric=['logloss', 'error', 'error@0.5'],
                        eval_set=[(X_train_encoded, y_train), (X_test_encoded, y_test)],
                        verbose=False)

    # Obtencion del mejor modelo entrenado
    best_xgb_model = grid_search_xgb.best_estimator_

    # predicciones con el mejor modelo
    y_pred_prob = best_xgb_model.predict_proba(X_test_encoded)[:, 1]
    y_pred = best_xgb_model.predict(X_test_encoded)

    y_pred_prob_xgb = best_xgb_model.predict_proba(X_test_encoded)[:, 1]

    # Almacenamiento de los valores reales y predichos de las etiquetas de clase
    y_pred_prob_all_xgb.extend(y_pred_prob_xgb)

    y_test_all.extend(y_test)

    #calculo del accuracy del modelo actual
    current_acc = accuracy_score(y_test, y_pred)

    # Métricas para best_xgb_model (variable de XGB con autoencoder, NN)
    acc_xgb = accuracy_score(y_test, y_pred)
    precision_xgb = precision_score(y_test, y_pred)
    recall_xgb = recall_score(y_test, y_pred)
    f1_xgb = f1_score(y_test, y_pred)
    cm_xgb = confusion_matrix(y_test, y_pred)
    auc_xgb = roc_auc_score(y_test, y_pred_prob)
    mcc_xgb = matthews_corrcoef(y_test, y_pred)
    loss_xgb = log_loss(y_test, y_pred_prob)

    #almaceniamiento de metricas en listas
    model_xgb_accuracies.append(acc_xgb)
    roc_auc_scores_xgb.append(auc_xgb)
    precision_scores_xgb.append(precision_xgb)
    recall_scores_xgb.append(recall_xgb)
    f1_scores_xgb.append(f1_xgb)
    losses_xgb.append(loss_xgb)
    mcc_scores_xgb.append(mcc_xgb)
    overall_cm_xgb += cm_xgb

    #impresion de resultados obtenidos de las metricas
    print(f"Metrics for model_nn in fold {fold}:")
    print(f"Accuracy: {acc_nn}")
    print(f"AUC: {auc_nn}")
    print(f"Precision: {precision_nn}")
    print(f"Recall: {recall_nn}")
    print(f"F1 Score: {f1_nn}")
    print(f"Loss: {loss_nn}")
    print(f"MCC: {mcc_nn}")
    print("Matriz de confusión:")
    print(cm_nn)
    print()

    print(f"Metrics for best_xgb_model in fold {fold}:")
    print(f"Accuracy: {acc_xgb}")
    print(f"AUC: {auc_xgb}")
    print(f"Precision: {precision_xgb}")
    print(f"Recall: {recall_xgb}")
    print(f"F1 Score: {f1_xgb}")
    print(f"Loss: {loss_xgb}")
    print(f"MCC: {mcc_xgb}")
    print("Matriz de confusión:")
    print(cm_xgb)
    print()

    #defincion de mejor modelo en caso de mejro accuracy, tal como en casos previos
    if best_acc < current_acc:
        model = best_xgb_model
        best_acc = current_acc
        best_cm_xgb = cm_xgb

    fold += 1

# Promedios y desviaciones estándar para model_nn (recopilacion de valores del autoencoder)
mean_acc_nn = np.mean(model_nn_accuracies)
std_acc_nn = np.std(model_nn_accuracies)
mean_auc_nn = np.mean(roc_auc_scores_nn)
std_auc_nn = np.std(roc_auc_scores_nn)
mean_precision_nn = np.mean(precision_scores_nn)
std_precision_nn = np.std(precision_scores_nn)
mean_recall_nn = np.mean(recall_scores_nn)
std_recall_nn = np.std(recall_scores_nn)
mean_f1_nn = np.mean(f1_scores_nn)
std_f1_nn = np.std(f1_scores_nn)

print("Promedios y desviaciones estándar para model_nn:")
print(f"Accuracy: Mean = {mean_acc_nn}, Std = {std_acc_nn}")
print(f"AUC: Mean = {mean_auc_nn}, Std = {std_auc_nn}")
print(f"Precision: Mean = {mean_precision_nn}, Std = {std_precision_nn}")
print(f"Recall: Mean = {mean_recall_nn}, Std = {std_recall_nn}")
print(f"F1 Score: Mean = {mean_f1_nn}, Std = {std_f1_nn}")
print()

# Promedios y desviaciones estándar para best_xgb_model (recopilacion de valores de XGBoost con autoencoder)
mean_acc_xgb = np.mean(model_xgb_accuracies)
std_acc_xgb = np.std(model_xgb_accuracies)
mean_auc_xgb = np.mean(roc_auc_scores_xgb)
std_auc_xgb = np.std(roc_auc_scores_xgb)
mean_precision_xgb = np.mean(precision_scores_xgb)
std_precision_xgb = np.std(precision_scores_xgb)
mean_recall_xgb = np.mean(recall_scores_xgb)
std_recall_xgb = np.std(recall_scores_xgb)
mean_f1_xgb = np.mean(f1_scores_xgb)
std_f1_xgb = np.std(f1_scores_xgb)

print("Promedios y desviaciones estándar para best_xgb_model:")
print(f"Accuracy: Mean = {mean_acc_xgb}, Std = {std_acc_xgb}")
print(f"AUC: Mean = {mean_auc_xgb}, Std = {std_auc_xgb}")
print(f"Precision: Mean = {mean_precision_xgb}, Std = {std_precision_xgb}")
print(f"Recall: Mean = {mean_recall_xgb}, Std = {std_recall_xgb}")
print(f"F1 Score: Mean = {mean_f1_xgb}, Std = {std_f1_xgb}")
print()


# Crear el plot del ROC promedio para la red neuronal (autoencoder) y XGBoost con autoencoder
#para este caso se tomane en cuenta los resultados totales
RocCurveDisplay.from_predictions(y_test_all, y_pred_prob_all_nn)
plt.title(f'ROC Curve - NN')
plt.show()

RocCurveDisplay.from_predictions(y_test_all, y_pred_prob_all_xgb)
plt.title(f'ROC Curve - XGB-NN')
plt.show()

# Calcular la precisión y el recall promedio para la red neuronal (autoencoder)
prec_nn, recall_nn, _ = precision_recall_curve(y_test_all, y_pred_prob_all_nn)

# Calcular la precisión y el recall promedio para XGBoost con autoencoder
prec_xgb, recall_xgb, _ = precision_recall_curve(y_test_all, y_pred_prob_all_xgb)

# Crear el plot de precisión-recall promedio para la red neuronal (autoencoder) y XGBoost con autoencoder
PrecisionRecallDisplay.from_predictions(y_test_all, y_pred_prob_all_nn)
plt.title(f'Precision-Recall Curve - NN')
plt.show()

PrecisionRecallDisplay.from_predictions(y_test_all, y_pred_prob_all_xgb)
plt.title(f'Precision-Recall Curve - XGB-NN')
plt.show()

# Generacion de la gráfica de la pérdida por época, para autoencoder y XGBoost con autoencoder
import matplotlib.pyplot as plt
#autoencoder
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss for Sequential Model (NN)')
plt.legend()
plt.show()

#XGB y autoencoder
results = model.evals_result()

plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['logloss'], label='train')
plt.plot(results['validation_1']['logloss'], label='test')
plt.xlabel('Iterations')
plt.ylabel('Log Loss')
plt.title('Training and Validation Log Loss for XGBoost-NN')
plt.legend()
plt.show()

#matrices de confusion general y para el mejro modelo en autoencoder y XGB con autoencoder
sns.heatmap(overall_cm_nn, annot=True)
plt.title("Matriz de Confusión - NN")
plt.show()

sns.heatmap(overall_cm_nn / np.sum(overall_cm_nn), annot=True, fmt='.2%', cmap='Blues')
plt.title("Matriz de Confusión - NN (Porcentage)")
plt.show()

sns.heatmap(overall_cm_xgb, annot=True)
plt.title("Matriz de Confusión - XGB-NN")
plt.show()

sns.heatmap(overall_cm_xgb / np.sum(overall_cm_xgb), annot=True, fmt='.2%', cmap='Blues')
plt.title("Matriz de Confusión - XGB-NN (Porcentage)")
plt.show()

sns.heatmap(best_cm_nn, annot=True)
plt.title("Matriz de Confusión - NN")
plt.show()

sns.heatmap(best_cm_nn / np.sum(best_cm_nn), annot=True, fmt='.2%', cmap='Blues')
plt.title("Matriz de Confusión - NN (Porcentage)")
plt.show()

sns.heatmap(best_cm_xgb, annot=True)
plt.title("Matriz de Confusión - XGB-NN")
plt.show()

sns.heatmap(best_cm_xgb / np.sum(best_cm_xgb), annot=True, fmt='.2%', cmap='Blues')
plt.title("Matriz de Confusión - XGB-NN (Porcentage)")
plt.show()