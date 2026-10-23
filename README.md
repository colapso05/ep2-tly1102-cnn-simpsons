# EP2 · TLY1102 — Clasificación de personajes de Los Simpsons con una CNN

Informe técnico de la Evaluación Parcial N°2 de **Técnicas Avanzadas de Machine Learning I**.
Todas las cifras de este documento salen del cuaderno
`notebooks/EP2_TLY1102_CNN_Simpsons_final.ipynb`, que reproduce cada etapa.

---

## 1. Problema de negocio

Una plataforma de contenido audiovisual necesita **etiquetar automáticamente** fotogramas de una serie
animada para alimentar su buscador y sus recomendaciones. Hoy lo hace una persona a mano y el costo
crece con el catálogo.

**Solución:** una CNN que devuelva el personaje del fotograma y, cuando no esté segura, derive la
imagen a revisión humana en vez de arriesgar una etiqueta falsa.

### KPIs

| KPI | Definición | Meta | Resultado |
|---|---|---|---|
| Accuracy en test | aciertos sobre el conjunto que no participó del diseño | ≥ 0,85 | **0.9787** |
| F1 macro | promedio no ponderado del F1 de las 18 clases | ≥ 0,85 | **0.9787** |
| Precisión con rechazo | precisión sobre lo etiquetado automáticamente (umbral 0,70) | ≥ 0,99 | **99.07%** |
| Cobertura con rechazo | fracción del volumen etiquetada sin intervención humana | ≥ 0,90 | **96.2%** |

### Objetivos

1. Preparar las imágenes en formato tensorial apto para una CNN.
2. Diseñar e implementar una arquitectura convolucional justificando cada componente.
3. Entrenar registrando el comportamiento del aprendizaje.
4. Evaluar con accuracy, precision, recall, F1 y matriz de confusión.
5. Comparar la CNN contra la red densa sobre los mismos datos.
6. Interpretar mapas de características y el mapa de activación de cada clase.
7. Analizar los errores y sus causas probables.

---

## 2. Entorno

TensorFlow/Keras sobre CPU. Todas las mediciones de tiempo de este informe se tomaron en el mismo
equipo, por eso son comparables entre sí.

---

## 3. Fuente de datos

| Atributo | Valor |
|---|---|
| Origen | The Simpsons Dataset (Kaggle, `alfaro96/los-simpson`) |
| Imágenes de entrenamiento | 18992 |
| Imágenes de test | 890 |
| Clases | 18 personajes (de 47 carpetas del dataset original) |
| Resolución original | 320×432 px de mediana (ancho: 256–1912 px) |
| Canales | 3 (RGB) |

**Por qué 18 y no 47 clases.** Las 29 carpetas restantes tienen muy pocas imágenes; con tan pocos
ejemplos por clase el modelo no puede aprender y la partición de validación queda desbalanceada.

**Sobre el cambio de conjunto respecto de la EP1.** El trabajo anterior usaba imágenes **monocanal de
28×28** con letras aisladas sobre fondo limpio. Este conjunto es **tricanal RGB**, con fondo complejo y
resolución variable. Cambiar a la vez la arquitectura y la naturaleza del dato haría inválida la
comparación que pide la evaluación; por eso se trabaja sobre el conjunto RGB, que además es el que la
pauta autoriza.

**Uso ético.** Fotogramas de una obra audiovisual con fines académicos. No contienen datos personales
ni información sensible y el repositorio no redistribuye las imágenes, sólo los tensores derivados y
las métricas.

---

## 4. Análisis exploratorio

- 18992 imágenes de entrenamiento y 890 de test, sin archivos corruptos.
- **Desbalanceo 6.27×**: Homer 2.246 imágenes contra Nelson 358. El piso real no es el
  5,56% del azar sino el **11,8%** que acierta quien siempre responde Homer.
- El test **está balanceado** (47–50 por clase): no premia favorecer a la clase mayoritaria.
- Anchos de 256 a 1912 px: el `resize` es obligatorio, no opcional. Y a 64×64 **deforma las
  panorámicas**: es una pérdida asumida, porque recortar el centro introduciría otro sesgo (el
  personaje no siempre está centrado).
- El color es informativo (pelo, ropa), así que se conservan los 3 canales en lugar de pasar a gris.

---

## 5. Preprocesamiento y partición

### 5.1 Normalización

Los píxeles vienen en 0–255 y se llevan a 0–1. Se hace **dentro del modelo** con
`layers.Rescaling(1/255)`, no antes:

| Forma | RAM de los datos | Tiempo por época |
|---|---|---|
| Convertir todo a `float32` antes | 760 MB | 187 s |
| Mantener `uint8` + `Rescaling` en el modelo | **190 MB** | **47 s** |

Las dos formas son matemáticamente idénticas. La segunda es más rápida porque no copia 760 MB por
lote, y deja el modelo guardado capaz de procesar una imagen cruda sin que nadie tenga que recordar
dividirla.

### 5.2 Partición 80/20

**Decisión: partición 80/20.** Se aparta **20% del entrenamiento como validación** (3.798 imágenes
sobre 18.992) y el test queda fuera de toda decisión. La proporción equilibra dos necesidades que
compiten: más datos para que el modelo aprenda, contra más datos para decidir con poca incertidumbre.
Con este volumen, el 20% deja un error estándar de aproximadamente ±0,28 puntos porcentuales en la
métrica de validación; bajarlo al 15% lo empeora a ±0,32 sin ganancia apreciable en entrenamiento.

Dos garantías: `stratify=y` mantiene la proporción de cada personaje en las dos partes —sin eso, una
clase con 358 imágenes podría quedar subrepresentada en validación y las curvas serían engañosas— y
`random_state` fija la partición para que sea idéntica en cada ejecución.

**El test (890 imágenes) no se toca hasta la evaluación final**: no participa del entrenamiento, ni de
la parada temprana, ni de la elección de hiperparámetros.

---

## 6. Arquitectura

| Componente | Configuración | Por qué |
|---|---|---|
| Filtros | 32 → 64 → 128 | se duplican al reducir resolución: cada vez hay menos posición y más "qué" |
| Kernel | 3×3 | bordes y curvas con 9 pesos por canal; dos 3×3 equivalen a un 5×5 con menos parámetros |
| Padding | `same` | sin padding los bordes se recorren menos veces y su información se pierde |
| Activación | ReLU | no satura en el lado positivo y es barata de derivar |
| Pooling | MaxPooling 2×2 | 64→32→16→8 y conserva la activación más fuerte de cada región |
| Normalización | BatchNorm | estabiliza la escala de activaciones y acelera la convergencia |
| Clasificador | `Flatten` + Dense(256) + Dropout(0.5) + Dense(18, softmax) | ver abajo |
| Pérdida | `sparse_categorical_crossentropy` | etiquetas enteras; entropía cruzada entre distribución predicha y real |
| Optimizador | Adam (1e-3) | tasa adaptativa por parámetro |
| **Parámetros** | **2.196.722** | ~2,2 M |

### La decisión que se tomó midiendo

`GlobalAveragePooling` es la recomendación de manual para bajar parámetros. **Se probó y dio 62,25%.**
El diagnóstico salió de la brecha entre entrenamiento y validación, que era de 0,024: demasiado chica.
Eso no es sobreajuste, es **subajuste** — al modelo le faltaba capacidad. GAP promedia los mapas y
pierde la información de **dónde** está cada rasgo, y para distinguir a Homer de Flanders importa dónde
está el bigote respecto de los ojos. Con `Flatten` y una densa de 256 el resultado subió al orden del
97%.

---

## 7. Entrenamiento

25 épocas, **25,7 minutos** en CPU. El orden de las técnicas importa: primero se busca accuracy,
después se ataca el sobreajuste con regularización y aumento de datos, y **recién al final** se aplica
parada temprana (`patience=6` sobre `val_loss`, `restore_best_weights`) para quedarse con la mejor
época. Aplicarla antes esconde el problema en vez de resolverlo.

Las curvas de entrenamiento y validación están en la sección 7.2 del cuaderno.

---

## 8. Experimentos controlados

Sobre una muestra de 6.000 imágenes se corrieron cinco experimentos con una sola variable cambiada
cada vez: arquitectura, profundidad, optimizador, aumento de datos y regularización. El objetivo es
que ninguna decisión de diseño quede sin respaldo medido.

### 8.3 Comparación CNN frente a red totalmente conectada

Misma cabeza de salida, mismos píxeles, mismos datos y mismas épocas. Lo único que cambia es la
arquitectura.

| | CNN (3 bloques) | Red densa (FC) |
|---|---|---|
| val_accuracy | **0,5928** | 0,1182 |
| Parámetros | **2.196.722** | 6.301.202 |
| Tiempo de entrenamiento | 214 s | **30 s** |
| Brecha train−validación | +0,4069 | +0,0005 |

**Diferencia: +47,46 puntos porcentuales a favor de la CNN, con tres veces menos parámetros.**

La red densa no sólo obtiene peor accuracy: lo hace con muchos más parámetros. La explicación es que
una capa densa aprende **un peso por cada píxel en cada posición**, mientras que la convolución
reutiliza el mismo kernel en toda la imagen. La brecha de 0,0005 de la red densa tampoco es una virtud:
significa que no aprendió lo suficiente ni para sobreajustar.

**Ventajas y desventajas.**

| | Ventajas | Desventajas |
|---|---|---|
| CNN | menos parámetros, invariancia a traslación, jerarquía de rasgos | más lenta por época; requiere estructura espacial en los datos |
| Red densa | simple y muy rápida de entrenar | ~3× más parámetros, sensible a la posición, no generaliza |

En el resto de los experimentos se mide profundidad, optimizador, aumento de datos y regularización;
los resultados están tabulados en 8.8 del cuaderno.

---

## 9. Modelo final

La configuración definitiva sale de los experimentos: bloques 32/64/128, `Flatten + Dense(256)`,
Dropout 0,5, Adam 1e-3, con aumento de datos en línea. El modelo entrenado se guarda en
`models/cnn_simpsons_final.keras`.

---

## 10. Evaluación en test

| Métrica | Valor |
|---|---|
| Accuracy | **0.9787** |
| Precision macro | 0.9794 |
| Recall macro | 0.9787 |
| F1 macro | **0.9787** |
| Errores | **19 de 890** |

![Matriz de confusión](images/fig_13_matriz_final.png)

Cada fila es el personaje real y cada columna lo predicho; la diagonal son los aciertos. Se reporta
**F1 macro** y no sólo accuracy porque macro promedia las 18 clases sin importar cuántas imágenes tiene
cada una: trata a Nelson igual que a Homer y no se puede inflar con la clase mayoritaria.

### Métricas por clase

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| abraham grampa simpson | 1.000 | 0.979 | 0.990 | 48 |
| apu nahasapeemapetilon | 0.943 | 1.000 | 0.971 | 50 |
| bart simpson | 0.980 | 0.980 | 0.980 | 50 |
| charles montgomery burns | 0.979 | 0.979 | 0.979 | 48 |
| chief wiggum | 0.961 | 0.980 | 0.970 | 50 |
| comic book guy | 1.000 | 0.939 | 0.968 | 49 |
| edna krabappel | 0.961 | 0.980 | 0.970 | 50 |
| homer simpson | 0.943 | 1.000 | 0.971 | 50 |
| kent brockman | 1.000 | 0.940 | 0.969 | 50 |
| krusty the clown | 0.980 | 0.980 | 0.980 | 50 |
| lisa simpson | 1.000 | 0.960 | 0.980 | 50 |
| marge simpson | 1.000 | 1.000 | 1.000 | 50 |
| milhouse van houten | 0.961 | 1.000 | 0.980 | 49 |
| moe szyslak | 0.962 | 1.000 | 0.980 | 50 |
| ned flanders | 0.980 | 0.980 | 0.980 | 49 |
| nelson muntz | 1.000 | 0.940 | 0.969 | 50 |
| principal skinner | 0.980 | 0.980 | 0.980 | 50 |
| sideshow bob | 1.000 | 1.000 | 1.000 | 47 |

**Peores recalls:** comic book guy (0.939), kent brockman (0.940), nelson muntz (0.940).

---

## 11. Mapas de características

Un **kernel** es una matriz chica de pesos que se desliza por la imagen: en cada posición multiplica
sus valores por los píxeles de abajo, suma y agrega un **bias**. El resultado es un **mapa de
características**. Los valores del kernel **no los elige nadie: los aprende la red**.

![Feature maps](images/fig_09_feature_maps.png)

En las primeras capas los mapas se parecen a la imagen: bordes, contrastes y zonas planas. En las
últimas ya no se reconoce la imagen: cada celda resume una región amplia y responde a combinaciones.
Esa jerarquía es lo que una red densa no construye.

## 12. Mapa de activación por clase (Grad-CAM)

![Grad-CAM](images/fig_10_gradcam_por_clase.png)

El feature map dice *qué* detectó un filtro; el Grad-CAM dice **dónde miró el modelo para decidir esa
clase**. En la mayoría de las 18 clases el calor se concentra sobre el rostro o la cabeza. En Marge y
Comic Book Guy se abre hacia el torso: son las clases donde el modelo se apoya en más contexto que la
cara, y conviene declararlo en lugar de esquivarlo.

---

## 13. Análisis de errores

Con 19 errores en 890 imágenes, las clases más débiles son las que tienen menos
ejemplos en el dataset. **Causas probables:**

1. **Poca resolución del rostro.** El `resize` a 64×64 comprime a los personajes que aparecen pequeños
   en el fotograma original; el rasgo discriminante se pierde.
2. **Atributos compartidos.** Piel y pelo amarillos y camisa blanca son píxeles idénticos entre
   personajes; el modelo debe apoyarse en detalles finos.
3. **Fondo como atajo.** Si un personaje aparece siempre en la misma locación, el modelo puede aprender
   el escenario. El Grad-CAM permite detectarlo.
4. **Clases minoritarias.** Nelson (358) y Edna comparten menos gradiente que Homer (2.246) y muestran
   peor recall.

---

## 14. Variante propia: clasificación con opción de rechazo

La pauta exige una variante que diferencie el trabajo. Se implementó un **umbral de confianza**: si la
probabilidad máxima no supera el umbral, el sistema responde `no identificado` y el fotograma se deriva
a revisión humana. El umbral opera sobre la probabilidad que el modelo ya produce, así que **no
requiere reentrenar**: la variante se evalúa aislando un solo cambio.

| Umbral | Cobertura | Precisión sobre lo aceptado | Aceptadas |
|---|---|---|---|
| 0.4 | 99.8% | 97.97% | 888 |
| 0.5 | 99.3% | 98.19% | 884 |
| 0.6 | 97.6% | 98.50% | 869 |
| 0.7 | 96.2% | 99.07% | 856 |
| 0.8 | 95.2% | 99.29% | 847 |
| 0.9 | 91.9% | 99.63% | 818 |

**Lectura de negocio.** Con umbral 0,70 se etiqueta automáticamente el **96.2%** del
volumen con **99.07%** de precisión, contra 97.87% sin rechazo. Se gana
precisión cediendo cobertura, y lo que se deriva a revisión humana es justamente la cola donde se
concentran los errores.

![Umbral de rechazo](images/fig_12_umbral_rechazo.png)

---

## 15. Conclusiones

1. **La arquitectura no se copia, se mide.** `GlobalAveragePooling` dio 62,25% y `Flatten + Dense` llevó
   el resultado al orden del 97%; el diagnóstico salió de la brecha train−validación.
2. **La CNN supera a la red densa con tres veces menos parámetros** (+47,46 puntos en el experimento
   controlado).
3. **El test no se usó para ninguna decisión**, y por eso el número es honesto.
4. **El umbral de rechazo convierte el clasificador en algo usable**: 96.2% del volumen
   automático con 99.07% de precisión.

### 15.3 Limitaciones

- **CPU**: se mantuvo 64×64 porque 128×128 multiplica por cuatro el cómputo (de 47 a ~190 s por época).
- La comparación de arquitecturas se hizo sobre una muestra de 6.000 imágenes para poder correr cinco
  experimentos en el tiempo disponible; el modelo final se entrenó con las 18.992.
- El aumento de datos con rotación y zoom frenaba la convergencia en este presupuesto de cómputo.

### 15.4 Trabajo futuro

- **Transfer learning** afinando las últimas capas de una ResNet50 (23,6 M de parámetros, 36.882
  entrenables, 168 s por época). Comparar una red preentrenada contra una desde cero no es una
  comparación limpia; usarla como punto de partida sí.
- **Tres semillas** por configuración para separar el efecto real del ruido.
- Cerrar la brecha train−validación con más aumento de datos antes que con más dropout.

---

## 16. Estructura y reproducibilidad

```
notebooks/EP2_TLY1102_CNN_Simpsons_final.ipynb   cuaderno ejecutado (entregable principal)
data/      metricas_por_clase_final.csv, umbral_rechazo.json, eda_resumen.json
images/    fig_01 … fig_13
models/    cnn_simpsons_final.keras
README.md
```

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/EP2_TLY1102_CNN_Simpsons_final.ipynb --ExecutePreprocessor.timeout=7200
```

Los tensores `X_train.npy` / `X_test.npy` no se versionan (233 MB): se regeneran desde `data/raw/`.
