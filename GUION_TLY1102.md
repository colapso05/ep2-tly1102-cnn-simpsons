# Guion de defensa · EP2 TLY1102 (10 minutos)

> No digas un número que no esté acá. Si preguntan algo que no está, usá las plantillas del final.

## 0:00 – 0:40 · Qué resolvimos
> "Una plataforma audiovisual necesita etiquetar fotogramas automáticamente; a mano el costo crece con
> el catálogo. Entrené una CNN para clasificar 18 personajes. En test dio **97.87% de accuracy**
> y **0.9787 de F1 macro**, con 19 errores en 890 imágenes."

## 0:40 – 2:00 · Los datos
> "18992 imágenes de entrenamiento y 890 de test, 18 clases, RGB. El desbalanceo
> es de 6.27 a 1. El baseline real no es el 5,56% del azar sino el **11,8%** que acierta
> quien siempre responde Homer. El test está balanceado, 47 a 50 por clase, así que no premia favorecer
> a Homer."

## 2:00 – 3:20 · La convolución
> "Un kernel es una matriz chica de pesos que se desliza por la imagen: multiplica, suma, agrega un bias
> y pasa por ReLU. Lo que sale es un mapa de características. Los valores del kernel **no los elige
> nadie: los aprende la red**.
> Una neurona densa que recibe 100x100 necesita 10.000 pesos; un kernel de 5x5 son 25, reutilizados en
> toda la imagen. Por eso mi CNN tiene **2,196,722 parámetros** y el MLP
> **3,150,610**."

## 3:20 – 4:50 · La arquitectura y la decisión que importa
> "Tres bloques de 32, 64 y 128 filtros: convolución 3x3, BatchNorm, ReLU y MaxPool que divide por dos.
> Al final aplanamos, una densa de 256 con dropout del 50% y softmax sobre 18 clases.
> La decisión que quiero defender: la versión de manual termina con `GlobalAveragePooling`. **La probé y
> dio 62,25%.** La pista era la brecha train-validación de 0,024: demasiado chica, eso es subajuste, no
> sobreajuste. GAP promedia los mapas y pierde el **dónde**. Con Flatten + Dense subió al 97%."

## 4:50 – 6:10 · CNN vs MLP
> "Mismos tensores, mismo test. Resultado: CNN **0.9787** contra MLP
> **0.1202** de accuracy. En F1 macro, 0.9787 contra 0.0543.
> Y la CNN lo logra con menos parámetros. El MLP aplana a 12.288 valores y pierde la vecindad espacial.
> La convolución es una **restricción**: le prohíbe aprender un detector distinto para cada posición."

## 6:10 – 7:20 · Feature maps y Grad-CAM
> "En las primeras capas los mapas detectan bordes; en las últimas responden a combinaciones. El Grad-CAM
> dice **dónde miró** el modelo: en la mayoría de las clases el calor está en el rostro. En Marge y
> Comic Book Guy se abre al torso, y eso lo digo yo antes de que lo pregunten."

## 7:20 – 8:40 · La variante propia
> "Convertí el clasificador cerrado en uno con opción de rechazo: si la confianza no supera 0,70, el
> sistema responde *no identificado* y deriva a revisión humana. Con ese umbral etiqueto el
> **96.2%** del volumen con **99.07%** de precisión, contra
> 97.87% sin rechazo. En un catálogo grande eso reduce el trabajo manual en un orden de
> magnitud sin degradar la calidad."

## 8:40 – 10:00 · Errores y cierre
> "Los peores recalls son comic book guy (0.939), kent brockman (0.940), nelson muntz (0.940).
> Causas: el resize comprime el rostro cuando el personaje aparece chico; los personajes comparten piel
> y pelo amarillos; y las clases con menos imágenes reciben menos gradiente.
> Tres conclusiones. **Una:** la arquitectura se mide, no se copia. **Dos:** el test no se usó para
> ninguna decisión. **Tres:** el umbral de rechazo hace el modelo usable en producción."

---

# Preguntas probables

**"¿Por qué no usaste transfer learning?"**
> "Lo medí: ResNet50 con base congelada tiene 23,6 M de parámetros y sólo 36.882 entrenables, y tardaba
> 168 s por época. Comparar una red que ya vio un millón de imágenes contra una desde cero no es una
> comparación limpia. Lo correcto sería afinarla; es trabajo futuro."

**"¿Por qué 64x64 y no 128?"**
> "Multiplica por cuatro el cómputo: de ~47 a ~190 segundos por época en este equipo."

**"¿Por qué F1 macro y no accuracy?"**
> "Macro promedia las 18 clases sin importar cuántas imágenes tiene cada una: trata a Nelson igual que a
> Homer. No se puede inflar con la clase mayoritaria."

**"¿Por qué el test da más que la validación?"**
> "El test está balanceado 47–50 por clase y la validación no. Lo importante es que el test no participó
> de ninguna decisión."

**"¿Qué es el bias en la convolución?"**
> "Una constante que se suma al resultado antes de la activación: desplaza el umbral del filtro."

# Plantillas
1. **No lo medí:** "Eso no lo medí. Lo que sí sé es [hecho]. Si tuviera que hipotetizar diría
   [razonamiento], pero no está medido."
2. **No probé esa variante:** "No la probé. Lo que sí medí fue [variante], y dio [número]."
3. **No revisé el detalle:** "No lo revisé. Lo puedo buscar en el cuaderno ahora."
