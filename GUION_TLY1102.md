# Guion de defensa · EP2 TLY1102 (10 minutos)

> Regla: no digas un número que no esté acá. Si te preguntan algo que no está, usá las plantillas del final.

---

## 0:00 – 0:40 · Qué resolvimos

> "El problema es de negocio: una plataforma audiovisual necesita etiquetar fotogramas de una serie de
> forma automática, porque hacerlo a mano cuesta linealmente con el catálogo. Entrené una CNN para
> clasificar 18 personajes. En test dio **42.92% de accuracy** y
> **0.3872 de F1 macro**."

## 0:40 – 2:00 · Los datos y mi variante

> "18,992 imágenes de entrenamiento y 890 de test, 18 clases, RGB.
> El desbalanceo original es de 6,27 a 1: Homer tiene 2.246 y Nelson 358. Eso importa porque el piso
> real no es el 5,56% del azar, sino el **11,8%** que acierta alguien que siempre responde Homer.
>
> Mi variante propia: **tope de 800 imágenes por clase** y partición con semilla 2026. El tope baja el
> desbalanceo a 2,2 a 1, así que el modelo ya no puede ganar respondiendo Homer. Conservé las 18 clases
> para que la comparación con el trabajo previo siga siendo sobre el mismo problema."

## 2:00 – 3:30 · La convolución, en dos frases

> "Un kernel es una matriz chica de pesos que se desliza por la imagen. En cada posición multiplica sus
> valores por los píxeles de abajo, suma, agrega un bias y pasa por ReLU. Lo que sale es un mapa de
> características: una imagen nueva donde cada píxel dice cuánto se parecía esa región al patrón.
> Los valores del kernel **no los elige nadie: los aprende la red.**"

> "La ventaja en parámetros: una neurona densa que recibe 100x100 necesita 10.000 pesos. Un kernel de
> 5x5 son 25, y esos 25 se reutilizan en toda la imagen. Por eso mi CNN tiene
> **2,196,178 parámetros** y el MLP **3,150,610**: 1.4 veces más."

## 3:30 – 5:00 · La arquitectura y la decisión que importa

> "Tres bloques de 32, 64 y 128 filtros. Cada uno: convolución 3x3 con padding `same`, BatchNorm, ReLU y
> MaxPool que divide por dos: 64, 32, 16, 8. Al final aplanamos y una densa de 256 con dropout del 50%
> decide entre las 18 clases.
>
> La decisión que quiero defender: la versión de manual termina con `GlobalAveragePooling`. **La probé y
> dio 62,25%.** La pista estaba en la brecha train-validación: 0,024, demasiado chica. Eso es
> **subajuste**, no sobreajuste: faltaba capacidad. GAP promedia los mapas y pierde el **dónde**, y para
> distinguir a Homer de Flanders importa dónde está el bigote respecto a los ojos. Con Flatten + Dense
> subió al orden del 97%."

## 5:00 – 6:30 · Entrenamiento

> "18 épocas, **31.0 minutos en CPU**. Adam con tasa 1e-3, pérdida
> `sparse_categorical_crossentropy`, batch 128. Aumento de datos dentro del modelo: flip, rotación de
> 5 grados, zoom de 10%.
>
> El orden de las técnicas importa: primero busco accuracy, después ataco el sobreajuste con
> regularización y aumento de datos, y **recién al final** aplico early stopping para quedarme con la
> mejor época. Si lo aplico al principio, escondo el problema en vez de resolverlo.
>
> Mejor validación: 0.4289. Brecha train-validación final: -0.1248."

## 6:30 – 8:00 · CNN vs MLP — el corazón del encargo

> "Para que la comparación sea honesta, el MLP compite en igualdad de condiciones: mismos tensores, misma
> partición, mismo batch, mismo optimizador, mismo máximo de épocas. Lo único que cambia es la
> arquitectura.
>
> Resultado: CNN **0.4292** contra MLP **0.1449** de accuracy. En F1 macro,
> 0.3872 contra 0.0893. Y la CNN lo logra con **1.4
> veces menos parámetros**.
>
> ¿Por qué? El MLP aplana la imagen a 12.288 valores y pierde la vecindad espacial: no sabe que un píxel
> está al lado del otro. La convolución es una **restricción**: le prohíbe al modelo aprender un detector
> distinto para cada posición. Esa restricción es lo que lo hace generalizar mejor."

## 8:00 – 9:20 · Mapas de características y de activación

> "En conv1 los mapas se parecen a la imagen: detectan bordes y contrastes. En conv2 aparecen formas:
> contornos de ojos, boca. En conv3, con 8x8, ya no se reconoce la imagen: cada celda resume una región
> amplia y responde a combinaciones. Esa jerarquía es lo que un MLP no construye.
>
> El Grad-CAM es otra cosa: dice **dónde miró el modelo para decidir**. Si el calor está en el rostro,
> usó la evidencia correcta. Si está en el fondo, está usando un atajo contextual, y eso explica errores
> que la matriz de confusión sola no muestra."

## 9:20 – 10:00 · Errores y cierre

> "Los tres peores recalls son comic book guy (0.000), edna krabappel (0.000), nelson muntz (0.000). Las causas probables: el resize a 64x64 comprime el rostro
> cuando el personaje aparece chico; los personajes comparten piel y pelo amarillos; y las clases con
> menos imágenes reciben menos gradiente.
>
> Tres conclusiones. **Una:** la arquitectura no se copia, se mide — descarté GAP con datos.
> **Dos:** el test no se usó para ninguna decisión, y por eso el número es honesto.
> **Tres:** los mapas de activación no son adorno, son auditoría."

---

# Preguntas probables

**"¿Por qué no usaste transfer learning?"**
> "Lo medí: una ResNet50 con base congelada tiene 23,6 M de parámetros y sólo 36.882 entrenables, y
> tardaba 168 segundos por época contra unos 50 de mi CNN. Además comparar una red que ya vio un millón
> de imágenes contra una desde cero **no es una comparación limpia**. Lo correcto sería usarla como punto
> de partida y afinar sus últimas capas: eso es trabajo futuro."

**"¿Por qué 64x64 y no 128?"**
> "Porque multiplica por cuatro el cómputo: de ~47 a ~190 segundos por época en este equipo. La ganancia
> no estaba justificada con el tiempo disponible."

**"¿El train terminó alto y la validación más baja, no es sobreajuste?"**
> "Sí, la brecha es -0.1248 y está documentada. La pregunta es si afecta la generalización: en test dio
> 0.4292. Para cerrarla subiría el aumento de datos antes que el dropout."

**"¿Qué es el bias en la convolución?"**
> "Una constante que se suma al resultado de la convolución antes de la activación. Igual que en una
> neurona densa: permite desplazar el umbral de activación del filtro."

**"¿Por qué F1 macro y no accuracy?"**
> "Porque macro promedia las 18 clases sin importar cuántas imágenes tiene cada una: trata a
> Nelson igual que a Homer. Es la medida que no se puede inflar con la clase mayoritaria."

**"¿Por qué el test da más que la validación?"**
> "El conjunto de test está balanceado 47–50 por clase y el de validación no. Un test balanceado favorece
> la accuracy global. Lo importante es que el test **no participó de ninguna decisión**."

**"¿Qué es BatchNorm y por qué lo pusiste?"**
> "Normaliza las activaciones por lote: resta la media y divide por el desvío, y aprende un factor de
> escala. Estabiliza el entrenamiento y permite tasas de aprendizaje más altas. Necesita suficientes
> pasos por época para estimar bien sus medias móviles: acá hay unos 76."

---

# Plantillas para lo que no sepas

1. **No lo medí:** "Eso no lo medí. Lo que sí sé es [hecho relacionado]. Si tuviera que hipotetizar
   diría [razonamiento], pero no lo tengo medido."
2. **No probé esa variante:** "Esa variante no la probé. Lo que sí medí fue [variante del cuaderno], y
   dio [número]."
3. **No revisé el detalle:** "No revisé ese detalle. Lo puedo buscar en el cuaderno ahora."

**Lo que nunca hay que hacer:** inventar un número, o defender algo que no se hizo.
