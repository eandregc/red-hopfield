# Red de Hopfield
Una red de Hopfield es un tipo de red neuronal artificial recurrente que funciona como un sistema de memoria asociativa. Su función principal es almacenar información y recuperar patrones completos o limpios a partir de datos de entrada incompletos o con ruido.

**Características clave:**

* **Arquitectura:** Está completamente conectada. Todas las neuronas se conectan entre sí, pero ninguna está conectada consigo misma.
* **Pesos simétricos:** La fuerza de conexión de la neurona A hacia la neurona B es exactamente la misma que de la B hacia la A.
* **Valores:** Sus nodos procesan datos binarios o bipolares (típicamente toman valores de 1 y -1).
* **Dinámica de energía:** La red evoluciona iterativamente minimizando una "función de energía". Cada cambio de estado de las neuronas hace que la red descienda hacia un valle de energía (mínimo local). Cuando la red se estabiliza y deja de cambiar, ha "recordado" o reconstruido el patrón.
* **Capacidad limitada:** Si se intenta almacenar un número excesivo de patrones en relación al tamaño de la red, los estados se mezclan y la red comienza a recordar patrones falsos (estados espurios).

## Distribución del Proyecto
proyecto_hopfield/
├── src/
│   ├── data.py       # Almacena los patrones limpios de entrenamiento y los ruidosos de prueba.
│   ├── model.py      # Contiene la lógica matemática y la clase HopfieldNetwork.
│   ├── train.py      # Script que entrena el modelo y exporta los pesos a 'pesos.npy'.
│   └── test.py       # Script que importa los pesos y evalúa la recuperación de patrones.
├── .gitignore        # Excluye la caché de Python y el archivo binario de pesos al subir a Git.
└── requirements.txt  # Dependencias del entorno virtual (numpy).