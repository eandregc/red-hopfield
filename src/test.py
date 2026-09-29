import numpy as np
from model import HopfieldNetwork
from data import get_test_data

def main():
    datos_prueba = get_test_data()
    tamano = len(datos_prueba["noisy"])
    
    red = HopfieldNetwork(size=tamano)
    red.load_weights("pesos.npy")
    
    print("Iniciando prueba con patrón ruidoso...")
    recuperado = red.predict(datos_prueba["noisy"])
    
    print("\nPatrón Original:\n", datos_prueba["original"].reshape(3, 3))
    print("\nPatrón Ruidoso:\n", datos_prueba["noisy"].reshape(3, 3))
    print("\nPatrón Recuperado:\n", recuperado.reshape(3, 3))
    
    exito = np.array_equal(recuperado, datos_prueba["original"])
    print(f"\nRecuperación exitosa?: {exito}")

if __name__ == "__main__":
    main()