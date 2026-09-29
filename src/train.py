from model import HopfieldNetwork
from data import get_training_data

def main():
    datos = get_training_data()
    tamano = len(datos[0])
    
    red = HopfieldNetwork(size=tamano)
    print("Entrenando la red...")
    red.train(datos)
    
    red.save_weights("pesos.npy")
    print("Entrenamiento finalizado. Pesos guardados en 'pesos.npy'.")

if __name__ == "__main__":
    main()