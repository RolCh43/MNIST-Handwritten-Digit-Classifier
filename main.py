from os.path import join
from dataloader import MnistDataloader
from neural_network import NeuralNetwork
from trainer import Trainer
from utils import one_hot

if __name__ == "__main__":
    input_path = "./data"
    training_images = join(input_path, "train-images-idx3-ubyte")
    training_labels = join(input_path, "train-labels-idx1-ubyte")
    test_images = join(input_path, "t10k-images-idx3-ubyte")
    test_labels = join(input_path, "t10k-labels-idx1-ubyte")

    # Carrega dados
    mnist = MnistDataloader(training_images, training_labels, test_images, test_labels)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Pré-processamento
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    # Cria rede neural e treina
    nn = NeuralNetwork([784, 128, 64, 32, 16, 10], lr=0.1)
    trainer = Trainer(nn, X_train, y_train, X_test, y_test, epochs=10, batch_size=128)
    trainer.train()