import numpy as np
import time

class Trainer:
    def __init__(self, model, X_train, y_train, X_test, y_test,
                 epochs=10, batch_size=64, outfile="saida.txt"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.outfile = outfile

    def train(self):
        start = time.time()
        with open(self.outfile, "w", encoding="utf-8") as log:
            log.write("Treinamento da Rede Neural (sigmoide, 4 camadas ocultas)\n")
            n = self.X_train.shape[0]

            for epoch in range(1, self.epochs + 1):
                indices = np.arange(n)
                np.random.shuffle(indices)
                X_train, y_train = self.X_train[indices], self.y_train[indices]
                losses = []

                for i in range(0, n, self.batch_size):
                    Xb = X_train[i:i+self.batch_size]
                    yb = y_train[i:i+self.batch_size]

                    acts = self.model.forward(Xb)
                    grads_W, grads_b = self.model.backward(acts, yb)
                    self.model.update(grads_W, grads_b)
                    losses.append(self.model.mse(acts[-1], yb))

                y_pred = self.model.predict(self.X_test)
                val_loss = self.model.mse(y_pred, self.y_test)
                line = f"Época {epoch:02d}: MSE treino={np.mean(losses):.6f}, MSE teste={val_loss:.6f}"
                print(line)
                log.write(line + "\n")

            acc = np.mean(np.argmax(self.model.predict(self.X_test), axis=1) ==
                          np.argmax(self.y_test, axis=1))
            log.write(f"\nAcurácia final: {acc:.4f}\n")

        print(f"Treinamento concluído em {time.time() - start:.1f}s | Acurácia final: {acc:.4f}")