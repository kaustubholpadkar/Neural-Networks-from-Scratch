import numpy as np
import matplotlib.pyplot as plt
from nn import dnn


def main():
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.expand_dims(np.array([0., 1., 1., 0.]), axis=1)

    model = dnn.DNN(2, 5, 1)
    losses = model.train(x, y, epochs=200, learning_rate=0.6)

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
