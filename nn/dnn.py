import numpy as np
from utils import activations


class DNN:
    """
    Class for a single hidden layer Neural Network
    """

    def __init__(self, input_units, hidden_units, output_units):
        """
        Constructor of DNN
        :param input_units: Number of Input units
        :param hidden_units: Number of Nodes in Hidden Layer
        :param output_units: Number of Output units
        """

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.W1 = np.random.normal(size=(self.input_units, self.hidden_units))
        self.B1 = np.random.normal(size=(1, self.hidden_units))

        self.W2 = np.random.normal(size=(self.hidden_units, self.output_units))
        self.B2 = np.random.normal(size=(1, self.output_units))

    def train(self, train_x, train_y, epochs=100, learning_rate=0.01):
        """

        :param train_x: training features data
        :param train_y: training label data
        :param epochs: number of training epochs
        :param learning_rate: learning rate
        :return: list of losses
        """
        losses = []
        for epoch in range(epochs):

            Z1 = np.matmul(train_x, self.W1) + self.B1
            A1 = activations.sigmoid(Z1)
            
            Z2 = np.matmul(A1, self.W2) + self.B2
            A2 = activations.sigmoid(Z2)

            loss = np.sum(np.square(train_y - A2))
            losses.append(loss)

            dA2 = A2 - train_y
            dZ2 = activations.sigmoid(Z2, derivative=True) * dA2
            dA1 = np.matmul(dZ2, self.W2.T)
            dW2 = np.matmul(A1.T, dZ2)
            dB2 = dZ2
            dZ1 = activations.sigmoid(Z1, derivative=True) * dA1
            dW1 = np.matmul(train_x.T, dZ1)
            dB1 = dZ1

            self.W1 = self.W1 - learning_rate * dW1
            self.B1 = self.B1 - learning_rate * dB1
            self.W2 = self.W2 - learning_rate * dW2
            self.B2 = self.B2 - learning_rate * dB2

        return losses

    def predict(self, x):
        """
        Predict the output for given input
        :param x: input data
        :return: prediction
        """
        Z1 = np.matmul(x, self.W1) + self.B1
        A1 = activations.sigmoid(Z1)

        Z2 = np.matmul(A1, self.W2) + self.B2
        A2 = activations.sigmoid(Z2)
        return A2
