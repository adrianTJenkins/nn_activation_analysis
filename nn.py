import numpy as np

class NN:

    def __init__(self, name):
        self.activation_name = name

    activation_name = ""
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    np.random.seed(1)

    synaptic_weights = 2 * np.random.random((3, 1)) - 1
    def sigmoid(self, x, is_derivative = False):
        if is_derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    print('Random starting synaptic weights: ')
    print(synaptic_weights)
        

    def learn(self, activation_method, x_value = 0, bool_value = False):
        for i in range(100000):

            input_layer = self.training_inputs

            x_value = np.dot(input_layer, self.synaptic_weights)
            bool_value = False
            outputs = activation_method(x_value, bool_value)

            error = self.training_outputs - outputs

            x_value = outputs
            bool_value = True
            adjustments = error * activation_method(x_value, bool_value)

            self.synaptic_weights += np.dot(input_layer.T, adjustments)

        print(self.activation_name)
        print('Synaptic weights after training')
        print(self.synaptic_weights)

        print('\nOutputs after training')
        print(outputs)

    def main(self):
        if self.activation_name == "sigmoid":
            self.learn(self.sigmoid)
        else:
            print(self.activation_name)

n = NN("sigmoid")
n.main()
