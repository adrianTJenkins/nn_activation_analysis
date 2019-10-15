import numpy as np

class NN:

    def __init__(self, name, training_in, training_out):
        np.random.seed(1)
        self.activation_name = name
        self.training_inputs = training_in
        self.training_outputs = training_out
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
    
    #Sigmoid Function
    def sigmoid(self, x, is_derivative = False):
        if is_derivative:
            return x * (1 - x)
        return 1 / (1 + np.e ** (-x))

    #tanh Function
    def tanh(self, x, is_derivative = False):
        if is_derivative:
            return 1 - np.tanh(x) ** 2
        else:
            return np.tanh(x)

    #Relu Function
    def relu(self, x, is_derivative = False):
        if is_derivative:
            return np.sign(x)
        else:
            return np.maximum(0, x)

    def learn(self, activation_method, x_value = 0, bool_value = False):
        print("\nActivation Function: " + self.activation_name)
        print("-----------------------------------------------")
        print('Random starting synaptic weights: ')
        print(self.synaptic_weights)
        for i in range(100000):
            input_layer = self.training_inputs

            bool_value = False
            outputs = activation_method(np.dot(input_layer, self.synaptic_weights), bool_value)

            error = self.training_outputs - outputs

            bool_value = True
            adjustments = error * activation_method(outputs, bool_value)

            self.synaptic_weights = self.synaptic_weights + np.dot(input_layer.T, adjustments)

        print('\nSynaptic weights after training\n')
        print(self.synaptic_weights)

        print('\nOutputs after training')
        print(outputs)
        print('')
        print('')

    def main(self):
        if self.activation_name == "sigmoid":
            self.learn(self.sigmoid)
        elif self.activation_name == "tanh":
            self.learn(self.tanh)
        elif self.activation_name == "relu":
            self.learn(self.relu)
        else:
            print(self.activation_name)

t_in = np.array([[0,0,1],
                 [1,1,1],
                 [1,0,1],
                 [0,1,1]])

t_out = np.array([[0,1,1,0]]).T

sig_net = NN("sigmoid", t_in, t_out)
sig_net.main()

tan_net = NN("tanh", t_in, t_out)
tan_net.main()

relu_net = NN("relu", t_in, t_out)
relu_net.main()