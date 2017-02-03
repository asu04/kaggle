import math
import numpy as np
from multiprocess import Pool, cpu_count

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class neuralNode:
    """Base class for node in neural network"""
    def __init__(self, weights, transform = sigmoid):
        if isinstance(weights, np.ndarray):
            if len(weights.shape) == 1:
                self.weights = weights
                self.transform = transform
        else:
            raise ValueError('Argument is not a one dimensional ndarray!')

    def output(self, inputs):
        return self.transform((inputs * self.weights).sum())

class neuralLayer:
    """Class for layer in network"""
    def __init__(self, weight_matrix, thresholded = False, transform = sigmoid):
        try:
            self.thresholded = thresholded
            self.weight_matrix = weight_matrix
            self.last_weight_update = np.zeros(weight_matrix.shape)
            self.transform = transform
        except:
            raise ValueError("Unexpected arguments")

    def output(self, inputs):
        output_values = []
        if self.thresholded:
            inputs = np.insert(inputs,0,1)
        for row in self.weight_matrix.T:
            output_values.append(neuralNode(row, transform = self.transform).output(inputs))
        output_vector = np.array(output_values)
        # Keep track of last output and input
        self.last_output = np.copy(output_vector)
        self.last_input = np.copy(inputs)
        return output_vector

    def emit_deltas(self, next_deltas):
        deltas = np.multiply(self.weight_matrix, next_deltas)
        if self.thresholded:
            deltas = np.delete(deltas,0,0)
            last_layer_output = np.delete(self.last_input, 0)
        else:
            last_layer_output = self.last_input
        return (last_layer_output)*(1-last_layer_output)*(deltas.sum(axis = 1))

class neuralNet:
    """Base class for neural net"""
    def __init__(self, n_layers, layer_list):
        if not len(layer_list) == n_layers:
            raise ValueError('Number of layers and number of matrices supplied do not match')
        try:
            self.n_layers = n_layers
            self.layers = []
            for layer in layer_list:
                self.layers.append(layer)
        except:
            raise ValueError('DEBUG_MESSAGE_HERE')

    def propagate(self, inputs):
        """Feedforward"""
        for layer in self.layers:
            inputs = layer.output(inputs)
        return inputs

    def compute_weight_update(self, data, rate):
        """Function to return weight updates based on one piece of data on current net"""
        weight_updates = []
        deltas = []
        target = data[1]
        features = data[0]
        final_output = self.propagate(features)
        error = target - final_output
        output_deltas = error*(final_output)*(1-final_output)
        last_deltas = np.copy(output_deltas)
        deltas.append(last_deltas)
        for i in reversed(range(1,self.n_layers)):
           next_deltas = self.layers[i].emit_deltas(last_deltas)
           deltas.append(next_deltas)
           last_deltas = next_deltas
        deltas.reverse()
        for i in range(self.n_layers):
            temp_weight = np.tile(self.layers[i].last_input, (self.layers[i].weight_matrix.shape[1],1)).T
            temp_weight *= (deltas[i] * rate)
            weight_updates.append(temp_weight)
        return weight_updates

    def gradient_descent(self, training_data, rate, momentum):
        """Full gradient descent"""
        weight_updates = []
        for i in range(self.n_layers):
            total_weight = np.zeros(self.layers[i].weight_matrix.shape)
            weight_updates.append(total_weight)
        for data in training_data:
            temp_weight =  self.compute_weight_update(data, rate)
            for i in range(self.n_layers):
                weight_updates[i] += temp_weight[i]
        for i in range(self.n_layers):
            weight_updates[i] += (momentum * self.layers[i].last_weight_update)
            self.layers[i].last_weight_update = np.copy(weight_updates[i])
            self.layers[i].weight_matrix += weight_updates[i]

    def mp_gradient_descent(self, training_data, rate, momentum, num_cpus = cpu_count()-1):
        """Embarassingly parallel gradient descent"""
        n_obs = len(training_data)
        weight_updates = []
        batch_updates = []
        pool = Pool(num_cpus)
        for i in range(self.n_layers):
            total_weight = np.zeros(self.layers[i].weight_matrix.shape)
            weight_updates.append(total_weight)

        zeros = np.copy(weight_updates)

        def batch_descent(training_data):
            result = np.copy(zeros)
            for data in training_data:
                temp_update = self.compute_weight_update(data, rate)
                for i in range(self.n_layers):
                    result[i] += temp_update[i]
            return result


        def addToWeights(x):
        #     for i in range(self.n_layers):
        #         weight_updates[i] += x[i]
            batch_updates.append(x)

        # Split data up into k sublists, where k = num_cpus

        k = int(math.ceil(n_obs/float(num_cpus)))
        sublist = [training_data[x:x+k] for x in xrange(0, n_obs, k)]

        for batch in sublist:
            pool.apply_async(batch_descent, (batch,),  callback = addToWeights)

        pool.close()
        pool.join()

        for update in batch_updates:
            for i in range(self.n_layers):
                weight_updates[i] += update[i]

        for i in range(self.n_layers):
            weight_updates[i] += (momentum * self.layers[i].last_weight_update)
            self.layers[i].last_weight_update = np.copy(weight_updates[i])
            self.layers[i].weight_matrix += weight_updates[i]

    def stochastic_descent(self, training_data, rate, momentum):
        """Stochastic gradient descent"""
        for data in training_data:
            deltas = []
            target = data[1]
            features = data[0]
            final_output = self.propagate(features)
            error = target - final_output
            output_deltas = error*(final_output)*(1-final_output)
            last_deltas = np.copy(output_deltas)
            deltas.append(last_deltas)
            # Now backpropgate all the output deltas to find the rest of the deltas
            for i in reversed(range(1,self.n_layers)):
                next_deltas = self.layers[i].emit_deltas(last_deltas)
                deltas.append(next_deltas)
                last_deltas = next_deltas
            deltas.reverse()
            # Now compute all necessary weight updates and store back in weight
            # matrix
            for i in range(self.n_layers):
                temp_weight = np.tile(self.layers[i].last_input, (self.layers[i].weight_matrix.shape[1],1)).T
                temp_weight *= (deltas[i] * rate)
                self.layers[i].weight_matrix += temp_weight + (momentum * self.layers[i].last_weight_update)
                self.layers[i].last_weight_update = np.copy(temp_weight)



