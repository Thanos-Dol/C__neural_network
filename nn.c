#include <malloc.h>
#include "nn.h"

Neuron* neuron_init(unsigned given_dim) {

    Neuron* n = (Neuron*) malloc(sizeof(Neuron));

    n -> dim = given_dim;
    n -> weights = (double*) calloc(given_dim, sizeof(double));

    for (int i = 0; i < given_dim; ++i)
    {
        n -> weights[i] = 0.0;
    }

    return n;
}

Layer* layer_init(unsigned number_of_neurons, unsigned number_of_neuron_weights) {

    Layer* l = (Layer*) malloc(sizeof(Layer));

    l -> dim = number_of_neurons;
    l -> neurons = (Neuron**) calloc(number_of_neurons, sizeof(Neuron*));

    for (int i = 0; i < number_of_neurons; ++i)
    {
        l -> neurons[i] = neuron_init(number_of_neuron_weights);
    }

    return l;
}

Network* network_init(unsigned given_number_of_layers, unsigned* number_of_neurons_per_layer_plus_initial, double (*given_activation_function)(double)) {

    Network* net = (Network*) malloc(sizeof(Network));

    net -> num_of_layers = given_number_of_layers;

    net -> layers = (Layer**) calloc(given_number_of_layers, sizeof(Layer*));

    for (int i = 0; i < given_number_of_layers; ++i)
    {
        net -> layers[i] = layer_init(number_of_neurons_per_layer_plus_initial[i + 1], number_of_neurons_per_layer_plus_initial[i] + 1);
    }

    net -> activation_function = given_activation_function;

    return net;
}

double* network_forward(Network* net, double* datapoint) {

    double* tmp_y_in = (double*) calloc(net -> layers[0] -> neurons[0] -> dim - 1, sizeof(double));
    double* tmp_y_out;

    for (int i = 0; i < net -> num_of_layers; ++i)
    {
        tmp_y_out = (double*) calloc(net -> layers[i] -> dim, sizeof(double));

        for (int j = 0; j < net -> layers[i] -> dim; ++j)
        {
            double sum = net -> layers[i] -> neurons[j] -> weights[0];
            for (int k = 1; k < net -> layers[i] -> neurons[j] -> dim; ++k)
            {
                sum += tmp_y_in[k - 1] * net -> layers[i] -> neurons[j] -> weights[k];
            }
            tmp_y_out[j] = net -> activation_function(sum);
        }

        free(tmp_y_in);
        tmp_y_in = (double*) calloc(net -> layers[i] -> dim, sizeof(double));
        for (int l = 0; l < net -> layers[i] -> dim; ++l)
        {
            tmp_y_in[l] = tmp_y_out[l];
        }
        free(tmp_y_out);
    }

    return tmp_y_in;
}