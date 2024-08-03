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

Network* network_init(unsigned given_number_of_layers, unsigned* number_of_neurons_per_layer_plus_initial, double (*given_activation_function)(double), double (*given_activation_function_derivative)(double), double given_learning_rate) {

    Network* net = (Network*) malloc(sizeof(Network));

    net -> num_of_layers = given_number_of_layers;

    net -> layers = (Layer**) calloc(given_number_of_layers, sizeof(Layer*));

    for (int i = 0; i < given_number_of_layers; ++i)
    {
        net -> layers[i] = layer_init(number_of_neurons_per_layer_plus_initial[i + 1], number_of_neurons_per_layer_plus_initial[i] + 1);
    }

    net -> activation_function = given_activation_function;

    net -> activation_function_derivative = given_activation_function_derivative;

    net -> learning_rate = given_learning_rate;

    return net;
}

double* network_forward(Network* net, double* datapoint) {

    double* tmp_y_in = (double*) calloc(net -> layers[0] -> neurons[0] -> dim - 1, sizeof(double));
    double* tmp_y_out;

    for (int i = 0; i < net -> layers[0] -> neurons[0] -> dim - 1; ++i)
    {
        tmp_y_in[i] = datapoint[i];
    }

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

void network_backpropagation_update(Network* net, double* datapoint, double* expected) {

    double** V = (double**) calloc(net -> num_of_layers, sizeof(double*));
    double* tmp_y_out;
    double* tmp_y_in;

    tmp_y_in = (double*) calloc(net -> layers[0] -> neurons[0] -> dim - 1, sizeof(double));

    for (int i = 0; i < net -> layers[0] -> neurons[0] -> dim - 1; ++i)
    {
        tmp_y_in[i] = datapoint[i];
    }

    for (int i = 0; i < net -> num_of_layers; ++i)
    {
        tmp_y_out = (double*) calloc(net -> layers[i] -> dim, sizeof(double));
        V[i] = (double*) calloc(net -> layers[i] -> dim, sizeof(double));

        for (int j = 0; j < net -> layers[i] -> dim; ++j)
        {
            double sum = net -> layers[i] -> neurons[j] -> weights[0];
            for (int k = 1; k < net -> layers[i] -> neurons[j] -> dim; ++k)
            {
                sum += V[i][k - 1] * net -> layers[i] -> neurons[j] -> weights[k];
            }
            V[i][j] = sum;
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

    double* P_old;
    double* P_new;
    double* new_weights;

    P_old = (double*) calloc(net -> layers[net -> num_of_layers - 1] -> dim, sizeof(double));
    tmp_y_out = (double*) calloc(net -> layers[net -> num_of_layers - 2] -> dim, sizeof(double));
    new_weights = (double*) calloc(net -> layers[net -> num_of_layers - 1] -> neurons[0] -> dim * net -> layers[net -> num_of_layers - 1] -> dim, sizeof(double));

    for (int i = 0; i < net -> layers[net -> num_of_layers - 1] -> dim; ++i)
    {
        P_old[i] = (-1) * (expected[i] - tmp_y_in[i]) * net -> activation_function_derivative(V[net -> num_of_layers - 1][i]);
    }

    free(V[net -> num_of_layers - 1]);

    free(tmp_y_in);

    for (int j = 0; j < net -> layers[net -> num_of_layers - 2] -> dim; ++j)
    {
        tmp_y_out[j] = net -> activation_function(V[net -> num_of_layers - 2][j]);
    }

    for (int i = 0, k = 0; i < net -> layers[net -> num_of_layers - 1] -> dim; ++i)
    {
        new_weights[k++] = net -> layers[net -> num_of_layers - 1] -> neurons[i] -> weights[0] - net -> learning_rate * P_old[i];
        for (int j = 1; j < net -> layers[net -> num_of_layers - 1] -> neurons[0] -> dim; ++j)
        {
            new_weights[k++] = net -> layers[net -> num_of_layers - 1] -> neurons[i] -> weights[j] - net -> learning_rate * P_old[i] * tmp_y_out[j - 1];
        }
    }

    free(tmp_y_out);

    for (int i = net -> num_of_layers - 2, j = 1; i > 0; --i, ++j)
    {
        P_new = (double*) calloc(net -> layers[i] -> dim, sizeof(double));
        tmp_y_out = (double*) calloc(net -> layers[i - 1] -> dim, sizeof(double));

        for (int j = 0; j < net -> layers[i] -> dim; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < net -> layers[i + 1] -> dim; ++k)
            {
                sum += P_old[k] * net -> layers[i + 1] -> neurons[k] -> weights[j + 1];
            }
            P_new[j] = sum * net -> activation_function_derivative(V[i][j]);
        }

        free(V[i]);  // TODO double check this
        free(P_old);

        // copying new weights previously calculated to neural networks corresponding weights as from now on those weights no longer need to be used in the back propagation process
        for (int j = 0, w = 0; j < net -> layers[i + 1] -> dim; ++j)
        {
            for (int k = 0; k < net -> layers[i + 1] -> neurons[0] -> dim; ++k)
            {
                net -> layers[i + 1] -> neurons[j] -> weights[k] = new_weights[w++];
            }
        }

        free(new_weights);
        new_weights = (double*) calloc(net -> layers[i] -> neurons[0] -> dim * net -> layers[i] -> dim, sizeof(double));

        for (int j = 0; j < net -> layers[i - 1] -> dim; ++j)
        {
            tmp_y_out[j] = net -> activation_function(V[i - 1][j]);
        }

        for (int j = 0, w = 0; j < net -> layers[i] -> dim; ++j)
        {
            new_weights[w++] = net -> layers[i] -> neurons[j] -> weights[0] - net -> learning_rate * P_new[j];
            for (int k = 1; k < net -> layers[i] -> neurons[0] -> dim; ++k)
            {
                new_weights[w++] = net -> layers[i] -> neurons[j] -> weights[k] - net -> learning_rate * P_new[j] * tmp_y_out[k - 1];
            }
        }

        free(tmp_y_out);
        P_old = P_new;
    }

    // update first layer
}