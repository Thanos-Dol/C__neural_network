#ifndef RESURVIVOR__FIRST_NN_ATTEMPT
#define RESURVIVOR__FIRST_NN_ATTEMPT




typedef struct {

    unsigned dim;  // bias included
    double* weights;

} Neuron;

typedef struct {

    unsigned dim;  // number of neurons
    Neuron** neurons;

} Layer;

typedef struct {

    unsigned num_of_layers;
    Layer** layers;
    double (*activation_function) (double);
    double (*activation_function_derivative) (double);
    double learning_rate;

} Network;




Neuron* neuron_init(unsigned given_dim);

Layer* layer_init(unsigned number_of_neurons, unsigned number_of_neuron_weights);

Network* network_init(unsigned given_number_of_layers, 
                        unsigned* number_of_neurons_per_layer_plus_initial, 
                        double (*given_activation_function)(double), 
                        double (*given_activation_function_derivative)(double), 
                        double given_learning_rate
                        );

double* network_forward(Network* net, double* datapoint);

void network_backpropagation_update(Network* net, double* datapoint, double* expected);

#endif /* RESURVIVOR__FIRST_NN_ATTEMPT */