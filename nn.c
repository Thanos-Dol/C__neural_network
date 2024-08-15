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

Network* network_init(unsigned given_number_of_layers, 
                        unsigned* number_of_neurons_per_layer_plus_initial, 
                        double (*given_activation_function)(double), 
                        double (*given_activation_function_derivative)(double), 
                        double given_learning_rate) {

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
                sum += tmp_y_in[k - 1] * net -> layers[i] -> neurons[j] -> weights[k];
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

    P_new = (double*) calloc(net -> layers[0] -> dim, sizeof(double));

    for (int i = 0; i < net -> layers[0] -> dim; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < net -> layers[1] -> dim; ++j)
        {
            sum += P_old[j] * net -> layers[1] -> neurons[j] -> weights[i + 1];
        }
        P_new[i] = sum * net -> activation_function_derivative(V[0][i]);
    }

    free(V[0]);
    free(V);
    free(P_old);

    for (int i = 0, w = 0; i < net -> layers[1] -> dim; ++i)
    {
        for (int j = 0; j < net -> layers[1] -> neurons[0] -> dim; ++j)
        {
            net -> layers[1] -> neurons[i] -> weights[j] = new_weights[w++];
        }
    }

    free(new_weights);

    for (int i = 0, w = 0; i < net -> layers[0] -> dim; ++i)
    {
        net -> layers[0] -> neurons[i] -> weights[0] = net -> layers[0] -> neurons[i] -> weights[0] - net -> learning_rate * P_new[i];
        for (int k = 1; k < net -> layers[0] -> neurons[0] -> dim; ++k)
        {
            net -> layers[0] -> neurons[i] -> weights[k] = net -> layers[0] -> neurons[i] -> weights[k] - net -> learning_rate * P_new[i] * datapoint[k - 1];
        }
    }

    free(P_new);

    return;
}

void network_backpropagation_batch_update(Network* net, unsigned batch_size, double** data, double** expected, unsigned datapoint_size) {

    double*** V;  // V [i] datapoint [j] layer [k] value / neuron
    double** network_outputs;  //  TODO : I could ditch this altogether and just use V, decide on that
    
    V = (double***) calloc(batch_size, sizeof(double**));
    for (int i = 0; i < batch_size; ++i)
    {
        V[i] = (double**) calloc(net -> num_of_layers, sizeof(double*));
    }

    network_outputs = (double**) calloc(batch_size, sizeof(double*));

    /* ----- forward phase ----- */

    for (int d = 0; d < batch_size; ++d)
    {
        double* tmp_y_in;
        double* tmp_y_out;

        tmp_y_in = (double*) calloc(datapoint_size, sizeof(double));

        for (int i = 0; i < datapoint_size; ++i)
        {
            tmp_y_in[i] = data[d][i];
        }

        for (int l = 0; l < net -> num_of_layers; ++l)
        {
            tmp_y_out = (double*) calloc(net -> layers[l] -> dim, sizeof(double));
            V[d][l] = (double*) calloc(net -> layers[l] -> dim, sizeof(double));

            for (int n = 0; n < net -> layers[l] -> dim; ++n)
            {
                double sum = net -> layers[l] -> neurons[n] -> weights[0];
                for (int w = 1; w < net -> layers[l] -> neurons[n] -> dim; ++w)
                {
                    sum += tmp_y_in[w - 1] * net -> layers[l] -> neurons[n] -> weights[w];
                }
                V[d][l][n] = sum;
                tmp_y_out[n] = net -> activation_function(sum);
            }

            free(tmp_y_in);
            tmp_y_in = (double*) calloc(net -> layers[l] -> dim, sizeof(double));
            for (int i = 0; i < net -> layers[l] -> dim; ++i)
            {
                tmp_y_in[i] = tmp_y_out[i];
            }
            free(tmp_y_out);
        }

        network_outputs[d] = tmp_y_in;
    }

    /* ----- helldive ----- */

    double** P_old;
    double** P_new;
    double* new_weights;

    new_weights = (double*) calloc(net -> layers[net -> num_of_layers - 1] -> neurons[0] -> dim * net -> layers[net -> num_of_layers - 1] -> dim, sizeof(double));

    P_old = (double**) calloc(batch_size, sizeof(double*));
    for (int i = 0; i < batch_size; ++i)
    {
        P_old[i] = (double*) calloc(net -> layers[net -> num_of_layers - 1] -> dim, sizeof(double));
    }

    for (int n = 0, nw = 0; n < net -> layers[net -> num_of_layers - 1] -> dim; ++n)
    {
        double sum = 0.0;

        // updating bias weight of neuron n
        for (int d = 0; d < batch_size; ++d)
        {
            sum += (expected[d][n] - network_outputs[d][n]) * net -> activation_function_derivative(V[d][net -> num_of_layers - 1][n]);
        }
        sum *= -(1.0 / (double)batch_size);
        new_weights[nw++] = net -> layers[net -> num_of_layers - 1] -> neurons[n] -> weights[0] - net -> learning_rate * sum;

        // updating no bias weights of neuron n
        for (int w = 1; w < net -> layers[net -> num_of_layers - 1] -> neurons[0] -> dim; ++w)
        {
            sum = 0.0;

            for (int d = 0; d < batch_size; ++d)
            {
                sum += (expected[d][n] - network_outputs[d][n]) * net -> activation_function_derivative(V[d][net -> num_of_layers - 1][n]) * net -> activation_function(V[d][net -> num_of_layers - 2][w - 1]);
            }
            sum *= -(1.0 / (double)batch_size);
            new_weights[nw++] = net -> layers[net -> num_of_layers - 1] -> neurons[n] -> weights[w] - net -> learning_rate * sum;
        }

        // creating sums to be carried to next layer backwards
        for (int d = 0; d < batch_size; ++d)
        {
            P_old[d][n] = net -> activation_function_derivative(V[d][net -> num_of_layers - 1][n]) * (expected[d][n] - net -> activation_function(V[d][net -> num_of_layers - 1][n]));
        }
    }

    for (int d = 0; d < batch_size; ++d)
    {
        free(V[d][net -> num_of_layers - 1]);
    }

    for (int d = 0; d < batch_size; ++d)
    {
        free(network_outputs[d]);
    }
    free(network_outputs);

    for (int l = net -> num_of_layers - 2; l > 0; --l)
    {
        P_new = (double**) calloc(batch_size, sizeof(double*));
        for (int d = 0; d < batch_size; ++d)
        {
            P_new[d] = (double*) calloc(net -> layers[l] -> dim, sizeof(double));
        }

        for (int d = 0; d < batch_size; ++d)
        {
            for (int n = 0; n < net -> layers[l] -> dim; ++n)
            {
                double sum = 0.0;
                for (int z = 0; z < net -> layers[l + 1] -> dim; ++z)
                {
                    sum += net -> layers[l + 1] -> neurons[z] -> weights[n + 1] * P_old[d][z];
                }
                P_new[d][n] = net -> activation_function_derivative(V[d][l][n]) * sum;
            }
        }

        for (int d = 0; d < batch_size; ++d)
        {
            free(P_old[d]);
        }
        free(P_old);

        for (int d = 0; d < batch_size; ++d)
        {
            free(V[d][l]);
        }

        // copying new weights previously calculated to neural networks corresponding weights as from now on those weights no longer need to be used in the back propagation process
        for (int n = 0, nw = 0; n < net -> layers[l + 1] -> dim; ++n)
        {
            for (int w = 0; w < net -> layers[l + 1] -> neurons[0] -> dim; ++w)
            {
                net -> layers[l + 1] -> neurons[n] -> weights[w] = new_weights[nw++];
            }
        }

        free(new_weights);
        new_weights = (double*) calloc(net -> layers[l] -> neurons[0] -> dim * net -> layers[l] -> dim, sizeof(double));

        for (int n = 0, nw = 0; n < net -> layers[l] -> dim; ++n)
        {
            double sum;
            
            sum = 0.0;
            for (int d = 0; d < batch_size; ++d)
            {
                sum += P_new[d][n];
            }
            sum *= -(1.0 / (double)batch_size);
            new_weights[nw++] = net -> layers[l] -> neurons[n] -> weights[0] - net -> learning_rate * sum;

            for (int w = 1; w < net -> layers[l] -> neurons[0] -> dim; ++w)
            {
                sum = 0.0;
                for (int d = 0; d < batch_size; ++d)
                {
                    sum += net -> activation_function(V[d][l - 1][w - 1]) * P_new[d][n];
                }
                sum *= -(1.0 / (double)batch_size);
                new_weights[nw++] = net -> layers[l] -> neurons[n] -> weights[w] - net -> learning_rate * sum;
            }
        }

        P_old = P_new;
    }

    P_new = (double**) calloc(batch_size, sizeof(double*));
    for (int d = 0; d < batch_size; ++d)
    {
        P_new[d] = (double*) calloc(net -> layers[0] -> dim, sizeof(double));
    }

    for (int d = 0; d < batch_size; ++d)
    {
        for (int n = 0; n < net -> layers[0] -> dim; ++n)
        {
            double sum = 0.0;
            for (int z = 0; z < net -> layers[1] -> dim; ++z)
            {
                sum += net -> layers[1] -> neurons[z] -> weights[n + 1] * P_old[d][z];
            }
            P_new[d][n] = net -> activation_function_derivative(V[d][0][n]) * sum;
        }
    }

    for (int d = 0; d < batch_size; ++d)
    {
        free(V[d][0]);
        free(V[d]);
    }
    free(V);

    for (int d = 0; d < batch_size; ++d)
    {
        free(P_old[d]);
    }
    free(P_old);

    for (int n = 0, nw = 0; n < net -> layers[1] -> dim; ++n)
    {
        for (int w = 0; w < net -> layers[1] -> neurons[0] -> dim; ++w)
        {
            net -> layers[1] -> neurons[n] -> weights[w] = new_weights[nw++];
        }
    }

    free(new_weights);

    for (int n = 0, nw = 0; n < net -> layers[0] -> dim; ++n)
    {
        double sum;
        
        sum = 0.0;
        for (int d = 0; d < batch_size; ++d)
        {
            sum += P_new[d][n];
        }
        sum *= -(1.0 / (double)batch_size);
        net -> layers[0] -> neurons[n] -> weights[0] = net -> layers[0] -> neurons[n] -> weights[0] - net -> learning_rate * sum;

        for (int w = 1; w < net -> layers[0] -> neurons[0] -> dim; ++w)
        {
            sum = 0.0;
            for (int d = 0; d < batch_size; ++d)
            {
                sum += data[d][w - 1] * P_new[d][n];
            }
            sum *= -(1.0 / (double)batch_size);
            net -> layers[0] -> neurons[n] -> weights[w] = net -> layers[0] -> neurons[n] -> weights[w] - net -> learning_rate * sum;
        }
    }

    for (int d = 0; d < batch_size; ++d)
    {
        free(P_new[d]);
    }
    free(P_new);

    return;
}

void network_backpropagation_batch_train(Network* net, unsigned batch_size, double** data, double** expected, unsigned dataset_size, unsigned datapoint_size) {

    for (int i = 0; 1; ++i)
    {
        unsigned upper_bound;  // TODO : rename this
        unsigned training_batch_size;

        upper_bound = (i + 1) * batch_size - 1;
        training_batch_size = batch_size;
        if (upper_bound >= dataset_size)
        {
            upper_bound = dataset_size - 1;
            training_batch_size = upper_bound - (i * batch_size) + 1;
        }

        double** train_batch;
        double** train_batch_expected_values;
        
        train_batch = (double**) calloc(training_batch_size, sizeof(double*));
        train_batch_expected_values = (double**) calloc(training_batch_size, sizeof(double*));

        for (int j = 0; j < training_batch_size; ++j)
        {
            train_batch[j] = data[i * batch_size + j];
            train_batch_expected_values[j] = expected[i * batch_size + j];
        }

        network_backpropagation_batch_update(net, training_batch_size, train_batch, train_batch_expected_values, datapoint_size);

        free(train_batch);
        free(train_batch_expected_values);

        if (upper_bound == dataset_size - 1)
        {
            return;
        }
    }
}

void print_network(Network* net) {

    printf("--------- Printing weights of Network ---------\n\n");
    for (int i = 0; i < net -> num_of_layers; ++i)
    {
        printf("--- Layer %d ---\n\n", i + 1);
        for (int j = 0; j < net -> layers[i] -> dim; ++j)
        {
            printf("- Neuron %d:\n\n", j + 1);
            for (int k = 0; k < net -> layers[i] -> neurons[0] -> dim; ++k)
            {
                printf("weight %d: %f\n", k, net -> layers[i] -> neurons[j] -> weights[k]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("--------------- end ---------------\n\n");
}