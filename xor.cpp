#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include<ctime>
using namespace std;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

struct Dataset {
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> outputs = {0, 1, 1, 0};
};

struct NeuralNetwork{
    vector<vector<double>> weights1; // Weights for input to hidden layer
    vector<double> biases1; // Biases for hidden layer
    vector<double> weights2; // Weights for hidden to output layer
    double bias2; // Bias for output layer
    double learning_rate; // Learning rate
};

void initialize(NeuralNetwork &nn, int inputsize, int hiddensize, double lr) {
    nn.learning_rate = lr;

    nn.weights1.resize(hiddensize, vector<double>(inputsize));

    for (int i = 0; i < hiddensize; i++){
        for (int j = 0; j < inputsize; j++){
            nn.weights1[i][j] = (rand() % 2000) / 1000.0 - 1;
            // nn.weights1[i][j] = ((double) rand() / RAND_MAX) * 2 - 1;
        }
    }

    nn.biases1.resize(hiddensize,0);

    nn.weights2.resize(hiddensize);
    
    for (int i = 0; i < hiddensize; i++){
        nn.weights2[i] = (rand() % 2000) / 1000.0 - 1;
        // nn.weights2[i] = ((double) rand() / RAND_MAX) * 2 - 1;
    }

    nn.bias2 = 0;
}

double pass(NeuralNetwork &nn, const vector<double> &input, vector<double> &hiddenactivation, vector<double> &zValues){
    for (int i = 0; i < nn.weights1.size(); i++){
        double z = 0.0;
        for (int j=0; j < input.size(); j++){
            z += input[j] * nn.weights1[i][j];
        }
        z += nn.biases1[i];
        zValues[i] = z;
        hiddenactivation[i] = relu(z);
    }

    double output = 0.0;
    for (int i = 0; i < nn.weights2.size(); i++){
        output += hiddenactivation[i] * nn.weights2[i];
    }
    output += nn.bias2;

    return sigmoid(output);
}

void train(NeuralNetwork &nn, const Dataset data, int epochs){
    vector<double> hiddenactivation(nn.weights1.size());
    vector<double> zValues(nn.weights1.size());
    
    for (int epoch = 1; epoch <= epochs; epoch++){
        double totalLoss = 0.0;

        for (int i = 0; i < data.inputs.size(); i++){

            double output = pass(nn, data.inputs[i], hiddenactivation, zValues);

            double error = output - data.outputs[i];
            totalLoss += error * error;

            double outputGradient = 2*error;

            vector<double> hiddengradients(nn.weights1.size());
            for (int j = 0; j < nn.weights1.size(); j++){
                hiddengradients[j] = outputGradient * nn.weights2[j] * relu_derivative(zValues[j]);
            }

            for (int j = 0; j < nn.weights2.size(); j++){
                nn.weights2[j] -= nn.learning_rate * outputGradient * sigmoid_derivative(output) * hiddenactivation[j];
            }
            nn.bias2 -= nn.learning_rate * outputGradient * sigmoid_derivative(output);

            for (int j = 0; j < nn.weights1.size(); j++){
                for (int k=0; k < data.inputs[i].size(); k++){
                    nn.weights1[j][k] -= nn.learning_rate * hiddengradients[j] * data.inputs[i][k];
                }
                nn.biases1[j] -= nn.learning_rate * hiddengradients[j];
            }

        }
        if (epoch % 1000 == 0){
            cout << "Epoch: " << epoch << " Loss: " << totalLoss << endl;
        }
    }
}

double predict(NeuralNetwork &nn, const vector<double> &input){
    vector<double> hiddenactivation(nn.weights1.size());
    vector<double> zValues(nn.weights1.size());
    return pass(nn, input, hiddenactivation, zValues);
}

int main(){
    srand(time(0));
    
    Dataset data;
    NeuralNetwork nn;

    initialize(nn, 2, 5, 0.005);
    train(nn, data, 50000);

    for (int i=0; i < data.inputs.size(); i++){
        double prediction = predict(nn, data.inputs[i]);
        cout << "Input: " << data.inputs[i][0] << " " << data.inputs[i][1] << " Prediction: " << prediction<< ", Target: " << data.outputs[i] << endl;
    }
}