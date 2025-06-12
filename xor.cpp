#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include<ctime>
using namespace std;

double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x){
    return x * (1 - x);
}

double relu(double x){
    if (x > 0){
        return x;
    }
    return 0;
}

double relu_derivative(double x){
    if (x > 0){
        return 1;
    }
    return 0;
}

struct dataset{
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> outputs = {0, 1, 1, 0};
};

struct NeuralNetwork{
    vector<vector<double>> weights1;
    vector<double> biases1;
    vector<double> weights2;
    double bias2;
    double lr;
};

void initialize(NeuralNetwork &nn, int inputsize, int hiddensize, double lr){
    nn.lr = lr;

    nn.weights1.resize(hiddensize, vector<double>(inputsize));

    for (int i = 0; i < hiddensize; i++){
        for (int j = 0; j < inputsize; j++){
            nn.weights1[i][j] = (rand() % 2000)/1000.0 - 1;
            // nn.weights1[i][j] = rand();
        }
    }

    nn.biases1.resize(hiddensize,0);

    nn.weights2.resize(hiddensize);
    
    for (int i = 0; i < hiddensize; i++){
        nn.weights2[i] = (rand() % 2000)/1000.0 - 1;
        // nn.weights2[i] = rand();
    }

    nn.bias2 = 0;
}

double pass(NeuralNetwork &nn, const vector<double> &input, vector<double> &hiddenactivation, vector<double> &zval){
    for (int i = 0; i < nn.weights1.size(); i++){
        double z = 0.0;
        for (int j=0; j < input.size(); j++){
            z += input[j] * nn.weights1[i][j];
        }
        z += nn.biases1[i];
        zval[i] = z;
        hiddenactivation[i] = relu(z);
    }

    double output = 0.0;
    for (int i = 0; i < nn.weights2.size(); i++){
        output += hiddenactivation[i] * nn.weights2[i];
    }
    output += nn.bias2;

    return sigmoid(output);
}

void train(NeuralNetwork &nn, const dataset data, int epochs){
    vector<double> hiddenactivation(nn.weights1.size());
    vector<double> zval(nn.weights1.size());
    for (int epoch = 1; epoch <= epochs; epoch++){
        double totalloss = 0.0;

        for (int i = 0; i < 4; i++){

            double output = pass(nn, data.inputs[i], hiddenactivation, zval);

            double error = output - data.outputs[i];
            totalloss += error * error;

            double outputgradient = 2*error;

            vector<double> hiddengradients(nn.weights1.size());
            for (int j = 0; j < nn.weights1.size(); j++){
                hiddengradients[j] = outputgradient * nn.weights2[j] * relu_derivative(zval[j]);
            }

            for (int j = 0; j < nn.weights2.size(); j++){
                nn.weights2[j] -= nn.lr * outputgradient * sigmoid_derivative(output) * hiddenactivation[j];
            }
            nn.bias2 -= nn.lr * outputgradient * sigmoid_derivative(output);

            for (int j = 0; j < nn.weights1.size(); j++){
                for (int k=0; k < data.inputs[i].size(); k++){
                    nn.weights1[j][k] -= nn.lr * hiddengradients[j] * data.inputs[i][k];
                }
                nn.biases1[j] -= nn.lr * hiddengradients[j];
            }

        }
        if (epoch % 1000 == 0){
            cout <<"Epoch: "<<epoch<<" loss: "<<totalloss<<endl;
        }
    }
}

double predict(NeuralNetwork &nn, const vector<double> &input){
    vector<double> hiddenactivation(nn.weights1.size());
    vector<double> zval(nn.weights1.size());
    return pass(nn, input, hiddenactivation, zval);
}

int main(){
    srand(time(0));

    dataset data;
    NeuralNetwork nn;
    initialize(nn, 2, 5, 0.005);

    train(nn, data, 20000);

    for (int i=0; i < 4; i++){
        double prediction = predict(nn, data.inputs[i]);
        cout <<"Input: "<<data.inputs[i][0]<<" "<<data.inputs[i][1]<<" Prediction: "<<prediction<<", Target: "<<data.outputs[i]<<endl;
    }
}