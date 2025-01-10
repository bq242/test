#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 定义网络结构
#define INPUT_NEURONS 2
#define HIDDEN_NEURONS 4
#define OUTPUT_NEURONS 1
#define LEARNING_RATE 0.1
#define MAX_EPOCHS 10000

// 激活函数（sigmoid）
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// sigmoid函数的导数
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// 随机生成权重
double random_weight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

// 神经网络结构
typedef struct {
    // 权重
    double hidden_weights[INPUT_NEURONS][HIDDEN_NEURONS];
    double output_weights[HIDDEN_NEURONS][OUTPUT_NEURONS];
    
    // 偏置
    double hidden_bias[HIDDEN_NEURONS];
    double output_bias[OUTPUT_NEURONS];
    
    // 神经元输出
    double hidden_layer[HIDDEN_NEURONS];
    double output_layer[OUTPUT_NEURONS];
} NeuralNetwork;

// 初始化神经网络
void initialize_network(NeuralNetwork* network) {
    // 初始化隐藏层权重和偏置
    for(int i = 0; i < INPUT_NEURONS; i++) {
        for(int j = 0; j < HIDDEN_NEURONS; j++) {
            network->hidden_weights[i][j] = random_weight();
        }
    }
    
    // 初始化输出层权重和偏置
    for(int i = 0; i < HIDDEN_NEURONS; i++) {
        network->hidden_bias[i] = random_weight();
        for(int j = 0; j < OUTPUT_NEURONS; j++) {
            network->output_weights[i][j] = random_weight();
        }
    }
    
    for(int i = 0; i < OUTPUT_NEURONS; i++) {
        network->output_bias[i] = random_weight();
    }
}

// 前向传播
void forward_propagate(NeuralNetwork* network, double input[INPUT_NEURONS]) {
    // 计算隐藏层
    for(int i = 0; i < HIDDEN_NEURONS; i++) {
        double activation = network->hidden_bias[i];
        for(int j = 0; j < INPUT_NEURONS; j++) {
            activation += input[j] * network->hidden_weights[j][i];
        }
        network->hidden_layer[i] = sigmoid(activation);
    }
    
    // 计算输出层
    for(int i = 0; i < OUTPUT_NEURONS; i++) {
        double activation = network->output_bias[i];
        for(int j = 0; j < HIDDEN_NEURONS; j++) {
            activation += network->hidden_layer[j] * network->output_weights[j][i];
        }
        network->output_layer[i] = sigmoid(activation);
    }
}

// 反向传播
void back_propagate(NeuralNetwork* network, double input[INPUT_NEURONS], double target[OUTPUT_NEURONS]) {
    double output_delta[OUTPUT_NEURONS];
    double hidden_delta[HIDDEN_NEURONS];
    
    // 计算输出层的误差
    for(int i = 0; i < OUTPUT_NEURONS; i++) {
        double error = target[i] - network->output_layer[i];
        output_delta[i] = error * sigmoid_derivative(network->output_layer[i]);
    }
    
    // 计算隐藏层的误差
    for(int i = 0; i < HIDDEN_NEURONS; i++) {
        double error = 0.0;
        for(int j = 0; j < OUTPUT_NEURONS; j++) {
            error += output_delta[j] * network->output_weights[i][j];
        }
        hidden_delta[i] = error * sigmoid_derivative(network->hidden_layer[i]);
    }
    
    // 更新输出层权重和偏置
    for(int i = 0; i < HIDDEN_NEURONS; i++) {
        for(int j = 0; j < OUTPUT_NEURONS; j++) {
            network->output_weights[i][j] += LEARNING_RATE * output_delta[j] * network->hidden_layer[i];
        }
    }
    
    for(int i = 0; i < OUTPUT_NEURONS; i++) {
        network->output_bias[i] += LEARNING_RATE * output_delta[i];
    }
    
    // 更新隐藏层权重和偏置
    for(int i = 0; i < INPUT_NEURONS; i++) {
        for(int j = 0; j < HIDDEN_NEURONS; j++) {
            network->hidden_weights[i][j] += LEARNING_RATE * hidden_delta[j] * input[i];
        }
    }
    
    for(int i = 0; i < HIDDEN_NEURONS; i++) {
        network->hidden_bias[i] += LEARNING_RATE * hidden_delta[i];
    }
}

// 训练神经网络
void train_network(NeuralNetwork* network, double training_inputs[][INPUT_NEURONS], 
                  double training_outputs[][OUTPUT_NEURONS], int num_samples) {
    for(int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        double total_error = 0.0;
        
        for(int i = 0; i < num_samples; i++) {
            forward_propagate(network, training_inputs[i]);
            back_propagate(network, training_inputs[i], training_outputs[i]);
            
            // 计算误差
            for(int j = 0; j < OUTPUT_NEURONS; j++) {
                double error = training_outputs[i][j] - network->output_layer[j];
                total_error += error * error;
            }
        }
        
        // 每1000轮打印一次误差
        if(epoch % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", epoch, total_error);
        }
    }
}

int main() {
    srand(time(NULL));
    
    // 创建神经网络
    NeuralNetwork network;
    initialize_network(&network);
    
    // 准备训练数据（XOR问题）
    double training_inputs[4][INPUT_NEURONS] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double training_outputs[4][OUTPUT_NEURONS] = {{0}, {1}, {1}, {0}};
    
    // 训练网络
    printf("Training Neural Network...\n");
    train_network(&network, training_inputs, training_outputs, 4);
    
    // 测试网络
    printf("\nTesting Neural Network:\n");
    for(int i = 0; i < 4; i++) {
        forward_propagate(&network, training_inputs[i]);
        printf("Input: %.0f %.0f -> Output: %f\n", 
               training_inputs[i][0], training_inputs[i][1], 
               network.output_layer[0]);
    }
    
    return 0;
} 