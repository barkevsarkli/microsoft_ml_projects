#include <iostream>
#include <cmath>

#define SIZE 7
#define TEST_SIZE 0.2

void train_test_split(int *X, int *y, int size, float test_size, int *X_train, int *y_train, int *X_test, int *y_test);
void linear_regression(int *X_train, int *y_train, int train_size, float *weight, float *bias, float learning_rate, unsigned int epochs);
float mean_squared_error(int value1, int value2);
float mean_absolute_error(int value1, int value2);

int main(void)
{
    int X[SIZE] = {1500, 1800, 2400, 3000, 3500, 4000, 4500};
    int y[SIZE] = {200000, 250000, 300000, 350000, 400000, 500000, 600000};

    int train_size = SIZE * (1 - TEST_SIZE);
    int test_size = SIZE - train_size;
    
    int X_train[train_size], y_train[train_size], X_test[test_size], y_test[test_size];

    train_test_split(X, y, SIZE, TEST_SIZE, X_train, y_train, X_test, y_test);

    float weight = 0.1, bias = 0.15;
    linear_regression(X_train, y_train, train_size, &weight, &bias, 0.0000001, 1000);
    std::cout << "Weight: " << weight << std::endl;
    std::cout << "Bias: " << bias << std::endl;

    for (int i = 0; i < test_size; i++)
    {
        std::cout << "Prediction: " << (weight * X_test[i]) + bias << std::endl;
        std::cout << "Actual: " << y_test[i] << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Difference: " << abs((weight * X_test[i]) + bias - y_test[i]) << std::endl;
    }

    return 0;
}

void train_test_split(int *X, int *y, int size, float test_size, int *X_train, int *y_train, int *X_test, int *y_test)
{
    int train_size = size * (1 - test_size);
    for (int i = 0; i < train_size; i++)
    {
        X_train[i] = X[i];
        y_train[i] = y[i];
    }

    for (int i = train_size; i < size; i++)
    {
        X_test[i - train_size] = X[i];
        y_test[i - train_size] = y[i];
    }
}

void linear_regression(int *X_train, int *y_train, int train_size, float *weight, float *bias, float learning_rate, unsigned int epochs)
{
    for (unsigned int epoch = 0; epoch < epochs; epoch++)
    {
        float cost = 0;
        float weight_gradient = 0;
        float bias_gradient = 0;
        for (int i = 0; i < train_size; i++)
        {
            float prediction = (*weight * X_train[i]) + (*bias);
            float loss = mean_squared_error(prediction, y_train[i]);
            cost += loss;
            weight_gradient += (prediction - y_train[i]) * X_train[i];
            bias_gradient += (prediction - y_train[i]);
        }
        cost /= train_size;
        std::cout << "Epoch " << epoch << " - Cost: " << cost << std::endl;
        *weight -= learning_rate * weight_gradient / train_size;
        *bias -= learning_rate * bias_gradient / train_size;
    }
}

float mean_squared_error(int value1, int value2)
{
    return pow(value1 - value2, 2);
}

float mean_absolute_error(int value1, int value2)
{
    return abs(value1 - value2);
}
