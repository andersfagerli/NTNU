#include "utilities.hpp"

void printArray(int array[], int size) {
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void randomizeArray(int array[], int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 101;
    }
}