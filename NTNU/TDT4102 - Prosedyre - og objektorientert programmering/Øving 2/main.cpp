#include <iostream>

void inputAndPrintInteger();
int inputInteger();
void inputIntegersAndPrintSum();
bool isOdd(int a);
void printHumanReadableTime(int seconds);
void inputIntegersUsingLoopAndPrintSum();

int main() {
    //inputAndPrintInteger();
    //int number = inputInteger();
    //std::cout << "Du skrev: " << number << std::endl;
    //inputIntegersAndPrintSum();
    //std::cout << isOdd(3) << std::endl;
    //printHumanReadableTime(10000);
    //inputIntegersUsingLoopAndPrintSum();
    return 0;
}

void inputAndPrintInteger() {
    std::cout << "Skriv inn et tall: ";
    int x;
    std::cin >> x;
    std::cout << "Du skrev: " << x << std::endl;
}

int inputInteger() {
    std::cout << "Skriv inn et tall: ";
    int x;
    std::cin >> x;
    return x;
}

void inputIntegersAndPrintSum() {
    int numberOne = inputInteger();
    int numberTwo = inputInteger();
    std::cout << "Summen av tallene: " << numberOne+numberTwo << std::endl;
}

bool isOdd(int a) {
    if ((a%2)) {
        return true;
    }
    else {
        return false;
    }
}

void printHumanReadableTime(int seconds) {
    int h = seconds/3600;
    int m = (seconds-h*3600)/60;
    int s = seconds - h*3600-m*60;
    std::cout << h << " timer, " << m << " minutter og " << s << " sekunder." << std::endl;
}

void inputIntegersUsingLoopAndPrintSum() {
    std::cout << "Hvor mange ganger skal tall summeres: ";
    int x;
    std::cin >> x;
    for (int i = 0; i < x; i++) {
        inputIntegersAndPrintSum();
    }
}