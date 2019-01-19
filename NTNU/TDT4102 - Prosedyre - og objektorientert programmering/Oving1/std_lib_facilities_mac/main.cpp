#include "std_lib_facilities.h"

int maxOfTwo(int a,int b);
int fibonacci(int n);
int squareNumberSum(int n);
void triangleNumbersBelow(int n);
bool isPrime(int n);
void naivePrimeNumberSearch(int n);
int findGreatestDivisor(int n);

int main() {
    /*
    std::cout << "Oppgave a)\n";
    std::cout << maxOfTwo(5, 6) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Oppgave b)\n";
    std::cout << fibonacci(5) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Oppgave c)\n";
    std::cout << squareNumberSum(3) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Oppgave d)\n";
    triangleNumbersBelow(10);
    std::cout << std::endl;
    
    std::cout << "Oppgave e)\n";
    std::cout << isPrime(4) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Oppgave f)\n";
    naivePrimeNumberSearch(10);
    std::cout << std::endl;
    
    std::cout << "Oppgave g)\n";
    std::cout << findGreatestDivisor(9) << std::endl;
    std::cout << std::endl;*/
    
    
    return 0;
}

/*
isFib = isFibonacciNumber(n)
    a = 0;
    b = 1;
    while (b < n)
        temp = b;
        b = b + a;
        a = temp;
    end
    isFib = b == n;
end

 def isFibonacciNumber(n):
    a = 0
    b = 1
    while (b < n):
        temp = b
        b = b + a
        a = temp
    return b == n
*/

int maxOfTwo(int a, int b) {
    if (a > b){
        std::cout << "A is greater than B" << std::endl;
        return a;
    }
    else {
        std::cout << "B is greater than A" << std::endl;
        return b;
    }
}

int fibonacci(int n) {
    int a = 0;
    int b = 1;
    std::cout << "Fibonacci numbers: " << std::endl;
    for (int x = 1; x < n; x++) {
        std::cout << x << ": " << b << std::endl;
        int temp = b;
        b = a + b;
        a = temp;
    }
    std::cout << "----" << std::endl;
    return b;
}

int squareNumberSum(int n) {
    int totalSum = 0;
    for (int i = 1; i < n; i++) {
        totalSum += pow(i,2);
        std::cout << pow(i,2) << std::endl;
    }
    std::cout << totalSum << std::endl;
    return totalSum;
}

void triangleNumbersBelow(int n) {
    int acc = 1;
    int num = 2;
    std::cout << "Triangle numbers below " << n << ": " << std::endl;
    while (acc < n) {
        std::cout << acc << " ";
        acc += num;
        num++;
    }
    std::cout << std::endl;
}

bool isPrime(int n) {
    int primeness = true;
    for (int i = 2; i < (n-1); i++) {
        if ((n % i) == 0) {
            primeness = false;
        }
    }
    return primeness;
}

void naivePrimeNumberSearch(int n) {
    for (int number = 2; number < (n-1); number++) {
        if (isPrime(number) == true) {
            std::cout << number << "is a prime\n";
        }
    }
}

int findGreatestDivisor(int n) {
    int greatestDivisor = 1;
    for (int divisor = (n-1); divisor > 0; divisor--) {
        if ((n % divisor) == 0) {
            greatestDivisor = divisor;
            break;
        }
    }
    return greatestDivisor;
}

