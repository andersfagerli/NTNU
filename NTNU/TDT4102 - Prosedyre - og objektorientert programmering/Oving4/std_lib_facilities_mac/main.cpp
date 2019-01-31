#include "std_lib_facilities.h"
#include "tests.hpp"

void printMenu();
void menu();

using namespace std;

int main(){
    srand(int(time(nullptr)));
    //menu();
    constexpr int a = 5;
    const int b = a;
    std::cout << a << std::endl;
    
    return 0;
}

void menu() {
    int choice;
    do {
        printMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 0:
                break;
            case 1:
                testCallByValue();
                break;
            case 2:
                testCallByReference();
                break;
            case 3:
                testVectorSorting();
                break;
            case 4:
                testString();
                break;
            default:
                break;
        }
    } while (choice != 0);
}

void printMenu() {
    std::cout << "Velg funksjon: \n";
    std::cout << "0: Avslutt\n";
    std::cout << "1: Test call-by-value\n";
    std::cout << "2: Test call-by-reference\n";
    std::cout << "3: Test vector sorting\n";
    std::cout << "4: Test string\n";
}
