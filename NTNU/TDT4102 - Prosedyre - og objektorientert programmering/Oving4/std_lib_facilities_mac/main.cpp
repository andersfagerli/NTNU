#include "std_lib_facilities.h"
#include "tests.hpp"

void printMenu();
void menu();

int main(){
    menu();
    
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
}
