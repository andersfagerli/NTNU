#include "std_lib_facilities.h"
#include "cannonball.hpp"

void testDeviation(double compareOperand, double toOperand, double maxError, std::string name);

int main() {
    srand(int(time(nullptr)));
    //playTargetPractice();
    double a = atan(1)*4;
    cout << a << endl;
    return 0;
}

void testDeviation(double compareOperand, double toOperand, double maxError, std::string name) {
    if (abs(compareOperand-toOperand) > maxError) {
        std::cout << name << " and " << toOperand << " are not equal\n";
    }
    else {
        std::cout << name << " and " << toOperand << " are equal\n";
    }
}


