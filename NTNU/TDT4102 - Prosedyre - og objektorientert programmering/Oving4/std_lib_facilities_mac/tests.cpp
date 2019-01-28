//
//  tests.cpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 24/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#include "tests.hpp"

void testCallByValue() {
    int v0 = 5;
    int increment = 2;
    int iterations = 10;
    int result = incrementByValueNumTimes(v0, increment, iterations);
    std::cout << "v0: " << v0
    << " increment: " << increment
    << " iterations: " << iterations
    << " result: " << result << std::endl;
}

void testCallByReference() {
    int v0 = 5;
    int increment = 2;
    int iterations = 10;
    incrementByValueNumTimesRef(v0, increment, iterations);
    std::cout << "v0: " << v0
    << " increment: " << increment
    << " iterations: " << iterations
    << " result: " << v0 << std::endl;
}
