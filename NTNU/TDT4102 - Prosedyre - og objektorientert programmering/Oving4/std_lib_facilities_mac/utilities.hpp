//
//  utilities.hpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 24/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#ifndef utilities_hpp
#define utilities_hpp

#include <stdio.h>
#include <iostream>

int incrementByValueNumTimes(int startValue, int increment, int numTimes);
void incrementByValueNumTimesRef(int& startValue, int increment, int numTimes);
void swap(int& a, int& b);

#endif /* utilities_hpp */
