//
//  utilities.cpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 22/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#include "utilities.hpp"

int randomWithLimits(int upper, int lower) {
    return std::rand() % (upper-lower+1) + lower;
}
