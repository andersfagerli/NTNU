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
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

int incrementByValueNumTimes(int startValue, int increment, int numTimes);
void incrementByValueNumTimesRef(int& startValue, int increment, int numTimes);
void swap(int& a, int& b);
void randomizeVector(vector<int>& vec, int n);
void sortVector(vector<int>& vec);
double medianOfVector(vector<int> vec);
string randomizeString(int length, char lower, char upper);
string readInputToString(int length, char lower, char upper);
int countChar(string s, char c);

#endif /* utilities_hpp */
