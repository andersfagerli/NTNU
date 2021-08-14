//
//  utilities.cpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 24/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#include "utilities.hpp"

int incrementByValueNumTimes(int startValue, int increment, int numTimes) {
    for (int i = 0; i < numTimes; i++) {
        startValue += increment;
    }
    return startValue;
}

void incrementByValueNumTimesRef(int& startValue, int increment, int numTimes) {
    for (int i = 0; i < numTimes; i++) {
        startValue += increment;
    }
}

void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}


void randomizeVector(vector<int>& vec, int n) {
    int randomNumber;
    for (int i = 0; i < n; i++) {
        randomNumber = std::rand() % (100+1);
        vec.push_back(randomNumber);
    }
}

void sortVector(vector<int>& vec) {
    if (vec.size() < 1) {
        std::cout << "Nothing to sort\n";
        return;
    }
    int i = 1;
    while (i < vec.size()) {
        int j = i;
        while (j > 0 && vec[j-1] > vec[j]) {
            swap(vec[j],vec[j-1]);
            j--;
        }
        i++;
    }
}

double medianOfVector(vector<int> vec) {
    int median;
    if (vec.size() < 1) {
        median = 0;
    }
    else if (vec.size() % 2 == 0) {
        median = (vec[vec.size()/2 - 1] + vec[vec.size()/2])/2.0;
    }
    else {
        median = vec[vec.size()/2 - 1];
    }
    return median;
}

string randomizeString(int length, char lower, char upper) {
    string randomString = "";
    for (int i = 0; i < length; i++) {
        randomString+=(char(std::rand() % (upper-lower+1) + lower));
    }
    return randomString;
}

string readInputToString(int length, char lower, char upper) {
    string output = "";
    char userInput;
    for (int i = 0; i < length; i++) {
        std::cout << "Input: ";
        std::cin >> userInput;
        userInput = toupper(userInput);
        
        if (userInput < lower || userInput > upper) {
            std::cout << "Not valid character\n";
            i--;
        }
        else {
            output += userInput;
        }
    }
    return output;
}

int countChar(string s, char c) {
    int count = 0;
    for (char ch : s) {
        count += (ch==c);
    }
    return count;
}
