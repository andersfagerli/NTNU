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

void testVectorSorting() {
    vector<int> percentages;
    randomizeVector(percentages, 10);
    
    std::cout << "Before swap:\t";
    for (int i = 0; i < 10; i++) {
        std::cout << percentages[i] << " ";
    }
    std::cout << std::endl;
    
    swap(percentages[0], percentages[1]);
    
    std::cout << "After swap:\t\t";
    for (int i = 0; i < 10; i++) {
        std::cout << percentages[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Median:\t" << medianOfVector(percentages) << std::endl;
    
    sortVector(percentages);
    std::cout << "After sorting:\t";
    for (int i = 0; i < 10; i++) {
        std::cout << percentages[i] << " ";
    }
    std::cout << "\n";
    
    std::cout << "Median:\t" << medianOfVector(percentages) << std::endl;
    std::cout << std::endl;
}

void testString() {
    int numGrades = 8;
    string grades = randomizeString(numGrades, 'A', 'F');
    std::cout << grades << std::endl;
    vector<int> gradeCount(6);
    for (char ch = 'A'; ch <= 'F'; ch++) {
        gradeCount[ch-'A'] = countChar(grades, ch);
        std::cout << gradeCount[ch-'A'] << " ";
    }
    std::cout << std::endl;
    double sum = 0;
    int weight = 5;
    for (int grade : gradeCount) {
        sum += grade*(weight--);
    }
    std::cout << "Mean:\t" << sum/numGrades << "\t\t";
}
