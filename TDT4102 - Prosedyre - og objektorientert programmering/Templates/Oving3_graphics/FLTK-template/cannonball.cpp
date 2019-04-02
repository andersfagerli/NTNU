//
//  cannonball.cpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 17/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#include "cannonball.hpp"

double acclY() {
    return -9.81;
}

double velY(double initVelocityY, double time) {
    return initVelocityY + acclY()*time;
}

double posX(double initPosition, double initVelocity, double time) {
    return initPosition + initVelocity*time;
}

double posY(double initPosition, double initVelocity, double time) {
    return initPosition + initVelocity*time + acclY()*time*time/2.0;
}

void printTime(double time) {
    int hours = time/3600;
    double seconds = time - double(hours*3600);
    int minutes = seconds/60;
    seconds -= double(minutes)*60.0;
    
    std::cout << hours << " hours, " << minutes << " minutes and " << seconds << " seconds\n";
}

double flightTime(double initVelocityY) {
    return -2*initVelocityY/acclY();
}

void getUserInput(double& theta, double& absVelocity) {
    std::cout << "Angle: ";
    std::cin >> theta;
    std::cout << "Velocity: ";
    std::cin >> absVelocity;
}

double degToRad(double deg) {
    return deg*3.1415/180.0; // pi = atan(1)*4
}

double getVelocityX(double theta, double absVelocity) {
    return absVelocity*cos(degToRad(theta));
}

double getVelocityY(double theta, double absVelocity) {
    return absVelocity*sin(degToRad(theta));
}

void getVelocityVector(double theta, double absVelocity, double& velocityX, double& velocityY) {
    velocityX = getVelocityX(theta, absVelocity);
    velocityY = getVelocityY(theta, absVelocity);
}

double getDistanceTraveled(double velocityX, double velocityY) {
    return posX(0, velocityX, flightTime(velocityY));
}

double targetPractice(double distanceToTarget, double velocityX,double velocityY) {
    return distanceToTarget-getDistanceTraveled(velocityX, velocityY);
}

void playTargetPractice() {
    int theta, velocity, difference;
    int distanceToTarget = randomWithLimits(1000, 10);
    std::cout << std::string(20, ' ') << "Welcome to Target Practice\n\n";
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "You have 10 attempts to hit a target placed between 100 and 1000 meters\n\n";
    for (int i = 1; i <= 10; i++) {
        std::cout << i << ": \n\n";
        std::cout << "Distance to target: " << distanceToTarget << std::endl;
        std::cout << "Angle: ";
        std::cin >> theta;
        std::cout << "Velocity: ";
        std::cin >> velocity;
        
        difference = targetPractice(distanceToTarget, getVelocityX(theta, velocity), getVelocityY(theta, velocity));
        std::cout << "Flight time: ";
        printTime(flightTime(getVelocityY(theta, velocity)));
        
        if (abs(difference) < 5) {
            std::cout << "Congratulations, you win!\n";
            return;
        }
        else if (difference > 0) {
            std::cout << abs(difference) << " too short\n\n";
        }
        else {
            std::cout << abs(difference) << " too long\n\n";
        }
        std::cout << std::string(70, '-') << std::endl;
    }
    std::cout << "You lose!\n";
}
