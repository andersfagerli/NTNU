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
    
    std::cout << hours << " hours, " << minutes << "minutes and " << seconds << "seconds\n";
}

