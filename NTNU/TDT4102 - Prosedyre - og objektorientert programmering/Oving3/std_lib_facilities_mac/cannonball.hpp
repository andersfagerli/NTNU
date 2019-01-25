//
//  cannonball.hpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 17/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#ifndef cannonball_h
#define cannonball_h

#include "utilities.hpp"

#include <iostream>
#include <cmath>

double acclY();
double velY(double initVelocityY, double time);
double posX(double initPosition, double initVelocity, double time);
double posY(double initPosition, double initVelocity, double time);
void printTime(double time);
double flightTime(double initVelocityY);
void getUserInput(double& theta, double& absVelocity);
double degToRad(double deg);
double getVelocityX(double theta, double absVelocity);
double getVelocityY(double theta, double absVelocity);
void getVelocityVector(double theta, double absVelocity, double& velocityX, double& velocityY);
double getDistanceTraveled(double velocityX, double velocityY);
double targetPractice(double distanceToTarget, double velocityX,double velocityY);
void playTargetPractice();

#endif /* cannonball_h */

