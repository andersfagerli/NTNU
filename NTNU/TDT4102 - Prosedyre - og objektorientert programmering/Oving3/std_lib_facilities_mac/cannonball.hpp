//
//  cannonball.hpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 17/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#ifndef cannonball_h
#define cannonball_h

#include <iostream>

double acclY();
double velY(double initVelocityY, double time);
double posX(double initPosition, double initVelocity, double time);
double posY(double initPosition, double initVelocity, double time);
void printTime(double time);
double flightTime(double initVelocityY);


#endif /* cannonball_h */

