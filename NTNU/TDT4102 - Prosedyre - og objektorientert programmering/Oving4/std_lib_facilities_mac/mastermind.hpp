//
//  mastermind.hpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 29/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#ifndef mastermind_hpp
#define mastermind_hpp

#include "utilities.hpp"
#include "masterVisual.hpp"

void playMastermind();
int checkCharactersAndPosition(string guess, string code);
int checkCharacters(string guess, string code);
#endif /* mastermind_hpp */
