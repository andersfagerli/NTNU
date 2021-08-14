//
//  mastermind.cpp
//  std_lib_facilities_mac
//
//  Created by Anders Fagerli on 29/01/2019.
//  Copyright © 2019 Lars Musaus. All rights reserved.
//

#include "mastermind.hpp"

void playMastermind() {
    
    constexpr int size = 4;
    constexpr int letters = 6;
    constexpr int numTries = 10;
    constexpr int win_w = 400;
    constexpr int win_h = 600;
    string code;
    string guess;
    int round = 0;
    int correctChar, correctPos;
    char startLetter = 'A';
    code = randomizeString(size, startLetter, startLetter+letters-1);
    
    
    
    MastermindWindow mwin{Point{900, 20}, win_w, win_h, "Mastermind"};
    addGuess(mwin, code, size, startLetter, round);
    do {
        cout << ++round << " attempt:\n";
        guess = readInputToString(size, 'A', 'A' + letters-1);
        correctChar = checkCharacters(guess, code);
        correctPos = checkCharactersAndPosition(guess, code);
        
        addGuess(mwin, guess, size, startLetter, round);
        mwin.wait_for_button();
        
    } while ((correctPos < size) && (round < numTries));
    
    if (round < 10) {
        cout << "Congratulations, you won!\n";
    }
    else {
        cout << "You lose!\n\n";
    }
}

int checkCharactersAndPosition(string guess, string code) {
    int correct = 0;
    for (int i = 0; i < guess.size(); i++) {
        if (guess[i] == code[i]) {
            correct++;
        }
    }
    return correct;
}

int checkCharacters(string guess, string code) {
    int correct = 0;
    vector<int> taken(guess.size()); //Accounts for letters already taken
    for (int i = 0; i < guess.size(); i++) {
        for (int j = 0; j < guess.size(); j++) {
            if (guess[i] == code[j] && taken[j] == 0) {
                correct++;
                taken[j] = 1;
                break;
            }
        }
    }
    return correct;
}
