#include "std_lib_facilities.h"



void inputAndPrintInteger();
int inputInteger();
void inputIntegersAndPrintSum();
bool isOdd(int n);
void printHumanReadableTime(int sec);
void inputIntegersUsingLoopAndPrintSum();
void nokToEuro();
void printMenu();
void gangeTabell();
double discriminent(double a, double b, double c);
void printRealRoots(double a, double b, double c);
void solveQuadraticEquation();
vector<int> calculateSeries(int loan, int years, int percentage);
vector<int> calculateAnnuity(double loan, int years, double percentage);
void printPayments();

int main(){
    srand(int(time(nullptr)));
    /*
    int choice;
    do {
        printMenu();
        std::cin >> choice;
        
        switch (choice) {
            case 0:
                break;
            case 1:
                inputIntegersAndPrintSum();
                break;
            case 2:
                inputIntegersUsingLoopAndPrintSum();
                break;
            case 3:
                nokToEuro();
                break;
            case 4:
                gangeTabell();
                break;
            case 5:
                solveQuadraticEquation();
                break;
            case 6:
                printPayments();
                break;
            default:
                std::cout << "Ikke en gyldig verdi" << std::endl;
        }
    } while(choice != 0);*/
    
    
    
    
    
    return 0;
}

void inputAndPrintInteger() {
    int x;
    std::cout << "Skriv inn et tall: ";
    std::cin >> x;
    std::cout << "Du skrev: " << x << std::endl;
}

int inputInteger() {
    int x;
    std::cout << "Skriv inn et tall: ";
    std::cin >> x;
    return x;
}

void inputIntegersAndPrintSum() {
    int x = inputInteger();
    int y = inputInteger();
    std::cout << "Summen av tallene: " << x + y << std::endl;
}

bool isOdd(int n) {
    if (n % 2 == 0) {
        return false;
    }
    return true;
}

void printHumanReadableTime(int sec) {
    int hours = sec/3600;
    int seconds = sec - hours*3600;
    int minutes = seconds/60;
    seconds = seconds - minutes*60;
    std::cout << hours << " timer, " << minutes << " minutter og " << seconds << " sekunder" << std::endl;
}

void inputIntegersUsingLoopAndPrintSum() {
    std::cout << "Total number of integers to sum: ";
    int x;
    std::cin >> x;
    int sum = 0;
    for (int i = 0; i < x; i++) {
        sum += inputInteger();
    }
    std::cout << "Sum: " << sum << std::endl;
}

double inputDouble() {
    double x;
    std::cout << "Skriv inn et desimaltall: ";
    std::cin >> x;
    return x;
}

void nokToEuro() {
    double kurs = 9.75;
    double nok;
    do {
        std::cout << "Antall NOK: ";
        std::cin >> nok;
    } while (nok < 0);
    std::cout << fixed << setprecision(2) << nok/kurs << std::endl;
}

void printMenu() {
    std::cout << "Velg funksjon:" << std::endl;
    std::cout << "0) Avslutt\n";
    std::cout << "1) Summer to tall\n";
    std::cout << "2) Summer flere tall\n";
    std::cout << "3) Konverter NOK til Euro\n";
    std::cout << "4) Gangetabell\n";
    std::cout << "5) Løs andregradsligning\n";
    std::cout << "6) Calculate loan payments\n";
}

void gangeTabell() {
    int bredde, høyde;
    std::cout << "Bredde: ";
    std::cin >> bredde;
    std::cout << "Høyde: ";
    std::cin >> høyde;
    
    std::cout << right;
    
    for (int i = 1; i <= høyde; i++) {
        for (int j = 1; j <= bredde; j++) {
            std::cout << i*j;
            if (j < bredde) {
                std::cout << "\t";
            }
        }
        std::cout << std::endl;
    }
}

double discriminent(double a, double b, double c) {
    return b*b-4*a*c;
}

void printRealRoots(double a, double b, double c) {
    double disc = discriminent(a, b, c);
    if (disc < 0) {
        std::cout << "Ingen reelle røtter" << std::endl;
    }
    else {
        double root1 = (-b + sqrt(disc))/(2*a);
        double root2 = (-b - sqrt(disc))/(2*a);
        
        if (disc == 0) {
            std::cout << root1 << std::endl;
        }
        else {
            std::cout << "1: " << root1 << std::endl;
            std::cout << "2: " << root2 << std::endl;
        }
    }
}

void solveQuadraticEquation() {
    double a,b,c;
    std::cout << "a: ";
    std::cin >> a;
    std::cout << "b: ";
    std::cin >> b;
    std::cout << "c: ";
    std::cin >> c;
    
    printRealRoots(a, b, c);
}

vector<int> calculateSeries(int loan, int years, int percentage) {
    vector<int> seriesLoan(years);
    double remainingLoan = loan;
    for (int i = 0; i < seriesLoan.size(); i++) {
        seriesLoan[i] = loan/years + percentage*remainingLoan/100.0;
        remainingLoan -= loan/years;
    }
    return seriesLoan;
}

vector<int> calculateAnnuity(double loan, int years, double percentage) {
    vector<int> annuityLoan(years);
    int payments = loan*(percentage/100)/(1-pow((1+percentage/100),-years));
    for (int i = 0; i < annuityLoan.size(); i++) {
        annuityLoan[i] = payments;
    }
    return annuityLoan;
}

void printPayments() {
    double loan, percentage;
    int years;
    std::cout << "Loan: ";
    std::cin >> loan;
    std::cout << "Number of payments: ";
    std::cin >> years;
    std::cout << "Percentage: ";
    std::cin >> percentage;
    
    vector<int> seriesLoan = calculateSeries(loan, years, percentage);
    vector<int> annuityLoan = calculateAnnuity(loan, years, percentage);
    
    int width = 15;
    int sumSeries = 0;
    int sumAnnuity = 0;
    
    std::cout << right;
    std::cout << setw(2) << "År" << setw(width) << "Annuitet" << setw(width) << "Serie" << setw(width) << "Differanse\n";
    for (int i = 0; i < years; i++) {
        std::cout << setw(2) << i+1 << setw(width) << annuityLoan[i] << setw(width) << seriesLoan[i] << setw(width) << annuityLoan[i] - seriesLoan[i] << std::endl;
        sumSeries += seriesLoan[i];
        sumAnnuity += annuityLoan[i];
    }
    std::cout << "__________________________________________" << std::endl;
    std::cout << std::endl;
    std::cout << setw(2) << "Total" << setw(width) << sumAnnuity << setw(width) << sumSeries << setw(width) << sumAnnuity-sumSeries << std::endl;
}

