#include <iostream>

using namespace std;

// The following function has a failing test case:
// Expected Outputs: 1, 40
// Actual Outputs: 40, 1
int main()
{
    int cents = 140;
    int dollars = cents % 100;
    int restCents = cents / 100;

    std::cout << dollars << ", " << restCents << std::endl;
}
