#include <iostream>

int calculate(int number1, int number2)
{
    int result = number1;

    for (int i = 1; i <= number2; i += 1)
    {
        result = result * number1;
    }

    return result;
};

// The following function has a failing test case:
// Expected Output: 8
// Actual Output: 16
int main()
{
    int result = calculate(2, 3);
    std::cout << result << std::endl;
}