#include <string>
#include <iostream>

class SignChecker
{
private:
    int number;

public:
    SignChecker(int currentNumber)
    {
        this->number = currentNumber;
    }

    std::string check()
    {
        std::string theSign = "";

        if (number < 0)
        {
            theSign = "negative";
        }
        else if (number >= 0)
        {
            theSign = "positive";
        }
        else
        {
            theSign = "null";
        }
        return theSign;
    }
};

// The following function has a failing test case:
// Expected Output: null
// Actual Output: positive
int main()
{
    SignChecker number1(0);
    std::cout << number1.check() << std::endl;
}
