#include <iostream>
#include <vector>

void sort(int i, std::vector<int> &unsorted);

std::vector<int> &sort(std::vector<int> &unsorted)
{
    for (int i = 1; i < unsorted.size(); i++)
    {
        sort(i, unsorted);
    }
    return unsorted;
}

void sort(int i, std::vector<int> &unsorted)
{
    for (int j = i; j > 1; j--)
    {
        int jthElement = unsorted[j];
        int jMinusOneElement = unsorted[j - 1];
        if (jthElement > jMinusOneElement)
        {
            unsorted[j - 1] = jthElement;
            unsorted[j] = jMinusOneElement;
        }
        else
        {
            break;
        }
    }
}

// The following function has a failing test case:
// Expected Output: 7, 5, 4, 3
// Actual Output: 3, 7, 5, 4
int main()
{
    std::vector<int> unsorted = {3, 7, 4, 5};
    std::vector<int> result = sort(unsorted);
    for (int i = 0; i < result.size(); i++)
    {
        std::cout << result[i] << std::endl;
    }
}