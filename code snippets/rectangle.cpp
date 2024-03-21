#include <iostream>

class Rectangle
{
public:
    Rectangle(int x1, int y1, int x2, int y2)
    {
        this->x1 = x1;
        this->y1 = y1;
        this->x2 = x2;
        this->y2 = y2;
    }

    int width()
    {
        return this->x2 - this->y1;
    }

    int height()
    {
        return this->x2 - this->y2;
    }

    double area()
    {
        return this->width() * this->height();
    }

private:
    int x1;
    int x2;
    int y1;
    int y2;
};

// The following function has a failing test case:
// Expected Output: 100
// Actual Output: 0
int main()
{
    Rectangle rect1(0, 0, 10, 10);
    std::cout << rect1.area() << std::endl;
    return -1;
}
