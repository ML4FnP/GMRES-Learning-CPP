#include <iostream> 
using namespace std; 

template <typename T>
T myMax(T x, T y)
{
    return (x > y)? x: y;
}


int main()
{
    cout << myMax<int>(3,7) << std::endl; 
    cout << myMax<double>(3.0,7.0) << std::endl;
    cout << myMax<char>('g','e') << std::endl;
    return 0;
}