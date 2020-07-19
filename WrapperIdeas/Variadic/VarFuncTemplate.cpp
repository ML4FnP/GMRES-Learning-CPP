#include <iostream>
using namespace std; 

void print() 
{
    std::cout << "I am an empty function and I am called at last. \n" << std::endl; 
}

// Variadic function template 
template <typename T, typename... Types>
void print(T var1, Types... var2)
{
    std::cout << var1 << std::endl;
    print(var2...);
}

int main()
{
    print(1,2,3,3.14,"Pass me any number of arguments and I will print \n");
    return 0;
}