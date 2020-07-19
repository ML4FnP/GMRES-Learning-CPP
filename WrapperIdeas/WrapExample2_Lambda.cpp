#include <iostream>
using namespace std; 


template<typename T>
T dbleSumFuncCalc(T x)
{
    return 2*x;
}


template<typename T, typename... ARGS>
T dbleSumFuncCalc(T x,ARGS... vars )
{
    return 2*x + dbleSumFuncCalc(vars...);
}



template<typename T,typename U, typename... ARGS >
void dbleSumFuncCall(T *y ,U x,ARGS... vars )
{
    *y=dbleSumFuncCalc(x,vars...);
}


template<typename F,typename R, typename U>
auto Wrapper(F func,R param1, U param2)
{
    auto new_function = [func,param1,param2](auto... args)
    {
        std::cout << "BEGIN decorating...\n"; 
        std::cout << "stuff before wrapee call. "<< "Param1 = "<< param1  << std::endl;
        func(args...);   
        std::cout << "stuff after wrapee call.  "<< "Param2 = "<< param2  << std::endl;
        std::cout << "END decorating\n";

    };
    return new_function;
}





int main()
{
    double a;

    void (*FuncPtr1)(double*,double,int,int,int,int)=&dbleSumFuncCall;
    auto dbleSumFuncCall_Wrapped=Wrapper(FuncPtr1, "parameter 1",3.14159) ;//, "parameter 1", 3.14159);

    dbleSumFuncCall_Wrapped(&a,1.0,2,3,4,5);
    std::cout << "Output:   " << a << std::endl;

}


