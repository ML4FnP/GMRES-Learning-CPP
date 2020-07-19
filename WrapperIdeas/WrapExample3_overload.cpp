#include <iostream>
#include <functional>


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






template <class> struct Wrapper;
template <class R, class... Args>
struct Wrapper<R(Args...)>
{
    Wrapper(std::function<R(Args ...)> f) : f_(f) {}
    R operator()(Args ... args)
    {
        std::cout << "BEGIN decorating...\n"; 
        f_(args...);
        std::cout << "END decorating\n";
        return;
    }
    std::function<R(Args ...)> f_;
};


template<class R, class... Args>
Wrapper<R(Args...)> makeWrapper(R (*f)(Args ...))
{
   return Wrapper<R(Args...)>(std::function<R(Args...)>(f));}






int main()
{
    double a;

    void (*FuncPtr1)(double*,double,int,int,int,int)=&dbleSumFuncCall;
    auto dbleSumFuncCall_Wrapped=makeWrapper(FuncPtr1);


    dbleSumFuncCall_Wrapped(&a,1.0,2,3,4,5);
    std::cout << "Output:   " << a << std::endl;

}


