#include <iostream>

#include "wrapper.h"


void q1(int x) {
    std::cout << "called q1 with x=" << x << std::endl;
}



double q2(double x, double y) {
    return (x+y)/2.;
}



template<typename Signature>
class ReturnWrapper;



template<typename F, typename ... Args>
class ReturnWrapper<F(Args ...)> : protected Wrapper<F(Args ...)>{
public:

    template<typename Function>
    ReturnWrapper(Function func) : Wrapper<F(Args ...)>(func) {}

    F operator()(Args && ... args) {
        arg_pack<Args ...> ap = make_arg_pack(std::move(args) ...);
        // wrapper code running before the wrapped function
        auto ret = this->caller(ap);
        // wrapper cude running after the wrapped function
        return ret;
    }
};



template<typename Signature>
class VoidWrapper;



template<typename F, typename ... Args>
class VoidWrapper<F(Args ...)> : protected Wrapper<F(Args ...)> {
public:

    template<typename Function>
    VoidWrapper(Function func) : Wrapper<F(Args...)>(func) {}

    F operator()(Args && ... args) {
        arg_pack<Args ...> ap = make_arg_pack(std::move(args) ...);
        // wrapper code running before the wrapped function
        this->caller(ap);
        // wrapper cude running after the wrapped function
    }
};



int main(int argc, char * argv[]) {

    {
        VoidWrapper<decltype(q1)> wrapped_q1(q1);
        wrapped_q1(1);
        ReturnWrapper<decltype(q2)> wrapped_q2(q2);
        std::cout << "q2(x,y) = " << wrapped_q2(0.1, 0.3) << std::endl;
    }
}
