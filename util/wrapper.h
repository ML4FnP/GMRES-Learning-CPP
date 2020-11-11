#ifndef __WRAPPER_H
#define __WRAPPER_H

#include "arg_pack.h"


template<typename Signature>
class Wrapper;


template<typename F, typename ... Args>
class Wrapper<F(Args ...)> {
public:

    template<typename Function>
    Wrapper(Function func) : caller(arg_pack_caller<F(Args ...)>(func)) {}

protected:
    arg_pack_caller<F(Args ...)> caller;
};


#endif
