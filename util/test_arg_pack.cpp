#include <iostream>
#include <array>

#include "arg_pack.h"


int f(int i, int j, int k) {
    return i + j*i + k*j*i;
}

template<typename T, size_t N>
T g(std::array<T, N> a) {
    
    T r = a[0];
    for (int i=1; i<N; ++i) r += a[i];
    return r;
}


int main(int argc, char * argv[]) {

    {
        enum args {foo, bar, str, baz};
        auto ap = make_arg_pack(1, 2, 'a', 3);

        std::cout << ap.get<foo>() << std::endl;
        std::cout << ap.get<bar>() << std::endl;
        std::cout << ap.get<str>() << std::endl;
        std::cout << ap.get<baz>() << std::endl;
    }


    {
        enum args {i, j, k};
        auto ap = make_arg_pack(1, 2, 3);
        int z = ap.apply(f);

        std::cout << "f(i,j,k) = " << z << std::endl;

    }

    {
        std::array<int, 3> a{1, 2, 3};
        auto ap = make_arg_pack(a);

        for (int i=0; i<3; ++i)
            std::cout << ap.get<0>()[i] << " ";
        std::cout << std::endl;

        std::cout << "g(a) = " << ap.apply(g<int, 3>) << std::endl;
    }

}
