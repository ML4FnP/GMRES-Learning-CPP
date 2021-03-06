#include <iostream>
#include <array>

#include "arg_pack.h"


template<typename T, T ... ints>
void print_sequence(std::integer_sequence<T, ints...> int_seq) {
    std::cout << "The sequence of size " << int_seq.size() << ": ";
    ((std::cout << ints << ' '),...);
    std::cout << '\n';
}



int f(int i, int j, int k) {
    return i + j*i + k*j*i;
}



template<typename T, size_t N>
T g(std::array<T, N> a) {
    T r = a[0];
    for (int i=1; i<N; ++i) r += a[i];
    return r;
}



int h(int i, int j) {
    return i+j;
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
        auto ap_caller = decltype(ap)::caller<>(ap);
        int z = ap_caller(f);

        std::cout << "f(i,j,k) = " << z << std::endl;
    }

    {
        std::array<int, 3> a{1, 2, 3};
        auto ap = make_arg_pack(a);
        auto ap_caller = decltype(ap)::caller<>(ap);

        for (int i=0; i<3; ++i)
            std::cout << ap.get<0>()[i] << " ";
        std::cout << std::endl;

        std::cout << "g(a) = " << ap_caller(g<int, 3>) << std::endl;
    }

    {
        print_sequence(make_index_sequence<10, 20>());

        enum args {i, j, k};
        auto ap = make_arg_pack(0, 1, 2, 3);
        auto ap_caller = decltype(ap)::caller<1, 3>(ap);
        int z = ap_caller(h);

        std::cout << "h(i,j) = " << z << std::endl;
    }

}
