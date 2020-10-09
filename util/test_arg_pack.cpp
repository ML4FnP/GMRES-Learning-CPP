#include <iostream>

#include "arg_pack.h"


int f(int i, int j, int k) {
    return i + j*i + k*j*i;
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

}
