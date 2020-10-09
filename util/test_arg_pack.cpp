#include <iostream>

#include "arg_pack.h"


int main(int argc, char * argv[]) {

    enum args {foo, bar, str, baz};
    auto ap = make_arg_pack(1, 2, 'a', 3);

    std::cout << ap.get<foo>() << std::endl;
    std::cout << ap.get<bar>() << std::endl;
    std::cout << ap.get<str>() << std::endl;
    std::cout << ap.get<baz>() << std::endl;

}
