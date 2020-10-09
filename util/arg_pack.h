#ifndef __ARG_PACK_H
#define __ARG_PACK_H

#include <tuple>



template<typename ... Args>
class arg_pack {

public:
    arg_pack(Args ... args) {
        m_args = std::make_tuple(args ...);
    }


    template<std::size_t i>
    auto get(){
        return std::get<i>(m_args);
    }

private:
    std::tuple<Args ...> m_args;
};



template<typename... Args>
auto make_arg_pack(Args && ... args){
    return arg_pack<Args ...>(args ...);
}

#endif
