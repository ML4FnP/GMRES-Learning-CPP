#ifndef __ARG_PACK_H
#define __ARG_PACK_H

#include <tuple>



template<typename ... Args>
class arg_pack {

public:
    arg_pack(Args ... args) {
        m_args = std::make_tuple(args ...);
    }


    template<std::size_t I>
    auto get() {
        return std::get<I>(m_args);
    }


    template<typename Function>
    auto apply(Function f) {
        return call(
            f, m_args,
            std::make_index_sequence<
                std::tuple_size<
                    std::tuple<Args ...>
                >::value
            >{}
        );
    }

private:
    std::tuple<Args ...> m_args;


    template<typename Function, typename Tuple, size_t ... I>
    static auto call(Function f, Tuple t, std::index_sequence<I ...>){
        return f(std::get<I>(t) ...);
    }
};



template<typename... Args>
auto make_arg_pack(Args && ... args){
    return arg_pack<Args ...>(args ...);
}

#endif
