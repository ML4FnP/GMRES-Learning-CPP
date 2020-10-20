#ifndef __ARG_PACK_H
#define __ARG_PACK_H

#include <tuple>


template <class T>
struct unwrap_refwrapper {
    using type = T;
};



template <class T>
struct unwrap_refwrapper<std::reference_wrapper<T>> {
    using type = T&;
};



template <class T>
using special_decay_t = typename unwrap_refwrapper<typename std::decay<T>::type>::type;



template <class... Args>
struct type_list {
   template <std::size_t N>
   using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
};



template<typename ... Args>
class arg_pack {

public:
    arg_pack(Args ... args) : m_args(std::forward<Args>(args)...) {}

    using types = type_list<Args ...>;

    template<std::size_t I>
    typename types::template type<I> get() {
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
    //std::tuple<Args ...> m_args;
    std::tuple<special_decay_t<Args> ...> m_args;


    template<typename Function, typename Tuple, size_t ... I>
    static auto call(Function f, Tuple t, std::index_sequence<I ...>){
        return f(std::get<I>(t) ...);
    }
};



template<typename... Args>
arg_pack<Args ...> make_arg_pack(Args && ... args){
    return arg_pack<Args ...>(args ...);
}

#endif
