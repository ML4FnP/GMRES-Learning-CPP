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



template <std::size_t N0, std::size_t N, std::size_t... Is>
auto make_index_sequence_impl() {
    // only one branch is considered. The other may be ill-formed
    if constexpr (N == N0) return std::index_sequence<Is...>();   // end case
    else return make_index_sequence_impl<N0, N-1, N-1, Is...>();  // recursion
}



template <std::size_t N0, std::size_t N>
using make_index_sequence = std::decay_t<
                                decltype(make_index_sequence_impl<N0, N>())
                            >;



template<typename ... Args>
class arg_pack {

public:
    arg_pack(Args ... args) : m_args(std::forward<Args>(args)...) {}

    using types = type_list<Args ...>;

    template<std::size_t I>
    typename types::template type<I> get() {
        return std::get<I>(m_args);
    }


    template<size_t start = 0,
             size_t end   = std::tuple_size<std::tuple<Args ...>>::value>
    class caller {
    public:
        caller(arg_pack<Args ...> & ap) : m_ap(ap) {}

        template<typename Function>
        auto operator()(Function f) {
            return call(
                f, m_ap.m_args,
                make_index_sequence<start, end>{}
            );
        }

    private:
        arg_pack<Args ...> & m_ap;

        template<typename Function, typename Tuple, size_t ... I>
        static auto call(Function f, Tuple t, std::index_sequence<I ...>){
            return f(std::get<I>(t) ...);
        }
    };


private:
    std::tuple<special_decay_t<Args> ...> m_args;
};



template<typename... Args>
arg_pack<Args ...> make_arg_pack(Args && ... args){
    return arg_pack<Args ...>(args ...);
}

#endif
