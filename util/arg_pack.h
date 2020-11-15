#ifndef __ARG_PACK_H
#define __ARG_PACK_H

#include <tuple>
#include <functional>


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
    arg_pack(Args & ... args) : m_args(std::forward<Args>(args)...) {}

    using types = type_list<Args ...>;

    template<std::size_t I>
    typename types::template type<I> & get() {
        return std::get<I>(m_args);
    }


    template<std::size_t ... I>
    auto subset(std::index_sequence<I ...>) {
        return make_arg_pack_loc(std::get<I>(m_args) ...);
    }

    auto & args() {return m_args;}

    static const std::size_t size = std::tuple_size<std::tuple<Args ...>>::value;

private:
    std::tuple<special_decay_t<Args> ...> m_args;

    template<typename... Args_loc>
    static auto make_arg_pack_loc(Args_loc & ... args){
        return arg_pack<Args_loc ...>(args ...);
    }
};



template<typename ... Args>
arg_pack<Args ...> make_arg_pack(Args && ... args){
    return arg_pack<Args ...>(args ...);
}



template<typename Signature>
class arg_pack_caller;



template<typename F, typename ... Args>
class arg_pack_caller<F(Args ...)> {
public:

    template<typename Function>
    arg_pack_caller(Function func) : m_func(func) {}

    auto operator()(arg_pack<Args ...> & ap) {
        return call(
            m_func, ap.args(),
            make_index_sequence<0, arg_pack<Args ...>::size>{}
        );
    }


private:
    std::function<F(Args ...)> m_func;

    template<typename Function, typename Tuple, std::size_t ... I>
    static auto call(Function f, Tuple & t, std::index_sequence<I ...>){
        return f(std::get<I>(t) ...);
    }
};

#endif
