# Basic Utilities


## `arg_pack` Argument Packer/Unpacker

The `arg_pack` class -- defined in `arg_pack.h` is a helper class that manages
a list of arguments for you. It is constructed using the `make_arg_pack` helper
function, eg:
```c++
enum args {foo, bar, str, baz};
auto ap = make_arg_pack(1, 2, 'a', 3);
```
If an `enum` is defined with the same order as the arguments, we can use it to
extract arguments form the argument pack by name, eg:
```c++
ap.get<foo>()
```


### Compiling the `arg_pack` Tester:

```
clang++ test_arg_pack.cpp -o test_arg_pack.ex
```
