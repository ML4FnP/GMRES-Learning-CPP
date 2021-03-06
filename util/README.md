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


### Function Calls / Applying Functions to `arg_pack`

The `arg_pack` is mainly use to manage the input arguments to a function -- at
some point we want to actually do the function call. This is done using the
`apply` method, eg:
```c++
// int f(int i, int j, int k)
auto ap = make_arg_pack(1, 2, 3);
int z = ap.apply(f);
```


### Compiling the `arg_pack` Tester:

```
clang++ test_arg_pack.cpp -o test_arg_pack.ex
```
