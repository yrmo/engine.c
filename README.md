# engine.c

This is a reimplementation of Karpathy's scalar-valued autograd engine for micrograd as a Python C extension. There's about a 9x speedup overall during the training of a MLP on the two moons dataset. See the [micrograd repository](https://github.com/karpathy/micrograd) and Karpathy's [backpropagation explanation video](https://www.youtube.com/watch?v=VMj-3S1tku0) for more information. Because micrograd's engine is nicely self-contained and isolated from the Python neural network code, micrograd's engine is easily replaced with a compiled shared object file, as shown in the demo below:

## [`engine.py`](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py)

This is micrograd's original engine in pure Python, about a 100 LOC `Value` engine class object:

https://github.com/yrmo/engine.c/assets/148522719/cb88597d-71b5-422b-bbe2-42237693009e

## [`engine.c`](https://github.com/yrmo/engine.c/blob/main/engine.c)

This is the Python C extension version, the `Value` engine reimplemented in C in about 500 LOC:

https://github.com/yrmo/engine.c/assets/148522719/5876a570-0d22-4a8f-872f-a97c39a32c07

# Build & Test

1) Clone https://github.com/karpathy/micrograd
2) Build `engine.c`:
```
pip install -r requirements.txt
python setup.py build_ext --inplace
python -m pytest test.py
```
3) Replace https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py with the built shared object file (as in the above demo)

# Development

I found a good introduction to the Python C API and the reference counting semantics to be [Sam Gross's nogil talk](https://www.youtube.com/watch?v=9OOJcTp8dqE). Sam Gross also makes a good point about [using reversible debuggers](https://youtu.be/9OOJcTp8dqE?t=2518) (`rr`) which I found helpful, as well as [Valgrind](https://www.youtube.com/watch?v=2e_u2eXe7P4) (while this extension works on any OS, Valgrind is Linux-only).

# License

MIT
