from value import Value

def show(v):
    assert type(v.data) == float
    assert type(v.grad) == float
    assert type(v._prev) == set
    assert type(v._op) == str
    print("-" * 40)
    print(f"{v.data=}", f"{type(v.data)=}")
    print(f"{v.grad=}", f"{type(v.grad)=}")
    print(f"{v._prev=}", f"{type(v._prev)=}")
    print(f"{v._op=}", f"{type(v._op)=}")

v = Value(42)
assert v._op == ""
show(v)

v = Value(42, _op="Operation")
assert v._op == "Operation"
assert v.data == 42
show(v)

v.data = 100
v.data = 100.0
v.grad = 42
v.grad = 42.0
assert v.data == 100
assert v.data == 100.0
assert v.grad == 42
assert v.grad == 42.0
show(v)
