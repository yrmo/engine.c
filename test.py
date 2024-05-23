from value import Value

def show(v):
    assert type(v.data) == float
    assert type(v.grad) == float
    print(f"{v.data=}", f"{type(v.data)=}")

v = Value(42)
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
