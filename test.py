from engine import Value

def show(v):
    assert type(v.data) == float
    assert type(v.grad) == float
    assert callable(v._backward)
    assert v._backward() == None
    assert type(v._prev) == set
    assert type(v._op) == str
    print(f"{v.data=}", f"{type(v.data)=}")
    print(f"{v.grad=}", f"{type(v.grad)=}")
    print(v._backward, v._backward())
    print(f"{v._prev=}", f"{type(v._prev)=}")
    print(f"{v._op=}", f"{type(v._op)=}")
    print("-" * 40)

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

a = Value(1)
b = Value(2)
c = Value(3, _children=(a, b))
print(c._prev)
assert c._prev == set([a, b])
show(c)

e = a + b
print(e.data, a.data, b.data)
assert e.data == 3
assert e._prev == set([a, b])
assert e._op == "+"
show(e)

f = a + 1
assert f.data == 2
f = a + 1.0
assert f.data == 2
g = 1 + a
assert g.data == 2
g = 1.0 + a
assert g.data == 2
