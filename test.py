from engine import Value

def show(v):
    assert type(v.data) == float
    assert type(v.grad) == float
    print(v.backward())
    assert v.backward() == None
    assert type(v._prev) == set
    assert type(v._op) == str
    print(f"{v.data=}", f"{type(v.data)=}")
    print(f"{v.grad=}", f"{type(v.grad)=}")
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

def test_backward_add():
    a = Value(1)
    b = Value(1)
    c = a + b
    c.backward()
    assert a.grad == 1
    assert b.grad == 1


test_backward_add()

def test_backward_add_neg():
    a = Value(1)
    b = Value(-1)
    c = a + b
    c.backward()
    assert a.grad == 1
    assert b.grad == 1

test_backward_add_neg()

def test_backward_radd():
    a = 1
    b = Value(1)
    c = a + b
    c.backward()
    assert b.grad == 1

test_backward_add()

def test_backward_add_twice():
    a = Value(1.0)
    c = a + a
    c.backward()
    assert a.grad == 2

test_backward_add_twice()

def test_backward_sub():
    a = Value(1.0)
    b = Value(2)
    c = a - b
    c.backward()
    assert a.grad == 1
    assert b.grad == -1

test_backward_sub()

def test_backward_rsub():
    a = 1.0
    b = Value(2)
    c = a - b
    c.backward()
    assert b.grad == -1

test_backward_rsub()

def test_backward_mul():
    a = Value(1)
    b = Value(2.0)
    c = a * b
    c.backward()
    assert a.grad == 2
    assert b.grad == 1

test_backward_mul()

def test_backward_rmul():
    a = 2.0
    b = Value(1.0)
    c = a * b
    c.backward()
    assert b.grad == 2.0

test_backward_rmul()

def test_backward_div():
    a = Value(1)
    b = Value(2.0)
    c = a / b
    c.backward()
    assert a.grad == 0.5
    assert b.grad == -0.25

test_backward_div()

def test_backward_rdiv():
    a = 1
    b = Value(2.0)
    c = a / b
    c.backward()
    assert b.grad == -0.25

test_backward_rdiv()

def test_backward_neg():
    a = Value(2.0)
    b = -a
    b.backward()
    assert a.grad == -1.0

test_backward_neg()
