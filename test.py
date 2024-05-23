from value import Value

def show(v):
    print(f"{v.data=}", f"{type(v.data)=}")

v = Value(42)
show(v)
assert v.data == 42

v.data = 100
v.data = 100.0
show(v)
assert v.data == 100
