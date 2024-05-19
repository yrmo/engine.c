from value import Value

v = Value(42)
print(f"{v.value=}")
assert v.value == 42

v.value = 100
print(f"{v.value=}")
assert v.value == 100
