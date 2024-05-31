#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct ValueObject ValueObject;
typedef void (*backward_binary_function)(ValueObject*, ValueObject*, ValueObject*);
// typedef void (*backward_unary_function)(ValueObject*, ValueObject*);

typedef struct {
    backward_binary_function func;
    ValueObject* self;
    ValueObject* other;
} BackwardBinaryClosure;

// typedef struct {
//     backward_unary_function func;
//     ValueObject* self;
// } BackwardUnaryClosure;

struct ValueObject {
    PyObject_HEAD
    double data;
    double grad;
    void* _backward;
    PyObject* _prev;
    const char* _op;
};

static PyTypeObject ValueType;

double add_op(double a, double b) { return a + b; }
double sub_op(double a, double b) { return a - b; }
double mul_op(double a, double b) { return a * b; }
double div_op(double a, double b) { return a / b; }
double pow_op(double a, double b) { return pow(a, b); }
double neg_op(double a, double b) { return -1 * a; }
double relu_op(double a, double b) {return (a > 0) ? a : 0; }

void backward_add(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += out->grad;
    other->grad += out->grad;
}

void backward_sub(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += out->grad;
    other->grad -= out->grad;
}

void backward_mul(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += other->data * out->grad;
    other->grad += self->data * out->grad;
}

void backward_div(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += out->grad / other->data;
    other->grad -= (self->data * out->grad) / (other->data * other->data);
}

void backward_pow(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += other->data * pow(self->data, other->data - 1) * out->grad;
    other->grad += pow(self->data, other->data) * log(self->data) * out->grad;
}

void backward_neg(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += -out->grad;
}

void backward_relu(ValueObject* self, ValueObject* other, ValueObject* out) {
    self->grad += ((self->data > 0.0) ? 1.0 : 0.0) * out->grad;
}

BackwardBinaryClosure* binary_closure(backward_binary_function func, ValueObject* self, ValueObject* other) {
    BackwardBinaryClosure* closure = (BackwardBinaryClosure*)malloc(sizeof(BackwardBinaryClosure));
    closure->func = func;
    Py_INCREF(self);
    closure->self = self;
    Py_INCREF(other);
    closure->other = other;
    return closure;
}

// BackwardUnaryClosure* unary_closure(backward_unary_function func, ValueObject* self) {
//     BackwardUnaryClosure* closure = (BackwardUnaryClosure*)malloc(sizeof(BackwardUnaryClosure));
//     closure->func = func;
//     Py_INCREF(self);
//     closure->self = self;
//     return closure;
// }

void free_binary_closure(BackwardBinaryClosure* closure) {
    Py_DECREF(closure->self);
    Py_DECREF(closure->other);
    free(closure);
}

// void free_unary_closure(BackwardUnaryClosure* closure) {
//     Py_DECREF(closure->self);
//     free(closure);
// }

static PyObject* _value(double data, PyObject* children, const char* op);
// typedef double (*unary_op_func)(double);
typedef double (*binary_op_func)(double, double);

// static PyObject* Value_unary_op(PyObject* self, unary_op_func op, const char* op_name, backward_unary_function backward_func) {
//     PyObject* value_self = NULL;

//     if (!PyObject_TypeCheck(self, &ValueType)) {
//         Py_RETURN_NOTIMPLEMENTED;
//     }

//     value_self = self;
//     Py_INCREF(value_self);

//     double result = op(((ValueObject*)value_self)->data);
//     PyObject* children = PyTuple_Pack(1, value_self);
//     if (children == NULL) {
//         Py_DECREF(value_self);
//         return NULL;
//     }
//     PyObject* out = _value(result, children, op_name);
//     Py_DECREF(children);
//     if (out == NULL) {
//         Py_DECREF(value_self);
//         return NULL;
//     }
//     ((ValueObject*)out)->_backward = unary_closure(backward_func, (ValueObject*)value_self);
//     Py_DECREF(value_self);
//     return out;
// }

static PyObject* Value_binary_op(PyObject* self, PyObject* other, binary_op_func op, const char* op_name, backward_binary_function backward_func) {
    PyObject* value_self = NULL;
    PyObject* value_other = NULL;

    if (PyObject_TypeCheck(self, &ValueType)) {
        value_self = self;
        Py_INCREF(value_self);
    } else if (PyFloat_Check(self) || PyLong_Check(self)) {
        value_self = _value(PyFloat_AsDouble(self), PyTuple_New(0), "");
        if (value_self == NULL) {
            return NULL;
        }
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }

    if (PyObject_TypeCheck(other, &ValueType)) {
        value_other = other;
        Py_INCREF(value_other);
    } else if (PyFloat_Check(other) || PyLong_Check(other)) {
        value_other = _value(PyFloat_AsDouble(other), PyTuple_New(0), "");
        if (value_other == NULL) {
            Py_DECREF(value_self);
            return NULL;
        }
    } else {
        Py_DECREF(value_self);
        Py_RETURN_NOTIMPLEMENTED;
    }

    double result = op(((ValueObject*)value_self)->data, ((ValueObject*)value_other)->data);
    PyObject* children = PyTuple_Pack(2, value_self, value_other);
    if (children == NULL) {
        Py_DECREF(value_self);
        Py_DECREF(value_other);
        return NULL;
    }
    PyObject* out = _value(result, children, op_name);
    Py_DECREF(children);
    if (out == NULL) {
        Py_DECREF(value_self);
        Py_DECREF(value_other);
        return NULL;
    }
    ((ValueObject*)out)->_backward = binary_closure(backward_func, (ValueObject*)value_self, (ValueObject*)value_other);
    Py_DECREF(value_self);
    Py_DECREF(value_other);
    return out;
}

static PyObject* Value_add(PyObject* self, PyObject* other) {
    return Value_binary_op(self, other, add_op, "+", backward_add);
}

static PyObject* Value_sub(PyObject* self, PyObject* other) {
    return Value_binary_op(self, other, sub_op, "-", backward_sub);
}

static PyObject* Value_mul(PyObject* self, PyObject* other) {
    return Value_binary_op(self, other, mul_op, "*", backward_mul);
}

static PyObject* Value_div(PyObject* self, PyObject* other) {
    return Value_binary_op(self, other, div_op, "/", backward_div);
}

static PyObject* Value_pow(PyObject* self, PyObject* other, PyObject* mod) {
    if (mod != Py_None) {
        PyErr_SetString(PyExc_TypeError, "Mod not supported");
        return NULL;
    }
    return Value_binary_op(self, other, pow_op, "^", backward_pow);
}

static PyObject* Value_neg(PyObject* self) {
    PyObject* dummy = _value(0.0, PyTuple_New(0), "");
    if (dummy == NULL) {
        return NULL;
    }
    PyObject* result = Value_binary_op(self, dummy, neg_op, "~", backward_neg);
    Py_DECREF(dummy);
    return result;
    // return Value_unary_op(self, neg_op, "~", backward_neg);
}

static PyObject* Value_relu(PyObject* self) {
    PyObject* dummy = _value(0.0, PyTuple_New(0), "");
    if (dummy == NULL) {
        return NULL;
    }
    PyObject* result = Value_binary_op(self, dummy, relu_op, "R", backward_relu);
    Py_DECREF(dummy);
    return result;
    // return Value_unary_op(self, relu_op, "R", backward_relu);
}

// int is_unary(const char* op) {
//     return (strcmp(op, "~") == 0 || strcmp(op, "R") == 0);
// }

static PyObject* Value_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    ValueObject *self;
    self = (ValueObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->data = 0.0;
        self->grad = 0.0;
        self->_prev = PySet_New(NULL);
        if (self->_prev == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        self->_op = NULL;
    }
    return (PyObject *)self;
}

static void Value_dealloc(ValueObject* self) {
    if (self->_backward) {
        // if (is_unary(self->_op)) {
        //     free_unary_closure((BackwardUnaryClosure*)self->_backward);
        // } else {
        free_binary_closure((BackwardBinaryClosure*)self->_backward);
        // }
    }
    Py_XDECREF(self->_prev);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int Value_init(ValueObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"data", "_op", "_children", NULL};
    const char* _op = "";
    PyObject* _children = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|sO", kwlist, &self->data, &_op, &_children))
        return -1;
    self->grad = 0.0;
    self->_backward = NULL;

    if (_children) {
        if (!PyIter_Check(_children) && !PySequence_Check(_children)) {
            PyErr_SetString(PyExc_TypeError, "The _children attribute must be an iterable");
            return -1;
        }
        PyObject* iterator = PyObject_GetIter(_children);
        if (iterator == NULL) {
            return -1;
        }
        PyObject* item;
        while ((item = PyIter_Next(iterator))) {
            if (PySet_Add(self->_prev, item) == -1) {
                Py_DECREF(item);
                Py_DECREF(iterator);
                return -1;
            }
            Py_DECREF(item);
        }
        Py_DECREF(iterator);
    }
    self->_op = _op;
    return 0;
}

static PyObject* Value_getdata(ValueObject* self, void* closure) {
    return PyFloat_FromDouble(self->data);
}

static PyObject* Value_getgrad(ValueObject* self, void* closure) {
    return PyFloat_FromDouble(self->grad);
}

static PyObject* Value_get_op(ValueObject* self, void* closure) {
    if (self->_op) {
        return PyUnicode_FromString(self->_op);
    } else {
        Py_RETURN_NONE;
    }
}

static int Value_setdata(ValueObject* self, PyObject* data, void* closure) {
    if (data == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the data attribute");
        return -1;
    }
    if (PyFloat_Check(data)) {
        self->data = PyFloat_AsDouble(data);
    } else if (PyLong_Check(data)) {
        self->data = (double)PyLong_AsLong(data);
    } else {
        PyErr_SetString(PyExc_TypeError, "The data attribute data must be a float or an int");
        return -1;
    }
    return 0;
}

static PyObject* Value_get_prev(ValueObject* self, void* closure) {
    if (self->_prev) {
        Py_INCREF(self->_prev);
        return self->_prev;
    } else {
        Py_RETURN_NONE;
    }
}

static int Value_setgrad(ValueObject* self, PyObject* grad, void* closure) {
    if (grad == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the grad attribute");
        return -1;
    }
    if (PyFloat_Check(grad)) {
        self->grad = PyFloat_AsDouble(grad);
    } else if (PyLong_Check(grad)) {
        self->grad = (double)PyLong_AsLong(grad);
    } else {
        PyErr_SetString(PyExc_TypeError, "The value attribute grad must be a float or an int");
        return -1;
    }
    return 0;
}

static PyObject* Value_get_backward(ValueObject *self, void *closure) {
    if (self->_backward) {
        // if (is_unary(self->_op)) {
        //     Py_INCREF(((BackwardUnaryClosure*)self->_backward)->self);
        // } else {
        Py_INCREF(((BackwardBinaryClosure*)self->_backward)->self);
        Py_INCREF(((BackwardBinaryClosure*)self->_backward)->other);
        // }
        return (PyObject*)self->_backward;
    }
    Py_RETURN_NONE;
}

static PyGetSetDef Value_getseters[] = {
    {"data", (getter)Value_getdata, (setter)Value_setdata, "Value's data", NULL},
    {"grad", (getter)Value_getgrad, (setter)Value_setgrad, "Value's grad", NULL},
    {"_backward", (getter)Value_get_backward, (setter)NULL, "Value's backpropagation method", NULL},
    {"_prev", (getter)Value_get_prev, (setter)NULL, "Value's children", NULL},
    {"_op", (getter)Value_get_op, (setter)NULL, "Value's source operation", NULL},
    {NULL}  /* Sentinel */
};

static PyNumberMethods Value_as_number = {
    .nb_add = (binaryfunc)Value_add,
    .nb_subtract = (binaryfunc)Value_sub,
    .nb_multiply = (binaryfunc)Value_mul,
    .nb_true_divide = (binaryfunc)Value_div,
    .nb_negative = (unaryfunc)Value_neg,
    .nb_power = (ternaryfunc)Value_pow,
};

void build_topo(ValueObject* value, PyObject* visited, PyObject* topo) {
    if (PySet_Contains(visited, (PyObject*)value)) {
        return;
    }
    PySet_Add(visited, (PyObject*)value);

    PyObject* iterator = PyObject_GetIter(value->_prev);
    if (iterator == NULL) {
        return;
    }
    PyObject* item;
    while ((item = PyIter_Next(iterator))) {
        build_topo((ValueObject*)item, visited, topo);
        Py_DECREF(item);
    }
    Py_DECREF(iterator);

    PyList_Append(topo, (PyObject*)value);
}

static PyObject* Value_backward(ValueObject* self, PyObject* args) {
    self->grad = 1.0;

    PyObject* visited = PySet_New(NULL);
    PyObject* topo = PyList_New(0);

    if (visited == NULL || topo == NULL) {
        Py_XDECREF(visited);
        Py_XDECREF(topo);
        return NULL;
    }

    build_topo(self, visited, topo);

    PyList_Reverse(topo);

    Py_ssize_t i, n = PyList_Size(topo);
    for (i = 0; i < n; ++i) {
        ValueObject* v = (ValueObject*)PyList_GetItem(topo, i);
        if (v->_backward) {
            // if (is_unary(self->_op)) {
                // ((BackwardUnaryClosure*)v->_backward)->func(((BackwardUnaryClosure*)v->_backward)->self, v);
            // } else {
            ((BackwardBinaryClosure*)v->_backward)->func(((BackwardBinaryClosure*)v->_backward)->self, ((BackwardBinaryClosure*)v->_backward)->other, v);
            // }
        }
    }

    Py_DECREF(visited);
    Py_DECREF(topo);
    Py_RETURN_NONE;
}

static PyMethodDef Value_methods[] = {
    {"backward", (PyCFunction)Value_backward, METH_NOARGS, "Backpropagate gradients"},
    {"relu", (PyCFunction)Value_relu, METH_NOARGS, "ReLU activation"},
    {NULL}  /* Sentinel */
};

static PyTypeObject ValueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "engine.Value",
    .tp_doc = "Stores a floating point number and its gradient",
    .tp_basicsize = sizeof(ValueObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_as_number = &Value_as_number,
    .tp_new = Value_new,
    .tp_init = (initproc)Value_init,
    .tp_dealloc = (destructor)Value_dealloc,
    .tp_getset = Value_getseters,
    .tp_methods = Value_methods,
};

static PyObject* _value(double data, PyObject* children, const char* op) {
    PyObject* args = Py_BuildValue("(d)", data);
    PyObject* kwargs = Py_BuildValue("{s:s,s:O}", "_op", op, "_children", children);
    PyObject* value = PyObject_Call((PyObject*)&ValueType, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return value;
}

static PyModuleDef engine = {
    PyModuleDef_HEAD_INIT,
    .m_name = "engine",
    .m_doc = "Engine module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_engine(void) {
    PyObject *m;
    if (PyType_Ready(&ValueType) < 0)
        return NULL;

    m = PyModule_Create(&engine);
    if (m == NULL)
        return NULL;

    Py_INCREF(&ValueType);
    if (PyModule_AddObject(m, "Value", (PyObject *)&ValueType) < 0) {
        Py_DECREF(&ValueType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
