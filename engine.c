#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
    double data;
    double grad;
    PyObject* _backward;
    PyObject* _prev;
    const char* _op;
} ValueObject;

static PyObject* _value(double data, PyObject* children, const char* op);
static PyObject* Value_add(PyObject* self, PyObject* other);

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
    Py_XDECREF(self->_backward);
    Py_XDECREF(self->_prev);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* _backward(PyObject *self, PyObject *args) {
    Py_RETURN_NONE;
}

static PyMethodDef _backward_def = {
    "_backward",
    (PyCFunction)_backward,
    METH_VARARGS,
    "Gradient backpropagation method"
};

static int Value_init(ValueObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"data", "_op", "_children", NULL};
    const char* _op = "";
    PyObject* _children = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|sO", kwlist, &self->data, &_op, &_children))
        return -1;
    self->grad = 0.0;

    self->_backward = PyCMethod_New(&_backward_def, (PyObject *)self, NULL, NULL);
    if (self->_backward == NULL) {
        return -1;
    }

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
    Py_INCREF(self->_backward);
    return self->_backward;
}

static PyGetSetDef Value_getseters[] = {
    {"data", (getter)Value_getdata, (setter)Value_setdata, "Value's data", NULL},
    {"grad", (getter)Value_getgrad, (setter)Value_setgrad, "Value's grad", NULL},
    // TODO(yrmo): setter
    {"_backward", (getter)Value_get_backward, (setter)NULL, "Value's backpropagation method", NULL},
    {"_prev", (getter)Value_get_prev, (setter)NULL, "Value's children", NULL},
    {"_op", (getter)Value_get_op, (setter)NULL, "Value's source operation", NULL},
    {NULL}  /* Sentinel */
};

static PyNumberMethods Value_as_number = {
    .nb_add = (binaryfunc)Value_add,
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
};


static PyObject* _value(double data, PyObject* children, const char* op) {
    PyObject* args = Py_BuildValue("(d)", data);
    PyObject* kwargs = Py_BuildValue("{s:s,s:O}", "_op", op, "_children", children);
    PyObject* value = PyObject_Call((PyObject*)&ValueType, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    return value;
}
static PyObject* Value_add(PyObject* self, PyObject* other) {
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

    double result = ((ValueObject*)value_self)->data + ((ValueObject*)value_other)->data;
    PyObject* children = PyTuple_Pack(2, value_self, value_other);
    if (children == NULL) {
        Py_DECREF(value_self);
        Py_DECREF(value_other);
        return NULL;
    }
    PyObject* out = _value(result, children, "+");
    Py_DECREF(children);
    Py_DECREF(value_self);
    Py_DECREF(value_other);
    return out;
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
