#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
    double data;
    double grad;
} ValueObject;

static PyObject* Value_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    ValueObject *self;
    self = (ValueObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->data = 0.0;
        self->grad = 0.0;
    }
    return (PyObject *)self;
}

static void Value_dealloc(ValueObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int Value_init(ValueObject *self, PyObject *args, PyObject *kwds) {
    if (!PyArg_ParseTuple(args, "d", &self->data))
        return -1;
    self->grad = 0.0;
    return 0;
}

static PyObject* Value_getdata(ValueObject* self, void* closure) {
    return PyFloat_FromDouble(self->data);
}

static PyObject* Value_getgrad(ValueObject* self, void* closure) {
    return PyFloat_FromDouble(self->grad);
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
        PyErr_SetString(PyExc_TypeError, "The value attribute data must be a float or an int");
        return -1;
    }
    self->data = PyFloat_AsDouble(data);
    return 0;
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
    self->grad = PyFloat_AsDouble(grad);
    return 0;
}

static PyGetSetDef Value_getseters[] = {
    {"data", (getter)Value_getdata, (setter)Value_setdata, "Value's data", NULL},
    {"grad", (getter)Value_getgrad, (setter)Value_setgrad, "Value's grad", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject ValueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "value.Value",
    .tp_doc = "Stores a floating point number and it's gradient",
    .tp_basicsize = sizeof(ValueObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Value_new,
    .tp_init = (initproc)Value_init,
    .tp_dealloc = (destructor)Value_dealloc,
    .tp_getset = Value_getseters,
};

static PyModuleDef value = {
    PyModuleDef_HEAD_INIT,
    .m_name = "value",
    .m_doc = "Value module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_value(void) {
    PyObject *m;
    if (PyType_Ready(&ValueType) < 0)
        return NULL;

    m = PyModule_Create(&value);
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
