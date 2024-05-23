#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef struct {
    PyObject_HEAD
    int value;
} ValueObject;

static PyObject* Value_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    ValueObject *self;
    self = (ValueObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->value = 0;
    }
    return (PyObject *)self;
}

static void Value_dealloc(ValueObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int Value_init(ValueObject *self, PyObject *args, PyObject *kwds) {
    if (!PyArg_ParseTuple(args, "i", &self->value))
        return -1;
    return 0;
}

static PyObject* Value_getvalue(ValueObject* self, void* closure) {
    return PyLong_FromLong(self->value);
}

static int Value_setvalue(ValueObject* self, PyObject* value, void* closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the value attribute");
        return -1;
    }
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "The value attribute value must be an int");
        return -1;
    }
    self->value = PyLong_AsLong(value);
    return 0;
}

static PyGetSetDef Value_getseters[] = {
    {"value", (getter)Value_getvalue, (setter)Value_setvalue, "value", NULL},
    {NULL}  /* Sentinel */
};

static PyTypeObject ValueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "value.Value",
    .tp_doc = "Value objects",
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
