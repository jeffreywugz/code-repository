#include <Python.h>
#include "layout.h"

static PyObject * layout_wrapper(PyObject *self, PyObject *args);
static PyMethodDef LayoutMethods[] = {
	{"layout",  layout_wrapper, METH_VARARGS, "layout a graph"}, 
	{NULL, NULL, 0, NULL} 
};

PyMODINIT_FUNC initlayout(void)
{
	(void) Py_InitModule("layout", LayoutMethods);
}

static PyObject * layout_wrapper(PyObject *self, PyObject *args)
{
	int i, n, seqlen, *edges;
	double pos[MAX_NUM_VERTEXS*2];
	PyObject *edge_seq, *pos_seq, *p;

	if (!PyArg_ParseTuple(args, "O",&edge_seq))
		return NULL;
	edge_seq = PySequence_Fast(edge_seq,  "argument must be iterable");
	if(!edge_seq)return NULL;
	seqlen = PySequence_Fast_GET_SIZE(edge_seq);
	edges = malloc(seqlen*sizeof(double));
	if(!edges){
		Py_DECREF(edge_seq);
		return PyErr_NoMemory(  );
	}

	for(i=0; i < seqlen; i++) {
		PyObject *item = PySequence_Fast_GET_ITEM(edge_seq, i);
		if(!item) {
			Py_DECREF(edge_seq);
			free(edges);
			return 0;
		}
		edges[i] = PyInt_AsLong(item);
		Py_DECREF(item);
	}    

	n=layout(edges, seqlen/2, pos);
	pos_seq=Py_BuildValue("[]");
	for(i=0; i<n; i++){
		p=Py_BuildValue("(f)", pos[i]);
		PySequence_InPlaceConcat(pos_seq, p);
		Py_DECREF(p);
	}
	return pos_seq;
}
