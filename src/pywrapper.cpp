#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "LampLD.h"

namespace py = pybind11;

PYBIND11_MODULE(lampld, m) {
    m.doc() = "Local ancestry inference";

    py::class_<LampLD>(m, "LampLD")
            .def(py::init<int, int, int, int>(),
                 py::arg("n_snp"),
                 py::arg("n_anc"),
                 py::arg("n_proto"),
                 py::arg("window_size"))
            .def("set_pos", &LampLD::set_pos)
            .def("fit", &LampLD::fit)
            .def("infer_lanc", &LampLD::infer_lanc);
}