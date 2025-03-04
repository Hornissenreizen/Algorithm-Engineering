# from setuptools import setup, Extension
# import numpy as np

# module = Extension(
#     "mymodule",
#     sources=["mymodule.c"],
#     include_dirs=[np.get_include()]
# )

# setup(
#     name="mymodule",
#     version="1.0",
#     ext_modules=[module]
# )


from setuptools import setup, Extension
import numpy as np

module = Extension(
    "mymodule",
    sources=["mymodule.cpp"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-fopenmp"],  # Optimization and OpenMP support
    extra_link_args=["-fopenmp"],            # Link OpenMP libraries
)

setup(
    name="mymodule",
    version="1.0",
    ext_modules=[module],
)


