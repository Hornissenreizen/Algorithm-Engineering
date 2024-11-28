all:
	cd build && cmake .. && make && mv einsum.so ../einsum.so && python3 ../einsum.py
python:
	python3 setup.py build_ext --inplace --verbose && python3 test.py