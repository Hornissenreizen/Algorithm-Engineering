all:
	cd build && cmake .. && make && mv src/einspeed.so ../einspeed.so && python3 ../einspeed.py
test:
	cd build && cmake .. && cd tests && make && ./catch2