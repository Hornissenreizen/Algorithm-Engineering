all:
	cd build && cmake .. && make && mv src/einspeed.so ../einspeed.so
test:
	cd build && cmake .. && cd tests && make && ./catch2