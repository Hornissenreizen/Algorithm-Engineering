all:
	cd build && cmake .. && make && mv src/einspeed.so ../einspeed.so
test:
	cd build && cmake .. && cd tests && make && ./catch2
init:
	cd external && git clone https://github.com/springer13/hptt && mv -f hptt_CMakeLists.txt hptt/CMakeLists.txt && cd ../build && rm -f CMakeCache.txt