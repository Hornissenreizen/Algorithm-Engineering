all:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make && mv src/einspeed.so ../einspeed.so
test:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cd tests && make && ./catch2
init:
	cd external && git clone https://github.com/springer13/hptt && mv -f hptt_CMakeLists.txt hptt/CMakeLists.txt
	mkdir -p build
	rm -f build/CMakeCache.txt