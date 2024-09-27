all: speck_ddt speck_improvedDD

speck_ddt:
	g++ -O3 -std=c++11 -fopenmp -march=native -o speck_ddt speck_ddt.cpp

speck_improvedDD:
	g++ -o speck_improvedDD -O3 ./speck_improvedDD.cpp