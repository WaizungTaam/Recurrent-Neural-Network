all: rnn_demo.o

rnn_demo.o: include/rnn.h test/rnn_demo.cc
	g++ -std=c++11 -fopenmp -msse4 include/rnn.h test/rnn_demo.cc -o rnn_demo.o