rnn: rnn.cpp
	nvcc -g -O0 rnn.cpp -o rnn -std=c++11
