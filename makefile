all: rnn
	
rnn: rnn.cpp
	nvcc -g -O0 rnn.cpp -o rnn -std=c++14 \
		--compiler-options -Wall --compiler-options -Werror \
		--compiler-options -fsanitize=address

