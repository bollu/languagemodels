lstm: lstm.cpp
	g++ lstm.cpp -fsanitize=address -fsanitize=undefined -O3 -o lstm
