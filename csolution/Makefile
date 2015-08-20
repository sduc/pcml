objects = main.o OutputNeuron.o HiddenNeuron.o
PCMLProject : $(objects)
	g++ -o PCMLProject -lm -fopenmp $(objects)
     
main.o : main.cpp def.h OutputNeuron.h HiddenNeuron.h
	g++ -fopenmp -c main.cpp 
HiddenNeuron.o : def.h
OutputNeuron.o : def.h
     
.PHONY : clean
clean :
	rm PCMLProject $(objects)
