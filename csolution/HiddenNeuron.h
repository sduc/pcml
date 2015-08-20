/*
 *  HiddenNeuron.h
 *  PCMLproject
 *
 *  Authors J. Paratte & D. Huwyler
 *
 */

#ifndef HIDDENNEURON_H
#define HIDDENNEURON_H

#include "def.h"

class HiddenNeuron {

public :
	HiddenNeuron();
	void init(int input_size, double learning_rate ,double alpha);
	void init_static(int input_size);

	void feed_input(double * inputs);
	double get_output();
	void update_weights(double * inputs);
	void compute_delta(double delta_bp);
	double * get_weights1();
	double * get_weights2();
	void set_weights1(double * w1);
	void set_weights2(double * w2);
	
private:	
	double gate_fct();
	double gdh1();
	double gdh2();

	int size;
	double * inputs;
	bool first_pass;
	bool has_momentum;
	double x_out;
	double alpha;
	double learning_rate;
	double h1;
	double h2;
	double delta1;
	double delta2;
	double * weights1;
	double * weights2;
	double * dw_prev1;
	double * dw_prev2;
};



#endif
