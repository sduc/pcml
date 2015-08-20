/*
 *  OutputNeuron.h
 *  PCMLproject
 *
 *  Authors J. Paratte & D. Huwyler
 *
 */

#ifndef OUTPUTNEURON_H
#define OUTPUTNEURON_H

class OutputNeuron {
public:
	OutputNeuron();
	void feed_input(double * inputs);
	double get_output();
	double gate_fct_deriv();
	double compute_delta(double target);
	void update_weights(double * inputs);
	double * get_weights();
	void set_weights(double * w);
	
	void init(int input_size, double learning_rate, double alpha);
	void init_static(int input_size);
	double get_weight(int index);
private:
	double gate_fct();
	double h;
	double * weights;
	int size;
	double * inputs;
	double delta;
	double * dw_prev;
	bool first_pass;
	bool has_momentum;
	double x_out;
	double alpha;
	double learning_rate;
};


#endif