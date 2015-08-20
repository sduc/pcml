/*
 *  OutputNeuron.cpp
 *  PCMLproject
 *
 *  Authors J. Paratte & D. Huwyler
 *
 */

#include "OutputNeuron.h"

#include <math.h>

#include <stdlib.h>

#include "def.h"

#include <iostream>

using namespace std;

OutputNeuron::OutputNeuron() {
	
}

void OutputNeuron::init(int input_size, double learning_rate, double alpha) {
	this->size = input_size;
	this->learning_rate = learning_rate;
	this->alpha = alpha;
	
	if (alpha == 0.0) {
		has_momentum = false;
	} else {
		has_momentum = true;
	}
	
	first_pass = true;
	
	delta = 0.0;
	
	dw_prev = new double[input_size + 1];
	
	weights = new double[input_size+1];
	
	for (int i = 0; i<input_size+1; i++) {
		weights[i] = 2*(rand()/(10.0*(float)RAND_MAX))-0.1;
	}
}

void OutputNeuron::init_static(int input_size) {
	this->size = input_size;
	
	weights = new double[input_size+1];
	
	for (int i = 0; i<input_size+1; i++) {
		weights[i] = 2*(rand()/(10.0*(float)RAND_MAX))-0.1;
	}
}

double * OutputNeuron::get_weights() {
	return weights;
}

void OutputNeuron::set_weights(double * w) {
	for (int i = 0; i<size+1; i++) {
		weights[i] = w[i];
	}
}

double OutputNeuron::get_output() {
	return x_out;
}

void OutputNeuron::feed_input(double * inputs) {
	h = 0.0;
	for (int i = 0; i<size;i++) {
		h += weights[i]*inputs[i];
	}
	h += (-1.0*weights[size]);
	
	
	if (h<0) {
		h = max_double(h, -10.0); 
	}
	if (h>0) {
		h = min_double(h, 10.0);
	}
	
	x_out = gate_fct();
}


void OutputNeuron::update_weights(double * inputs) {
	double dw;
	for (int i = 0; i<size;i++) {
		dw = learning_rate*delta*inputs[i];
		weights[i] += dw;
		
		if (has_momentum) {
			if (!first_pass) {
				weights[i] += alpha * dw_prev[i];
			}
			// Care !! The instruction below changes the element just after it has been used
			dw_prev[i] = dw;
		}
	}
	
	weights[size] += learning_rate*delta*(-1.0);
	if (has_momentum) {
		if (!first_pass) {
			weights[size] += alpha*dw_prev[size];
		}
		dw_prev[size] = learning_rate*delta*(-1.0);
		
	}
	
	if (first_pass) {
		first_pass = false;
	}
}

double OutputNeuron::compute_delta(double target) {
	delta = gate_fct_deriv() * (target -  x_out);
	return delta;
}

double OutputNeuron::gate_fct() {
	return (1.0 / (1.0 + exp(-h)));
}

double OutputNeuron::gate_fct_deriv() {
	return(exp(-h) / ((1.0+exp(-h))*(1.0+exp(-h))));
}

double OutputNeuron::get_weight(int index) {
	return weights[index];
}
