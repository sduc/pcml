/*
 *  HiddenNeuron.cpp
 *  PCMLproject
 *
 *  Authors J. Paratte & D. Huwyler
 *
 */

#include "HiddenNeuron.h"
#include <stdlib.h>

#include <math.h>

#include <iostream>

#include "def.h"

using namespace std;

HiddenNeuron::HiddenNeuron() {

}

void HiddenNeuron::init(int input_size, double learning_rate , double alpha) {
	this->size = input_size;
	this->learning_rate = learning_rate;
	this->alpha = alpha;
	
	if (alpha == 0.0) {
		has_momentum = false;
	} else {
		has_momentum = true;
	}

	first_pass = true;
	
	delta1 = 0.0;
	delta2 = 0.0;
	
	// Previous weights
	dw_prev1 = new double[input_size + 1];
	dw_prev2 = new double[input_size + 1];
	
	weights1 = new double[input_size + 1];
	for (int i = 0; i<input_size+1; i++) {
		weights1[i] = 2*(rand()/(10.0*(float)RAND_MAX))-0.1;
		//cout << weights1[i] << endl;
	}
	
	weights2 = new double[input_size+1];
	for (int i = 0; i<input_size+1; i++) {
		weights2[i] = 2*(rand()/(10.0*(float)RAND_MAX))-0.1;
	}
	
}

void HiddenNeuron::init_static(int input_size) {
	this->size = input_size;
	
	weights1 = new double[input_size + 1];
	for (int i = 0; i<input_size+1; i++) {
		weights1[i] = 2*(rand()/(10.0*(float)RAND_MAX))-0.1;
		//cout << weights1[i] << endl;
	}
	
	weights2 = new double[input_size+1];
	for (int i = 0; i<input_size+1; i++) {
		weights2[i] = 2*(rand()/(10.0*(float)RAND_MAX))-0.1;
	}
	
}


double * HiddenNeuron::get_weights1() {
	return weights1;
}

double * HiddenNeuron::get_weights2() {
	return weights2;
}

void HiddenNeuron::set_weights1(double * w1) {
	for (int i = 0; i<size+1; i++) {
		weights1[i] = w1[i];
	}
}

void HiddenNeuron::set_weights2(double * w2) {
	for (int i = 0; i<size+1; i++) {
		weights2[i] = w2[i];
	}
}

void HiddenNeuron::feed_input(double * inputs) {
	h1 = 0.0;
	for (int i = 0; i<size;i++) {
		h1 += weights1[i]*inputs[i];
	}
	h1 += (-1.0*weights1[size]);
	
	h2 = 0.0;
	for (int i = 0; i<size;i++) {
		h2 += weights2[i]*inputs[i];
	}
	h2 += (-1.0*weights2[size]);
	
	if (h1<0) {
		h1 = max_double(h1, -10.0); 
	}
	if (h1>0) {
		h1 = min_double(h1, 10.0);
	}
	
	if (h2<0) {
		h2 = max_double(h2, -10.0); 
	}
	if (h2>0) {
		h2 = min_double(h2, 10.0);
	}
	x_out = gate_fct();
}

double HiddenNeuron::get_output() {
	return x_out;
}


void HiddenNeuron::update_weights(double * inputs) {
	double dw1, dw2;
	for (int i = 0; i<size;i++) {
		dw1 = learning_rate*delta1*inputs[i];
		dw2 = learning_rate*delta2*inputs[i];
		weights1[i] += dw1;
		weights2[i] += dw2;
		
		if (has_momentum) {
			if (!first_pass) {
				weights1[i] += alpha * dw_prev1[i];
				weights2[i] += alpha * dw_prev2[i];
			}
			// Care !! The instruction below changes the element just after it has been used
			dw_prev1[i] = dw1;
			dw_prev2[i] = dw2;
		}
	}
	
	weights1[size] += learning_rate*delta1*(-1.0);
	weights2[size] += learning_rate*delta2*(-1.0);
	if (has_momentum) {
		if (!first_pass) {
			weights1[size] += alpha*dw_prev1[size];
			weights2[size] += alpha*dw_prev2[size];
		}
		dw_prev1[size] = learning_rate*delta1*(-1.0);
		dw_prev2[size] = learning_rate*delta2*(-1.0);

	}
		
	if (first_pass) {
		first_pass = false;
	}
}

void HiddenNeuron::compute_delta(double sum) {
	delta1 = gdh1() * sum;
	delta2 = gdh2() * sum;
}

double HiddenNeuron::gate_fct() {
	return (h1 / (1.0 + exp(-h2)));
}

double HiddenNeuron::gdh1() {
	return (1.0 / (1.0 + exp(-h2)));
}

double HiddenNeuron::gdh2() {
	return h1*exp(-h2) / ( (1.0 + exp(-h2))*(1.0 + exp(-h2)) );
}
