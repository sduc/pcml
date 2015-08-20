/* Main file of PCMLProject
 * Authors : J. Paratte & D. Huwyler 
 *
 * For quick reading see run_SVM and run_mlp core functions
 * 
 * If opencv is available uncomment #define OPENCV
 * If openmp is available uncomment #define OMP (experimental)
 * 
 * To get back data when learning launch as : ./PCMLProject 2> out.txt
 * To avoid having too much information when learning launch as : ./PCMLProject 2> /dev/null
 * 
 */

#include <iostream>
#include <string>
#include <vector>
#include <float.h>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#include <cmath>
#include <omp.h>

#include "OutputNeuron.h"
#include "HiddenNeuron.h"

//#define OPENCV
//#define OMP

#ifdef OPENCV
#include <opencv/opencv.h>
#endif

#include <iostream>

#include "def.h"

using namespace std;

const int NB_IMAGES = 24300;

string path_dat = "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat";
string path_cat = "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat";

string path_dat_t = "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat";
string path_cat_t = "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat";

const int DATA_TESTING = 0;
const int DATA_TRAINING = 1;

const int size_n = 96;
const int size_m = 36;
const int size_s = 24;

bool small_img;
bool medium_img;

int nb_classes = 2;
int nb_dim;

int nb_samples_training;
int nb_samples_validation;
int nb_samples_testing;

vector<int> rand_indexes_validation;
vector<int> rand_indexes_testing;

unsigned char ** tab; 

float * targets_testing_SVM;
float * targets_training_SVM;

double ** images_training;
float ** targets_training;

double ** images_testing;
float ** targets_testing;

/** Utility functions **/

#ifdef OPENCV
void tab_to_img(IplImage * im, unsigned char * tab, int size) {
	
	for (int y=0; y<size; y++) {
		unsigned char* pimage = (unsigned char*)(im->imageData+y*im->widthStep);
		for (int x=0; x<size; x++, pimage++) {
			*pimage = tab[y*size+x];
		}
	}
}
#endif

float mean(unsigned char ** tab, int index, int nb_img) {
	float sum = 0.0;
	for (int i = 0; i<nb_img; i++) {
		sum+= tab[i][index];
	}
	return sum/nb_img;
}

float variance(unsigned char ** tab, int index, int nb_img) {
	float sum_x2 = 0.0;
	for (int i = 0; i<nb_img; i++) {
		sum_x2+= tab[i][index]*tab[i][index];
	}
	float e_x2 = sum_x2 / nb_img;
	float mu = mean(tab, index, nb_img);
	return (e_x2 - (mu*mu));
}

float variance_m(unsigned char ** tab, int index, int nb_img, float mean) {
	float sum_x2 = 0.0;
	for (int i = 0; i<nb_img; i++) {
		sum_x2+= tab[i][index]*tab[i][index];
	}
	float e_x2 = sum_x2 / nb_img;
	return (e_x2 - (mean * mean));
}

void display_progress_bar(int i, int max) {
	cout << '\xd';
	cout << "[";
	int percent = (int)((i / (float)(max))*100);
	for (int i = 0; i < percent/2; i++) {
		cout << "=";
	}
	for (int i = percent/2; i<50; i++) {
		cout << ".";
	}
	cout << "] ";
	cout << percent << "%";
	cout << flush;
}

/** Normalization function **/
/** Reduce and center on a pixel by pixel basis **/
void normalize(unsigned char ** tab, double ** tab_d, int n_elem) {
	float mu = 0.0;
	float sigma = 0.0;
	
	cout << flush;
	cout << "Preprocessing images" << endl;
	for (int i = 0; i<nb_dim; i++) {
		//Compute mean
		mu = mean(tab, i, n_elem);
		//Compute sigma
		sigma = variance_m(tab, i, n_elem, mu);
		
		// Care of zero values for sigma
		if (sigma == 0) {
			for (int j = 0; j<n_elem;j++) {
				tab_d[j][i] = ((double)tab[j][i] - mu);
			}
		} else {
			for (int j = 0; j<n_elem;j++) {
				tab_d[j][i] = ((double)tab[j][i] - mu)/ sigma;
			}
		}
		display_progress_bar(i+1, nb_dim);
	}

	cout << endl;
}

/** Resize function **/
/** Apply a bilinear interpolation **/
void resize(unsigned char * img, unsigned char * img_r, int size, int size_r) {
	float ratio = size/(float)size_r;
	int u, v, p, q;
	
	for (int x = 0; x < size_r; x++) {
		for (int y = 0; y < size_r; y++) {
			u = (int) x*ratio;
			v = (int) y*ratio;
			p = x*ratio - u;
			q = y*ratio - v;
			img_r[x*size_r + y] =(unsigned char)( (1-p)*(1-q)*img[u*size+v] + (1-p)*q*img[u*size+(v+1)] + p*(1-q)*img[(u+1)*size + v] + p*q*img[(u+1)*size + (v+1)]);
		}
	}
	
}

/** The Gaussian Kernel function **/
/** Used by SVM **/
double kernel_gaussian(double * x1, double * x2, int size, double sigma) {
	double sum = 0.0;
	for (int i = 0; i<size;i++) {
		sum+= (x1[i] - x2[i])*(x1[i] - x2[i]);
	}
	return exp( (-1.0)*(sum/ (2*sigma*sigma)));
}

/** prepare_data() : main utility function **/
/** This function reads the .mat files, resize images (if needed) and call preprocessing **/
void prepare_data(double ** tab_d, float ** targets, float * targets_SVM, int data_mode, bool full_database) {
	
	FILE* f_dat;
	FILE* f_cat;
	
	cout << "Reading Files ... ";
	
	if (data_mode == DATA_TRAINING) {
		f_dat = fopen(path_dat.c_str(), "r");
		if (f_dat == NULL) {
			cout << "Error reading the file dat!" << endl;
		}
		
		f_cat = fopen(path_cat.c_str(), "r");
		if (f_cat == NULL) {
			cout << "Error reading the file cat!" << endl;
		}
	} else {
		f_dat = fopen(path_dat_t.c_str(), "r");
		if (f_dat == NULL) {
			cout << "Error reading the file dat!" << endl;
		}
		
		f_cat = fopen(path_cat_t.c_str(), "r");
		if (f_cat == NULL) {
			cout << "Error reading the file cat!" << endl;
		}
	}	
	cout << "ok" << endl;
	
	bool verbose = false;
	
	if (verbose)
		cout << "DAT HEADER FILE" << endl;
	for (int i = 0; i<6; i++) {
		for (int i = 0; i< 4; i++) {
			if (verbose)
				cout << fgetc(f_dat) << " ";
			else
				fgetc(f_dat);
		}
		if (verbose)
			cout << endl;
	}
	
	
	if (verbose)
		cout << endl;
	
	if (verbose)
		cout << "CAT HEADER FILE" << endl;
	for (int i = 0; i<5; i++) {
		for (int i = 0; i< 4; i++) {
			if (verbose)
				cout << fgetc(f_cat) << " ";
			else
				fgetc(f_cat);
		}
		if (verbose)
			cout << endl;
	}
	
	if (verbose)
		cout << endl;
	
	cout << "Creating images and targets database : " << endl;
	
	int im_size;
	if (small_img) {
		im_size = size_s;
	} else {
		if (medium_img) {
			im_size = size_m;
		} else {
			im_size = size_n;
		}
	}
	
	int nb_img;
	if (full_database) {
		nb_img = NB_IMAGES;
	} else {
		nb_img = 2*(NB_IMAGES/5);
	}

	
	
	int c;
	
	unsigned char * temp = 0;
	if (small_img || medium_img) {
		temp = new unsigned char[size_n*size_n];
	}
	
	int nb = 0;
	int index = 0;
	
	while (nb < NB_IMAGES) {
		nb++;
		
		c = fgetc(f_cat);
		for (int i = 0; i<3; i++) {
			fgetc(f_cat);
		}
		
		display_progress_bar(nb, NB_IMAGES);
		
		bool write_tab = false;
		
		if (full_database) {
			write_tab = true;
			for (int i = 0; i<5; i++) {
				if (c==i) {
					targets[index][i] = 1.0;
				} else {
					targets[index][i] = 0.0;
				}

			}
		} else {
			if ((c == 2) || (c == 3)) {
				if (c==2) {
					targets[index][0] = 1.0;
					targets[index][1] = 0.0;
					targets_SVM[index] = -1.0;
				} else {
					targets[index][0] = 0.0;
					targets[index][1] = 1.0;
					targets_SVM[index] = 1.0;
				}
				
				write_tab = true;
			}
		}
		
		// Read image in temp
		if (small_img || medium_img) {
			for (int y=0; y<size_n; y++) {
				for (int x=0; x<size_n; x++) {
					unsigned char val = (unsigned char)fgetc(f_dat);
					if (write_tab) {
						temp[y*size_n + x] = val;
					}
				}
			}
			// Resize and put in general images tab
			if (write_tab) {
				if (small_img) {
					resize(temp, tab[index], size_n, size_s);
				} else {
					resize(temp, tab[index], size_n, size_m);
				}
			}
		} else {
			// No resize, simply reads and put a linearized version of the image in images tab
			unsigned char val;
			for (int y=0; y<size_n; y++) {
				for (int x=0; x<size_n; x++) {
					val = (unsigned char)fgetc(f_dat);
					if (write_tab) {
						tab[index][y*size_n + x] = val;
					}
				}
			}
		}
		
		//Ignoring second image of the pair
		for (int y=0; y<size_n; y++) {
			for (int x=0; x<size_n; x++) {
				fgetc(f_dat);
			}
		}
		
		if (write_tab) {
			index++;
		}
	}
	
	//Release
	delete [] temp;
	
	cout << flush;
	cout << endl;
	
	
	normalize(tab, tab_d, nb_img);
	
	/*
	 for (int i = 0; i<nb_img; i++) {
	 delete [] tab[i];
	 }
	 delete [] tab;
	 */
}


/** This function is called in "User mode" **/
/** It uses optimal parameters already computed and red from file **/
/** A testing error can be computed and a interactive display is available
 * if OpenCV is used */
void run_svm_using_file(string path_file) {
	float sigma = 0.1;
	float learning_rate = 0.01;
	FILE* f_opt;
	f_opt = fopen(path_file.c_str(), "r");
	if (f_opt == NULL) {
		cout << "Error reading the file dat!" << endl;
	}
	
	float alpha[nb_samples_training];
	
	char c;
	int index_buffer = 0;
	
	float f;
	
	// Main reading loop 
	for (int mu = 0; mu<nb_samples_training;mu++) {
		char tmp[20];
		index_buffer = 0;
		do {
			c = fgetc(f_opt);
			if (c != 32 && c != 10) {
				tmp[index_buffer] = c;
				index_buffer++;
			}
		} while(c != 32 && c != 10);
		
		f = atof(tmp);
		alpha[mu] = f;
	}
		
	
	
	char input[8];
	
	cout << "Do you want to evaluate the performance of these parameters ? [y/n] \n This evaluation can be long.  " << endl;
	cin >> input;
	
	double error_testing = 0.0;
	double sum = 0.0;
	double z_t;
	double etha = 0.0;
	
	/** Computing testing error **/
	if (input[0] == 'y') {
		cout << "Computing Testing error with " << nb_samples_testing << " samples ... " << endl;
		for (int i = 0; i <nb_samples_testing; i++) {
			display_progress_bar(i+1, nb_samples_testing);
			sum = 0.0;
			
			
			for (int rho = 0; rho < nb_samples_training; rho++) {
				sum += alpha[rho]*targets_training_SVM[rho]*kernel_gaussian(images_testing[rand_indexes_testing[i]], images_training[rho], nb_dim, sigma);
			}
			z_t = (-1.0)*etha + sum;
			
			if (z_t > 0) {
				error_testing += abs(targets_testing_SVM[rand_indexes_testing[i]] - 1.0);
			} else {
				error_testing += abs(targets_testing_SVM[rand_indexes_testing[i]] + 1.0);
			}
			
		}
		cout << flush;
		cout << endl;
		double err_test_perc = error_testing/(2.0*nb_samples_testing);
		cout << "Testing error : " << err_test_perc <<endl;
	}
	
	
#ifdef OPENCV
	/** Interactive display of testing samples and corresponding predicted classes **/
	IplImage * im_s = cvCreateImage(cvSize(size_s, size_s), IPL_DEPTH_8U, 1);
	IplImage * im_m = cvCreateImage(cvSize(size_m, size_m), IPL_DEPTH_8U, 1);
	IplImage * im_n = cvCreateImage(cvSize(size_n, size_n), IPL_DEPTH_8U, 1);
	cout << endl;
	cout << "Displaying 20 random samples and predicted classes ... " << endl;
	int nb_examples = 0;
	while(nb_examples < 25) {
		nb_examples++;
		
		int id = (int) (9000*(rand()/(float)RAND_MAX));
		
		sum = 0.0;
		
		for (int rho = 0; rho < nb_samples_training; rho++) {
			sum += alpha[rho]*targets_training_SVM[rho]*kernel_gaussian(images_testing[rand_indexes_testing[id]], images_training[rho], nb_dim, sigma);
		}
		
		z_t = (-1.0)*etha + sum;
		
		if (z_t > 0) {
			cout << "Truck" << endl;
		} else {
			cout << "Plane" << endl;
		}
		
		if (!small_img) {
			if (!medium_img) {
				tab_to_img(im_n, tab[rand_indexes_testing[id]], size_n);
				cvShowImage("Show", im_n);
			} else {
				tab_to_img(im_m, tab[rand_indexes_testing[id]], size_m);
				cvShowImage("Show", im_m);
			}
		} else {
			tab_to_img(im_s, tab[rand_indexes_testing[id]], size_s);
			cvShowImage("Show", im_s);
		}
		
		cvWaitKey(0);
		
	}
#endif
	
	
}

/** Same as above, but for MLP (see run_svm_using_file fct) **/
void run_mlp_using_file(string path_file) {
	FILE* f_opt;
	f_opt = fopen(path_file.c_str(), "r");
	if (f_opt == NULL) {
		cout << "Error reading the file dat!" << endl;
	}
	
	int nb_hidden = 50;
	int nb_input = nb_dim;
	
	int nb_output = nb_classes;
	
	HiddenNeuron hidden_layer[nb_hidden];
	
	for (int i = 0; i<nb_hidden;i++) {
		hidden_layer[i].init_static(nb_input);
	}
	
	OutputNeuron output_layer[nb_output];
	for (int i = 0; i<nb_output;i++) {
		output_layer[i].init_static(nb_hidden);
	}
	
	double ** optimal_weights_h1 = new double*[nb_hidden];
	double ** optimal_weights_h2 = new double*[nb_hidden];
	for (int i = 0; i<nb_hidden; i++) {
		optimal_weights_h1[i] = new double[nb_input+1];
		optimal_weights_h2[i] = new double[nb_input+1];
		
	}
	
	double ** optimal_weights_out = new double*[nb_output];
	for (int i = 0; i<nb_output; i++) {
		optimal_weights_out[i] = new double[nb_hidden+1];
	}
	
	// Main reading loop 
	
	char c;
	int index_buffer = 0;
	
	float f;
	
	for (int i = 0; i<nb_hidden; i++) {
		for (int j = 0; j<nb_input+1; j++) {
			char tmp_1[20];
			index_buffer = 0;
			do {
				c = fgetc(f_opt);
				if (c != 32 && c != 10) {
					tmp_1[index_buffer] = c;
					index_buffer++;
				}
			} while(c != 32 && c != 10);
			
			f = atof(tmp_1);
			optimal_weights_h1[i][j] = f;
			
			char tmp_2[20];
			index_buffer = 0;
			do {
				c = fgetc(f_opt);
				if (c != 32 && c != 10) {
					tmp_2[index_buffer] = c;
					index_buffer++;
				}
			} while(c != 32 && c != 10);
			
			f = atof(tmp_2);
			optimal_weights_h2[i][j] = f;
			
		}
		c = fgetc(f_opt);
	}
	c = fgetc(f_opt);
	
	for (int i = 0; i<nb_output; i++) {
		for (int j = 0; j<nb_hidden+1; j++) {
			char tmp[20];
			index_buffer = 0;
			do {
				c = fgetc(f_opt);
				if (c != 32 && c != 10) {
					tmp[index_buffer] = c;
					index_buffer++;
				}
			} while(c != 32 && c != 10);
			
			f = atof(tmp);
			optimal_weights_out[i][j] = f;
		}
		c = fgetc(f_opt);
	}
	
	// Setting weights
	for (int i = 0; i<nb_hidden; i++) {
		hidden_layer[i].set_weights1(optimal_weights_h1[i]);
		hidden_layer[i].set_weights2(optimal_weights_h2[i]);
	}

	for (int i = 0; i<nb_output; i++) {
		output_layer[i].set_weights(optimal_weights_out[i]);
	}
	
	
	
	char input[8];
	cout << "Do you want to evaluate the performance of these parameters ? [y/n] \n This evaluation can be long.  " << endl;
	cin >> input;
	
	double in_h[nb_hidden];
	double local_err_t = 0.0;
	double local_err_t_mse = 0.0;
	double out, t;
	
	if (input[0] == 'y') {
		cout << "Computing Testing error with " << nb_samples_testing << " samples ... " << endl;
		/** Testing error **/
		
		for (int i = 0; i<nb_samples_testing; i++) {
			for (int j = 0; j<nb_hidden;j++) {
				hidden_layer[j].feed_input(images_testing[rand_indexes_testing[i]]);
				in_h[j] = hidden_layer[j].get_output();
			}
			for (int j = 0; j<nb_output; j++) {
				output_layer[j].feed_input(in_h);
			}
			for (int j = 0; j<nb_output; j++) {
				out = output_layer[j].get_output();
				t = targets_testing[rand_indexes_testing[i]][j];
				local_err_t_mse += (out - t)*(out - t);
			}
			
			int max_id = 0;
			float max = 0.0;
			double val = 0.0;
			for (int j = 0; j<nb_output; j++) {
				val = output_layer[j].get_output();
				if (val > max) {
					max = val;
					max_id = j;
				}
			}
			for (int j = 0; j<nb_output; j++) {
				if (j == max_id) {
					local_err_t += abs(1.0 - targets_testing[rand_indexes_testing[i]][j]);
				} else {
					local_err_t += targets_testing[rand_indexes_testing[i]][j];
				}
				
			}
		}
		
		local_err_t /= nb_classes;
		
		double local_perc_t = local_err_t / nb_samples_testing; // Classification error
		double local_perc_t_mse = local_err_t_mse / nb_samples_testing; // MSE (not used in user mode)
		
		cout << "Testing error :" << local_perc_t << endl;
	}

#ifdef OPENCV
	cout << endl;
	cout << "Displaying 20 random samples and predicted classes ... " << endl;
	IplImage * im_s = cvCreateImage(cvSize(size_s, size_s), IPL_DEPTH_8U, 1);
	IplImage * im_m = cvCreateImage(cvSize(size_m, size_m), IPL_DEPTH_8U, 1);
	IplImage * im_n = cvCreateImage(cvSize(size_n, size_n), IPL_DEPTH_8U, 1);
	int nb_examples = 0;
	while(nb_examples < 20) {
		nb_examples++;
		
		int id = (int) (9000*(rand()/(float)RAND_MAX));
		
		for (int j = 0; j<nb_hidden;j++) {
			hidden_layer[j].feed_input(images_testing[rand_indexes_testing[id]]);
			in_h[j] = hidden_layer[j].get_output();
		}
		for (int j = 0; j<nb_output; j++) {
			output_layer[j].feed_input(in_h);
		}
		
		int max_id = 0;
		float max = 0.0;
		double val = 0.0;
		for (int j = 0; j<nb_output; j++) {
			val = output_layer[j].get_output();
			if (val > max) {
				max = val;
				max_id = j;
			}
		}
		
		if (nb_classes == 2) {
			max_id += 2;
		}
		
		switch (max_id) {
			case 0:
				cout << "Animal";
				break;
			case 1:
				cout << "Human ";
				break;
			case 2:
				cout << "Plane ";
				break;
			case 3:
				cout << "Truck ";
				break;
			case 4:
				cout << "Car   ";
				break;
			default:
				cout << "Should not be there !!" << endl;
				break;
		}
		cout << endl;
		
		if (!small_img) {
			if (!medium_img) {
				tab_to_img(im_n, tab[rand_indexes_testing[id]], size_n);
				cvShowImage("Show", im_n);
			} else {
				tab_to_img(im_m, tab[rand_indexes_testing[id]], size_m);
				cvShowImage("Show", im_m);
			}
		} else {
			tab_to_img(im_s, tab[rand_indexes_testing[id]], size_s);
			cvShowImage("Show", im_s);
		}
		
		cvWaitKey(0);
		
	}
#endif
	
			
}

#pragma mark Core Functions

/******************************************
 ******* Kerel Adatron SVM function *******
 ******************************************/

void run_SVM(int nb_iter, float * targets, double ** images, float * targets_t, double ** images_t, float learning_rate, double sigma) {
	cout << "\nRunning Kernel-Adatron SVM" << endl;
	
	// Initializations 
	int nb_mu = nb_samples_training;
	
	float alpha[nb_mu];
	float optimal_alpha[nb_mu];
	
	for (int mu = 0; mu<nb_mu;mu++) {
		alpha[mu] = 0.01;
		optimal_alpha[mu] = 0.0;
	}
	
	float etha = 0;
	float z[nb_mu];
	
	for (int mu = 0; mu<nb_mu;mu++) {
		z[mu] = 10*targets[mu];
	}
	
	bool stop = false;
	
	int nb_rho = nb_mu;
	
	
	double sum = 0.0;
	
	double dalpha = 0.0;
	
	float C = 0.05;
	
	float zplus = 0;
	float zminus = 0;
	
	float u = 0.0;
	
	int count = 0;
	
	double pk = 0.0;
	
	double z_t = 0.0;
	double error_training = 0.0;
	double error_validation = 0.0;
	
	double min_error_training = 1.0;
	double min_error_validation = 1.0;
	
	time_t start, end;
	
	srand ( unsigned ( time (NULL) ) );
	vector<int> rand_indexes_training;
	
	// set some values:
	for (int i=0; i<nb_samples_training; i++) {
		rand_indexes_training.push_back(i);
	}
	
	int max_iter = 20;
	
	// To run the version of the algorithm from the original paper KA
	bool alternate_mode = true;
	
	while (!stop) {
		time(&start);
		// Present samples in randomized order each new iteration
		random_shuffle ( rand_indexes_training.begin(), rand_indexes_training.end() );
		
		cout << "\niteration #" << count+1 << endl;
		for (int mu = 0; mu < nb_mu; mu++) {
			display_progress_bar(mu+1, nb_mu);
			
			sum = 0.0;
			
			// Compute the sum
			for (int rho = 0; rho < nb_rho; rho++) {
				sum += alpha[rho]*targets[rho]*kernel_gaussian(images[rand_indexes_training[mu]], images[rho], nb_dim, sigma);
			}
			z[rand_indexes_training[mu]] = (-1.0)*etha + sum;
			
			// Compute delta_alpha
			dalpha = learning_rate*(1.0 - targets[rand_indexes_training[mu]]*z[rand_indexes_training[mu]]);
			alpha[rand_indexes_training[mu]] += dalpha;
			
			// Keep alphas in reasonible range
			if (alpha[rand_indexes_training[mu]] <= 0) {
				alpha[rand_indexes_training[mu]] = 0;
			}
			if (alpha[rand_indexes_training[mu]] > C) {
				alpha[rand_indexes_training[mu]] = C;
			}
			
			zplus = 10;
			zminus = -10;
			
			// Compute min of +1 targets and max of -1 targets (zplus and zminus)
			for (int rho = 0; rho<nb_rho;rho++) {
				if ( (targets[rho] == 1.0) & (zplus > z[rho]) & (alpha[rho] < C )) {
					zplus = z[rho];
				}
				
				if ( (targets[rho] == -1.0) & (zminus < z[rho]) & (alpha[rho] < C )) {
					zminus = z[rho];
				}
			}
			
			u = 0.5*(zplus + zminus);
			
			// Stop if the margin gets degenerated
			if (0.5*(zplus - zminus) >= 10.0) {
				stop = true;
				break;
			} else {
				if (!alternate_mode) {
					etha = u + etha;
					
					for (int rho= 0;rho<nb_rho; rho++) {
						z[rho] = z[rho] - u;
				}
			}
				 
			}
			
			
		}
		
		// Stop if consistently close to one
		if ((1.0 - (0.5*(zplus - zminus))) < 0.1) {
			stop = true;
		}
		// Stop if the final margin gets degenerated
		if (0.5*(zplus - zminus) >= 2.0) {
			stop = true;
		}
		
		cout << flush;
		cout << endl;
		
		char sign;
		if (((0.5*(zplus - zminus)) - pk) > 0) {
			sign = '+';
		} else {
			sign = ' ';
		}
		
		cout << "Margin k=" << (0.5*(zplus - zminus)) << " " << sign << ((0.5*(zplus - zminus)) - pk) << ")";
		
		cout << flush;
		
		pk = (0.5*(zplus - zminus));
		count++;
		
		double err_train_perc = 0.0;
		double err_valid_perc = 0.0;
		bool compute_error = true;
		
		// Computing error at each epoch allow to play some sort of "early stopping" in SVM
		if (compute_error) {
			cout << endl;
			cout << "Computing error training" << endl;
			error_training = 0.0;
			for (int i = 0; i <nb_samples_training; i++) {
				display_progress_bar(i+1, nb_samples_training);
				sum = 0.0;
				
				
				for (int rho = 0; rho < nb_rho; rho++) {
					sum += alpha[rho]*targets[rho]*kernel_gaussian(images[i], images[rho], nb_dim, sigma);
				}
				z_t = (-1.0)*etha + sum;
				
				if (z_t > 0) {
					error_training += abs(targets[i] - 1.0);
				} else {
					error_training += abs(targets[i] + 1.0);
				}
				
			}
			cout << flush;
			cout << endl;
			err_train_perc = error_training/(2.0*nb_samples_training);
			cout << "Error training: " << err_train_perc <<endl;
			
			cout << "Computing error validation" << endl;
			error_validation = 0.0;
			for (int i = 0; i <nb_samples_validation; i++) {
				display_progress_bar(i+1, nb_samples_validation);
				sum = 0.0;
				
				
				for (int rho = 0; rho < nb_rho; rho++) {
					sum += alpha[rho]*targets[rho]*kernel_gaussian(images_t[rand_indexes_validation[i]], images[rho], nb_dim, sigma);
				}
				z_t = (-1.0)*etha + sum;
				
				if (z_t > 0) {
					error_validation += abs(targets_t[rand_indexes_validation[i]] - 1.0);
				} else {
					error_validation += abs(targets_t[rand_indexes_validation[i]] + 1.0);
				}
				
			}
			cout << flush;
			cout << endl;
			err_valid_perc = error_validation/(2.0*nb_samples_validation);
			cout << "Error validation: " << err_valid_perc << endl;
			
			if ((err_valid_perc <= 0.001) && (err_train_perc <= 0.001)) {
				stop = true;
			}
			if (count >= max_iter) {
				stop = true;
			}
		}
		
		time(&end);
		
		// Early stopping for SVM
		if (err_valid_perc < min_error_validation) {
			min_error_validation = err_valid_perc;
			min_error_training = err_train_perc;
			
			if (err_train_perc < 0.25) {
				for (int mu = 0; mu<nb_mu;mu++) {
					optimal_alpha[mu] = alpha[mu];
				}
			}
		}
		
		cerr << count << " " << (0.5*(zplus - zminus)) << " "<<  err_train_perc << " " << err_valid_perc << endl;
		
	}
	cerr << "Saving optimal parameters" << endl;
	
	for (int mu = 0; mu<nb_mu;mu++) {
		cerr << optimal_alpha[mu] << endl;
	}
	
	for (int mu = 0; mu<nb_mu;mu++) {
		alpha[mu] = optimal_alpha[mu];
	}
	
	
	cout << "Computing final error testing" << endl;
	double error_testing = 0.0;
	
	
	// Computing final error using optimal alphas
	for (int i = 0; i <nb_samples_testing; i++) {
		display_progress_bar(i+1, nb_samples_testing);
		sum = 0.0;
		
		
		for (int rho = 0; rho < nb_rho; rho++) {
			sum += alpha[rho]*targets[rho]*kernel_gaussian(images_t[rand_indexes_testing[i]], images[rho], nb_dim, sigma);
		}
		z_t = (-1.0)*etha + sum;
		
		if (z_t > 0) {
			error_testing += abs(targets_t[rand_indexes_testing[i]] - 1.0);
		} else {
			error_testing += abs(targets_t[rand_indexes_testing[i]] + 1.0);
		}
		
	}
	cout << flush;
	cout << endl;
	double err_test_perc = error_testing/(2.0*nb_samples_testing);
	cout << "Error testing: " << err_test_perc <<endl;
	cerr << "Error testing: " << err_test_perc <<endl;
	
#ifdef OPENCV
	IplImage * im_s = cvCreateImage(cvSize(size_s, size_s), IPL_DEPTH_8U, 1);
	IplImage * im_m = cvCreateImage(cvSize(size_m, size_m), IPL_DEPTH_8U, 1);
	IplImage * im_n = cvCreateImage(cvSize(size_n, size_n), IPL_DEPTH_8U, 1);
	
	int nb_examples = 0;
	while(nb_examples < 25) {
		nb_examples++;
		
		int id = (int) (9000*(rand()/(float)RAND_MAX));
		
		sum = 0.0;
		
		for (int rho = 0; rho < nb_rho; rho++) {
			sum += alpha[rho]*targets[rho]*kernel_gaussian(images_t[id], images[rho], nb_dim, sigma);
		}
		z_t = (-1.0)*etha + sum;
		
		if (z_t > 0) {
			cout << "Truck" << endl;
		} else {
			cout << "Plane" << endl;
		}
		
		if (!small_img) {
			if (!medium_img) {
				tab_to_img(im_n, tab[id], size_n);
				cvShowImage("Show", im_n);
			} else {
				tab_to_img(im_m, tab[id], size_m);
				cvShowImage("Show", im_m);
			}
		} else {
			tab_to_img(im_s, tab[id], size_s);
			cvShowImage("Show", im_s);
		}
		
		cvWaitKey(0);
		
	}
#endif
}


/******************************************
 ********** Backprop MLP function *********
 ******************************************/

void run_backprop(double learning_rate = 0.01, double momentum = 0.9, int nb_hidden = 50) {
	
	cout << "\nRunning BackProp" << endl;
	
	// Initializations
	int nb_output = nb_classes;
	
	HiddenNeuron hidden_layer[nb_hidden];
	
	for (int i = 0; i<nb_hidden;i++) {
		hidden_layer[i].init(nb_dim, learning_rate, momentum);
	}
	
	OutputNeuron output_layer[nb_output];
	for (int i = 0; i<nb_output;i++) {
		output_layer[i].init(nb_hidden, learning_rate, momentum);
	}
	
	double ** optimal_weights_h1 = new double*[nb_hidden];
	double ** optimal_weights_h2 = new double*[nb_hidden];
	for (int i = 0; i<nb_hidden; i++) {
		optimal_weights_h1[i] = new double[nb_dim+1];
		optimal_weights_h2[i] = new double[nb_dim+1];
		
	}
	
	double ** optimal_weights_out = new double*[nb_output];
	for (int i = 0; i<nb_output; i++) {
		optimal_weights_out[i] = new double[nb_hidden+1];
	}
	
	srand ( unsigned ( time (NULL) ) );
	vector<int> rand_indexes_training;
	
	// set some values:
	for (int i=0; i<nb_samples_training; i++) {
		rand_indexes_training.push_back(i);
	}
	
	random_shuffle ( rand_indexes_training.begin(), rand_indexes_training.end() );
	
	double in_h[nb_hidden];
	
	time_t start, end;
	double dif;
	
	double total_time = 0.0;
	
	int nb_iter = 0;
	int max_iter = 10;
	
	double min_error_training = 1.0;
	double min_error_validation = 1.0;
	
	int id_early;
	
	double delta_out[nb_output];
	
	while((nb_iter < max_iter)){
		
		time(&start);
		
		// Present samples in randomized order
		random_shuffle (rand_indexes_training.begin(), rand_indexes_training.end());
		
		for (int i = 0; i<nb_samples_training;i++) {
			// Feed  input to hidden layer and get intermediate outputs
#ifdef OMP
#pragma omp parallel for
#endif
			for (int j = 0; j<nb_hidden;j++) {
				hidden_layer[j].feed_input(images_training[rand_indexes_training.at(i)]);
				in_h[j] = hidden_layer[j].get_output();
			}
			// Feed input to output layer and get output
			for (int j = 0; j< nb_output; j++) {
				output_layer[j].feed_input(in_h);
			}
			
			//Compute delta in output layer by providing target
			for (int j = 0; j< nb_output; j++) {
				delta_out[j] = output_layer[j].compute_delta(targets_training[rand_indexes_training.at(i)][j]);
			}
			
			//Compute delta in hidden layer by providing delta and output_weights
#ifdef OMP
#pragma omp parallel for
#endif
			for (int j = 0; j<nb_hidden;j++) {
				double sum = 0.0;
				for (int k = 0; k<nb_output; k++) {
					sum += delta_out[k]*output_layer[k].get_weight(j);
				}
				hidden_layer[j].compute_delta(sum);
			}
			
			//Update weights in every layer
#ifdef OMP
#pragma omp parallel for
#endif			
			for (int j=0; j<nb_hidden; j++) {
				hidden_layer[j].update_weights(images_training[rand_indexes_training.at(i)]);
			}
			
			for (int j = 0; j<nb_output; j++) {
				output_layer[j].update_weights(images_training[rand_indexes_training.at(i)]);
			}
			
			
		}
		
		bool mse = true;
		
		// Compute error on output neuron
		double local_err = 0.0;
		double local_err_mse = 0.0;
		double out, t;
#ifdef OMP
#pragma omp parallel for
#endif
		for (int i = 0; i<nb_samples_training; i++) {
			for (int j = 0; j<nb_hidden;j++) {
				hidden_layer[j].feed_input(images_training[rand_indexes_training.at(i)]);
				in_h[j] = hidden_layer[j].get_output();
			}
			for (int j = 0; j<nb_output; j++) {
				output_layer[j].feed_input(in_h);
			}
			
			for (int j = 0; j<nb_output; j++) {
				out = output_layer[j].get_output();
				t = targets_training[rand_indexes_training.at(i)][j];
				local_err_mse += (out - t)*(out - t);
			}
			int max_id = 0;
			float max = 0.0;
			double val = 0.0;
			for (int j = 0; j<nb_output; j++) {
				val = output_layer[j].get_output();
				if (val > max) {
					max = val;
					max_id = j;
				}
			}
			for (int j = 0; j<nb_output; j++) {
				if (j == max_id) {
					local_err += abs(1.0 - targets_training[rand_indexes_training.at(i)][j]);
				} else {
					local_err += targets_training[rand_indexes_training.at(i)][j];
				}
				
			}
			
		}
		
		local_err /= nb_classes;
		
		
		float local_perc = local_err / nb_samples_training;
		
		float local_perc_mse = local_err_mse / nb_samples_training;
		
		
		double local_err_t = 0.0;
		
		double local_err_t_mse = 0.0;	
		
		
		// Computing validation error
#ifdef OMP
#pragma omp parallel for
#endif
		for (int i = 0; i<nb_samples_validation; i++) {
			for (int j = 0; j<nb_hidden;j++) {
				hidden_layer[j].feed_input(images_testing[rand_indexes_validation[i]]);
				in_h[j] = hidden_layer[j].get_output();
			}
			for (int j = 0; j<nb_output; j++) {
				output_layer[j].feed_input(in_h);
			}
			for (int j = 0; j<nb_output; j++) {
				out = output_layer[j].get_output();
				t = targets_testing[rand_indexes_validation[i]][j];
				local_err_t_mse += (out - t)*(out - t);
			}
			int max_id = 0;
			float max = 0.0;
			double val = 0.0;
			for (int j = 0; j<nb_output; j++) {
				val = output_layer[j].get_output();
				if (val > max) {
					max = val;
					max_id = j;
				}
			}
			for (int j = 0; j<nb_output; j++) {
				if (j == max_id) {
					local_err_t += abs(1.0 - targets_testing[rand_indexes_validation[i]][j]);
				} else {
					local_err_t += targets_testing[rand_indexes_validation[i]][j];
				}
				
			}
		}
		
		
		local_err_t /= nb_classes;
		
		float local_perc_t = local_err_t / nb_samples_validation;
		float local_perc_t_mse = local_err_t_mse / nb_samples_validation;
		
		nb_iter++;
		
		time(&end);
		
		dif = difftime (end,start);
		
		total_time += dif;
		
		printf ("\xd iteration #%d training %.2lf validation %.2lf  (~%d s)",nb_iter, local_perc, local_perc_t, (int)dif );
		
		bool verbose = true;
		
		if (verbose) {
			cerr << nb_iter << " " << local_perc_mse << " " << local_perc_t_mse << " " << local_perc << " " << local_perc_t << endl;
		}
		
		// Early stopping
		if (local_perc_t < min_error_validation) {
			min_error_validation = local_perc_t;
			min_error_training = local_perc;
			
			if (local_perc_t < 0.25) {
				id_early = nb_iter;
				for (int i = 0; i<nb_hidden; i++) {
					optimal_weights_h1[i] = hidden_layer[i].get_weights1();
					optimal_weights_h2[i] = hidden_layer[i].get_weights2();
				}
				
				for (int i = 0; i<nb_output; i++) {
					optimal_weights_out[i] = output_layer[i].get_weights();
				}
			}
			
		}
		
		cout << flush;
	}
	
	cout << "Saving optimal paramters ... ";
	
	cerr << "Optimal parameters" << endl;
	
	cout << endl;
	for (int i = 0; i<nb_hidden; i++) {
		for (int j = 0; j<nb_dim+1; j++) {
			cerr << optimal_weights_h1[i][j] << " " << optimal_weights_h2[i][j] << endl;
		}
		cerr << endl;
		hidden_layer[i].set_weights1(optimal_weights_h1[i]);
		hidden_layer[i].set_weights2(optimal_weights_h2[i]);
	}
	cerr << endl;
	for (int i = 0; i<nb_output; i++) {
		for (int j = 0; j<nb_hidden+1; j++) {
			cerr << optimal_weights_out[i][j] << endl;
		}
		cerr << endl;
		output_layer[i].set_weights(optimal_weights_out[i]);
	}
	
	cout << flush;
	cout << endl;
	
	cout << "Displaying with error finded while early stopping : " << min_error_training << " " << min_error_validation << endl;
	
	cerr << "Early stopping : iter#"<< id_early << " " << min_error_training << " " << min_error_validation << endl;
	
	cout << "Computing final error with testing set : " << endl;
	
	double local_err_t = 0.0;
	double local_err_t_mse = 0.0;
	double out, t;
	for (int i = 0; i<nb_samples_testing; i++) {
		for (int j = 0; j<nb_hidden;j++) {
			hidden_layer[j].feed_input(images_testing[rand_indexes_testing[i]]);
			in_h[j] = hidden_layer[j].get_output();
		}
		for (int j = 0; j<nb_output; j++) {
			output_layer[j].feed_input(in_h);
		}
		for (int j = 0; j<nb_output; j++) {
			out = output_layer[j].get_output();
			t = targets_testing[rand_indexes_testing[i]][j];
			local_err_t_mse += (out - t)*(out - t);
		}
		
		int max_id = 0;
		float max = 0.0;
		double val = 0.0;
		for (int j = 0; j<nb_output; j++) {
			val = output_layer[j].get_output();
			if (val > max) {
				max = val;
				max_id = j;
			}
		}
		for (int j = 0; j<nb_output; j++) {
			if (j == max_id) {
				local_err_t += abs(1.0 - targets_testing[rand_indexes_testing[i]][j]);
			} else {
				local_err_t += targets_testing[rand_indexes_testing[i]][j];
			}
			
		}
	}
	
	local_err_t /= nb_classes;
	
	double local_perc_t = local_err_t / nb_samples_validation;
	double local_perc_t_mse = local_err_t_mse / nb_samples_validation;
	
	cout << "Testing error :" << local_perc_t_mse << " " << local_perc_t << endl;
	cerr << "Testing error :" << local_perc_t_mse << " " << local_perc_t << endl;
	
#ifdef OPENCV
	
	IplImage * im_s = cvCreateImage(cvSize(size_s, size_s), IPL_DEPTH_8U, 1);
	IplImage * im_m = cvCreateImage(cvSize(size_m, size_m), IPL_DEPTH_8U, 1);
	IplImage * im_n = cvCreateImage(cvSize(size_n, size_n), IPL_DEPTH_8U, 1);
	int nb_examples = 0;
	while(nb_examples < 20) {
		nb_examples++;
		
		int id = (int) (9000*(rand()/(float)RAND_MAX));
		
		for (int j = 0; j<nb_hidden;j++) {
			hidden_layer[j].feed_input(images_testing[id]);
			in_h[j] = hidden_layer[j].get_output();
		}
		for (int j = 0; j<nb_output; j++) {
			output_layer[j].feed_input(in_h);
		}
		
		int max_id = 0;
		float max = 0.0;
		double val = 0.0;
		for (int j = 0; j<nb_output; j++) {
			val = output_layer[j].get_output();
			if (val > max) {
				max = val;
				max_id = j;
			}
		}
		
		switch (max_id) {
			case 0:
				cout << "Animal";
				break;
			case 1:
				cout << "Human ";
				break;
			case 2:
				cout << "Plane ";
				break;
			case 3:
				cout << "Truck ";
				break;
			case 4:
				cout << "Car   ";
				break;
			default:
				cout << "Should not be there !!" << endl;
				break;
		}
		cout << endl;
		
		if (!small_img) {
			if (!medium_img) {
				tab_to_img(im_n, tab[id], size_n);
				cvShowImage("Show", im_n);
			} else {
				tab_to_img(im_m, tab[id], size_m);
				cvShowImage("Show", im_m);
			}
		} else {
			tab_to_img(im_s, tab[id], size_s);
			cvShowImage("Show", im_s);
		}
		
		cvWaitKey(0);
		
	}
#endif
	cout << endl;
}


int main (int argc, char * const argv[]) {

	int nb_img;
	bool user_mode = false;
	
	if (argc <= 2) {
		cout << "\nPCML Project : D. Huwyler / J. Paratte\n " << endl;
		char input[8];
		
		cout << "Do you want to run new learning processes (instead of using existing files) [y/n]? " << endl;
		cin >> input;
		
		if (input[0] == 'n') {
			cout << "User mode chosen." << endl;
			user_mode = true;
		} else {
			cout << "Learning mode chosen" << endl;
			user_mode = false;
		}

		
		cout << "Do you want to run 2-toys [2] (planes/trucks) or 5-toys [5] ?" << endl;
		cin >> input;
		
		
		bool full_db = false;
		if (input[0] == '2') {
			nb_img = 2*(NB_IMAGES/5);
			nb_classes = 2;
		} else {
			nb_img = NB_IMAGES;
			cout << "Warning : only MLP is available with the 5-toys mode" << endl;
			nb_classes = 5;
			full_db = true;
		}
		
		cout << "Do you want to work with small [s], medium [m] or normal [n] image size ?"<<endl;
		cin >> input;
		
		if (input[0] == 'n') {
			small_img = false;
			medium_img = false;
			cout << "Working with full images (96px x 96px)" << endl;
			if (nb_classes == 5) {
				cout << "Warning : you need ~4GB RAM to run with 5-toys mode and full resolution !\nYou need also to run a 64bits version" << endl;
			}
		} else if (input[0] == 'm') {
			small_img = false;
			medium_img = true;
			cout << "Working with medium images (36px x 36px)" << endl;
		} else {
			cout << "Working with small images (24px x 24px)" << endl;
			small_img = true;
			medium_img = false;
		}
		
		int im_size;
		
		if (small_img) {
			im_size = size_s;
		} else {
			if (medium_img) {
				im_size = size_m;
			} else {
				im_size = size_n;
			}
		}
		
		nb_dim = im_size * im_size;
		
		srand ( unsigned ( time (NULL) ) );
		vector<int> rand_indexes_full;
		
		// set some values:
		for (int i=0; i<nb_img; i++) {
			rand_indexes_full.push_back(i);
		}
		
		// using built-in random generator:
		random_shuffle ( rand_indexes_full.begin(), rand_indexes_full.end() );
		
		// Allocating memory 
		targets_training_SVM = new float[nb_img];
		targets_testing_SVM = new float[nb_img];
		
		images_training = new double *[nb_img];
		for (int i = 0; i<nb_img; i++) {
			images_training[i]= new double[nb_dim];
		}
		
		images_testing = new double *[nb_img];
		for (int i = 0; i<nb_img; i++) {
			images_testing[i]= new double[nb_dim];
		}
		
		targets_training = new float*[nb_img];
		for (int i = 0; i<nb_img; i++) {
			targets_training[i] = new float[nb_classes];
		}
		
		targets_testing = new float*[nb_img];
		for (int i = 0; i<nb_img; i++) {
			targets_testing[i] = new float[nb_classes];
		}
		
		tab = new unsigned char *[nb_img];
		for (int i = 0; i<nb_img; i++) {
			tab[i]= new unsigned char[nb_dim];
		}
		
		// Reading files and preparing images and targets databases
		cout << "\nPrepare TRAINING data" << endl;
		prepare_data(images_training, targets_training, targets_training_SVM, DATA_TRAINING, full_db);
		
		cout << "\nPrepare TESTING data" << endl;
		prepare_data(images_testing, targets_testing, targets_testing_SVM, DATA_TESTING, full_db);
		
		if (user_mode) {
			bool end = false;
			nb_samples_training = nb_img;
			nb_samples_validation = 0;
			nb_samples_testing = nb_img - nb_samples_validation;
			
			rand_indexes_validation.clear();
			for (int i=0; i<nb_samples_validation; i++) {
				rand_indexes_validation.push_back(i);
			}
			
			nb_samples_testing = nb_img - nb_samples_validation;
			rand_indexes_testing.clear();
			for (int i=0; i<nb_samples_testing; i++) {
				rand_indexes_testing.push_back(i);
			}
			
			if (nb_classes == 5) {
				string path_file_mlp;
				if (small_img) {
					path_file_mlp = "opt_full_s_p_005_095_50.txt";
				} else {
					if (medium_img) {
						path_file_mlp = "opt_full_m_p_005_095_50.txt";
					} else {
						path_file_mlp = "opt_full_n_p_005_095_50.txt";
					}

				}
				run_mlp_using_file(path_file_mlp);
			} else {
				do {
					
					cout << "\nChoose between Multi Layer Perceptron [p] and Support Vector Machine [s]" << endl;
					cin >> input;
					
					if (input[0] == 'p') {
						string path_file_mlp;
						if (small_img) {
							path_file_mlp = "opt_s_p_005_095_50.txt";
						} else {
							if (medium_img) {
								path_file_mlp = "opt_m_p_005_095_50.txt";
							} else {
								path_file_mlp = "opt_n_p_005_095_50.txt";
							}
						}
						run_mlp_using_file(path_file_mlp);
					} else {
						string path_file_svm;
						if (small_img) {
							path_file_svm  = "opt_s_s_001_01.txt";
						} else {
							if (medium_img) {
								path_file_svm  = "opt_m_s_001_01.txt";

							} else {
								cout << "Parameters not computed sorry ..." << endl;
							}
						}
						run_svm_using_file(path_file_svm);
					}
					
					cout << "\n Do you want to continue [c], or exit [e] ?" << endl;
					cin >> input;
					
					if (input[0] == 'c') {
						end = false;
					} else {
						end = true;
					}
				} while(!end);
			}
			
			cout << "Releasing memory ..." ;
			cout << flush;
			
			// RELEASE MEMORY
			delete [] targets_training_SVM;
			delete [] targets_testing_SVM;
			
			for (int i = 0; i<nb_img; i++) {
				delete [] targets_training[i];
			}
			delete [] targets_training;
			
			for (int i = 0; i<nb_img; i++) {
				delete [] targets_testing[i];
			}
			delete [] targets_testing;
			
			for (int i = 0; i<nb_img; i++) {
				delete [] images_training[i];
			}
			delete [] images_training;
			
			for (int i = 0; i<nb_img; i++) {
				delete [] images_testing[i];
			}
			delete [] images_testing;
			
			for (int i = 0; i<nb_img; i++) {
				delete [] tab[i];
			}
			delete [] tab;
			
			cout << "ok" << endl;
			
			cout << "Ending nicely." << endl;
			return 0;
		}
		
		bool end = false;
		
		do {
			cout << "Choose how many images you want for training [100 - " << nb_img << "]" << endl;
			cin >> input;
			nb_samples_training = atoi(input); 
			
			cout << "Choose how many images you want for validation [100 - " << nb_img/2 << "]" << endl;
			cin >> input;
			nb_samples_validation = atoi(input);
			
			rand_indexes_validation.clear();
			for (int i=0; i<nb_samples_validation; i++) {
				rand_indexes_validation.push_back(i);
			}
			
			nb_samples_testing = nb_img - nb_samples_validation;
			rand_indexes_testing.clear();
			for (int i=0; i<nb_samples_testing; i++) {
				rand_indexes_testing.push_back(i);
			}
			
			if (nb_classes == 5) {
				cout << "Define the parameters for MLP" << endl;
				cout << "Learning rate : " << endl;
				cin >> input;
				float lr = atof(input);
				
				cout << "Momentum : " << endl;
				cin >> input;
				float mo = atof(input);
				
				cout << "Number of hidden neurons : " << endl;
				cin >> input;
				int nbh = atoi(input);
				
				cout << "Parameters choosed : learning rate " << lr << " momentum " << mo << " # hidden neurons " << nbh << endl;
				run_backprop(lr, mo, nbh);
			} else {
				cout << "\nChoose between Multi Layer Perceptron [p] and Support Vector Machine [s]" << endl;
				cin >> input;
				
				if (input[0] == 'p') {
					cout << "Define the parameters for MLP" << endl;
					cout << "Learning rate : " << endl;
					cin >> input;
					float lr = atof(input);
					
					cout << "Momentum : " << endl;
					cin >> input;
					float mo = atof(input);
					
					cout << "Number of hidden neurons : " << endl;
					cin >> input;
					int nbh = atoi(input);
					
					cout << "Parameters choosed : learning rate " << lr << " momentum " << mo << " # hidden neurons " << nbh << endl;
					run_backprop(lr, mo, nbh);
				} else {
					cout << "Define the parameters for SVM" << endl;
					cout << "Learning rate : " << endl;
					cin >> input;
					float lr = atof(input);
					
					cout << "Sigma : " << endl;
					cin >> input;
					float sigma = atof(input);
					
					cout << "Parameters choosed : learning rate " << lr << " sigma " << sigma << endl;
					run_SVM(10, targets_training_SVM, images_training, targets_testing_SVM, images_testing, lr, sigma);
				}
				
			}

			cout << "\n Do you want to continue [c], or exit [e] ?" << endl;
			cin >> input;
			
			if (input[0] == 'c') {
				end = false;
			} else {
				end = true;
			}
			
		} while (!end);
	} else {
		// s/m/n - [p - lr - mom - nbhid]/[s - lr - sigma] 
		
		if (argv[1][0] == 'n') {
			small_img = false;
			medium_img = false;
			cout << "Working with full images (96px x 96px)" << endl;
		} else if (argv[1][0] == 'm') {
			small_img = false;
			medium_img = true;
			cout << "Working with medium images (36px x 36px)" << endl;
		} else {
			cout << "Working with small images (24px x 24px)" << endl;
			small_img = true;
			medium_img = false;
		}
		
		int im_size;
		
		if (small_img) {
			im_size = size_s;
		} else {
			if (medium_img) {
				im_size = size_m;
			} else {
				im_size = size_n;
			}
		}
		nb_dim = im_size * im_size;
		
		bool full_db = false;
		
		int nb_img = 2*(NB_IMAGES/5);
		nb_classes = 2;
		
		nb_samples_training = nb_img; 
		nb_samples_validation = nb_img/2;
		
		srand ( unsigned ( time (NULL) ) );
		vector<int> rand_indexes_full;
		
		// set some values:
		for (int i=0; i<nb_img; i++) {
			rand_indexes_full.push_back(i);
		}
		
		// using built-in random generator:
		random_shuffle ( rand_indexes_full.begin(), rand_indexes_full.end() );
		
		// Allocating memory 
		targets_training_SVM = new float[nb_img];
		targets_testing_SVM = new float[nb_img];
		
		images_training = new double *[nb_img];
		for (int i = 0; i<nb_img; i++) {
			images_training[i]= new double[nb_dim];
		}
		
		images_testing = new double *[nb_img];
		for (int i = 0; i<nb_img; i++) {
			images_testing[i]= new double[nb_dim];
		}
		
		targets_training = new float*[nb_img];
		for (int i = 0; i<nb_img; i++) {
			targets_training[i] = new float[nb_classes];
		}
		
		targets_testing = new float*[nb_img];
		for (int i = 0; i<nb_img; i++) {
			targets_testing[i] = new float[nb_classes];
		}
		
		tab = new unsigned char *[nb_img];
		for (int i = 0; i<nb_img; i++) {
			tab[i]= new unsigned char[nb_dim];
		}
		
		// Reading files and preparing images and targets databases
		cout << "\nPrepare TRAINING data" << endl;
		prepare_data(images_training, targets_training, targets_training_SVM, DATA_TRAINING, full_db);
		
		cout << "\nPrepare TESTING data" << endl;
		prepare_data(images_testing, targets_testing, targets_testing_SVM, DATA_TESTING, full_db);
		
		rand_indexes_validation.clear();
		for (int i=0; i<nb_samples_validation; i++) {
			rand_indexes_validation.push_back(i);
		}
		
		nb_samples_testing = nb_img - nb_samples_validation;
		rand_indexes_testing.clear();
		for (int i=0; i<nb_samples_testing; i++) {
			rand_indexes_testing.push_back(i);
		}
		
		if (argv[2][0] == 'p') {
			float lr = atof(argv[3]);
			float mo = atof(argv[4]);
			int nbh = atoi(argv[5]);
			
			cout << "Parameters choosed : learning rate " << lr << " momentum " << mo << " # hidden neurons " << nbh << endl;
			run_backprop(lr, mo, nbh);
		} else {
			float lr = atof(argv[3]);
			float sigma = atof(argv[4]);
			
			cout << "Parameters choosed : learning rate " << lr << " sigma " << sigma << endl;
			run_SVM(10, targets_training_SVM, images_training, targets_testing_SVM, images_testing, lr, sigma);
		}
		return 0;
	}
	
	cout << "Releasing memory ..." ;
	cout << flush;
	
	// RELEASE MEMORY
	delete [] targets_training_SVM;
	delete [] targets_testing_SVM;
	
	for (int i = 0; i<nb_img; i++) {
		delete [] targets_training[i];
	}
	delete [] targets_training;
	
	for (int i = 0; i<nb_img; i++) {
		delete [] targets_testing[i];
	}
	delete [] targets_testing;
	
	for (int i = 0; i<nb_img; i++) {
		delete [] images_training[i];
	}
	delete [] images_training;
	
	for (int i = 0; i<nb_img; i++) {
		delete [] images_testing[i];
	}
	delete [] images_testing;
	
	for (int i = 0; i<nb_img; i++) {
		delete [] tab[i];
	}
	delete [] tab;
	
	cout << "ok" << endl;
	
	cout << "Ending nicely." << endl;
	
	return 1;
}
