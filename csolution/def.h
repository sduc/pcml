/*
 *  def.h
 *  PCMLproject
 *
 *  Authors J. Paratte & D. Huwyler
 *
 */

#ifndef DEF_H
#define DEF_H

inline float min_float(float a , float b) {
	if (a > b) {
		return b;
	}
	return	a;
}

inline float max_float(float a, float b ) {
	if (a > b) {
		return a;
	}
	return	b;
}

inline double min_double(double a , double b) {
	if (a > b) {
		return b;
	}
	return	a;
}

inline double max_double(double a, double b ) {
	if (a > b) {
		return a;
	}
	return	b;
}


#endif