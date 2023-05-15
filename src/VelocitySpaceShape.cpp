#include "include/VelocitySpaceShape.hpp"

double SingleMaxwellian::operator()(double v) {

	return std::exp(-std::pow(v,2));
}

double DoubleMaxwellian::operator()(double v) {

	return v;
}

double SingleFermiDirac::operator()(double v) {

	return v;
}

double DoubleFermiDirac::operator()(double v) {

	return v;
}

double SingleMaxwellian::operator()(double v1, double v2) {

	return std::exp(-std::pow(v1,2)-std::pow(v2,2));
}

double DoubleMaxwellian::operator()(double v1, double v2) {

	return v1;
}

double SingleFermiDirac::operator()(double v1, double v2) {

	return v1;
}

double DoubleFermiDirac::operator()(double v1, double v2) {

	return v1;
}
