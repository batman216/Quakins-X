#ifndef _VELOCITY_SPACE_SHAPE_HPP_
#define _VELOCITY_SPACE_SHAPE_HPP_

#include <cmath>
#include <vector>
#include <iostream>

// old-fasioned dynamic polymorphism, one can see 
// that the code looks ugly without template
class VelocitySpaceShape {

	std::vector<double> v;
public:
	virtual double operator()(double) = 0;
	virtual double operator()(double,double) = 0;

	virtual ~VelocitySpaceShape() {}
};

class SingleMaxwellian : public VelocitySpaceShape {
public:
	virtual double operator()(double) override;
	virtual double operator()(double,double) override;
};

class DoubleMaxwellian : public VelocitySpaceShape {
public:
	virtual double operator()(double) override;
	virtual double operator()(double,double) override;
};

class SingleFermiDirac : public VelocitySpaceShape {
public:
	virtual double operator()(double) override;
	virtual double operator()(double,double) override;
};

class DoubleFermiDirac : public VelocitySpaceShape {
public:
	virtual double operator()(double) override;
	virtual double operator()(double,double) override;
};


#endif /* _VELOCITY_SPACE_SHAPE_HPP_ */
