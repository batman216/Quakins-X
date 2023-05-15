#ifndef _PHASESPACEINITIALIZATION_HPP_
#define _PHASESPACEINITIALIZATION_HPP_

#include <fstream>
#include "util.hpp"
#include "initialization.hpp"
#include "VelocitySpaceShape.hpp"

namespace quakins {

class PhaseSpaceInitialization {

public:
	template <typename idx_type, typename val_type, idx_type dim>
	PhaseSpaceInitialization(Parameters<idx_type,val_type,dim> *p) {

		ConcreteFactory<VelocitySpaceShape, SingleMaxwellian> sm("SingleMaxwellian");
		ConcreteFactory<VelocitySpaceShape, DoubleMaxwellian> dm("DoubleMaxwellian");
		ConcreteFactory<VelocitySpaceShape, SingleFermiDirac> sf("SingleFermiDirac");
		ConcreteFactory<VelocitySpaceShape, DoubleFermiDirac> df("DoubleFermiDirac");
	
		VelocitySpaceShape* f0 = Proxy<VelocitySpaceShape>::Instance().get(p->v_shape);
		std::cout << static_cast<val_type>(
		          (*f0)(static_cast<double>(1))) << std::endl;
	}

	template <typename it_type>
	void operator()(it_type begin_itor, it_type end_itor) {
		
		
	}

};

} // namespace quakins


#endif /* _PHASESPACEINITIALIZATION_HPP_ */


