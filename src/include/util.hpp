#ifndef _UTIL_HPP_
#define _UTIL_HPP_
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <string>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>


// dynamic polyorphism
template <typename t_Product> 
class Factory {
public:
	virtual t_Product *produce() = 0;
protected:
	Factory() {}	
	virtual ~Factory() {}
};

/// forward declaration
template <typename t_Product> class Proxy; 
	
template <typename t_Product, typename t_ConcreteProduct> 
class ConcreteFactory : public Factory<t_Product> {
public:
	explicit ConcreteFactory(std::string p_name) {
		Proxy<t_Product>::Instance().registerProduct(this,p_name);
	}
	t_Product *produce() { return new t_ConcreteProduct; }

};

/**
 *   The class Proxy is like a flyweight factory
 */
template <typename t_Product>
class Proxy {
	Proxy() {}  /// private creator for singleton
	Proxy(const Proxy& other) {}  
public:
	/**
	 *  The client fetch/purchase concrete product by p_name
	 *  according to the regedit
	 */
	std::map<std::string, Factory<t_Product>*> regedit;
	
	static Proxy<t_Product>& Instance() {
		static Proxy<t_Product> instance;
		/// Meyers Singleton: create an instance only when this function is called.
		return instance;
	}
	void registerProduct(Factory<t_Product>* reg, std::string p_name) {
		regedit[p_name] = std::move(reg);
	}

	t_Product* get(std::string p_name) { /// flyweight singleton
		if (regedit.find(p_name) != regedit.end())
			return regedit[p_name]->produce();
			/// produce	
		else {
			std::cout << "no product named " << p_name 
					<< "registered." << std::endl;	
			return nullptr;
		}
	}
};



// assign value from string
template <class A>
void assign(A& val, const std::string& name, 
    std::map<std::string, std::string> input) {
        
	std::stringstream buffer(input[name]);
	buffer >> val;
}

std::map<std::string, std::string> 
    read_box(std::ifstream& is,    
             std::string box_name);

template <typename T,
          template<typename...> typename Container>
std::ostream& operator<<(std::ostream& os, const Container<T>& obj) {
	
	thrust::copy(obj.begin(), obj.end(),
	          std::ostream_iterator<T>(os,"	"));
	return os;
}


#endif /* _UTIL_HPP_ */
