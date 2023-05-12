#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <string>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>


template <class A>
void assign(A& val, const std::string& name, 
    std::map<std::string, std::string> input) {
        
	std::stringstream buffer(input[name]);
	buffer >> val;
}

std::map<std::string, std::string> 
    read_box(std::ifstream& is,    
             std::string box_name);



#endif /* _UTIL_HPP_ */
