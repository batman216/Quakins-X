#include "include/util.hpp"

using namespace std;

map<string, string>
read_box(ifstream& is, string box_name) {

  map<string, string> input_map;
	string              s, key, value;

	do { getline(is, s); }
	while (s.find(box_name)==string::npos);

	for (getline(is, s); s.find(box_name)==string::npos;
	  getline(is, s)) {

		auto equ = s.find('=');

		if (equ != string::npos) {
    	key = s.substr(0, equ - 1);
			key.erase(std::remove(key.begin(),
                     key.end(),' '),key.end());
    	value = s.substr(equ + 1, s.size());
                        value.erase(std::remove(value.begin(),
                          value.end(),' '),value.end());
    	input_map[key] = value;
		}
	}

    return input_map;
}
