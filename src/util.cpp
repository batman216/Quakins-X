#include "include/util.hpp"

using namespace std;


vector<string> splitString(string str, const char split)
{
  istringstream iss(str);
  string token;
  vector<string> str_vec;
  while (getline(iss, token, split)) {
    str_vec.push_back(token);
  }
  return str_vec;
}

LinuxCommand::LinuxCommand() {}
LinuxCommand::LinuxCommand(int start, int intv, string command) 
  : start(start), intv(intv), command(command) {} 


unordered_map<string, LinuxCommand> 
readRuntimeCommand(ifstream& is) {

  unordered_map<string, LinuxCommand> command_list;

  string s, key, command;

  do { getline(is, s); }
  while (s.find("runtime")==string::npos);

  for (getline(is, s); s.find("runtime")==string::npos;
    getline(is, s)) {

    auto equ = s.find('=');

    if (equ != string::npos) {
      key = s.substr(0, equ - 1);
      key.erase(std::remove(key.begin(),
                key.end(),' '),key.end());
      command = s.substr(equ + 1, s.size());
      auto command_vec = splitString(command,',');
      
      assert(command_vec.size()==3);
      command_list[key] = LinuxCommand{std::stoi(command_vec[0]),
                                       std::stoi(command_vec[1]),
                                       command_vec[2]};
      std::cout << key <<"," << command_vec[2] << std::endl;
    }
  }

  return command_list;
}

map<string, string>
readBox(ifstream& is, string box_name) {

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

uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}
