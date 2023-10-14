#pragma once

class gpuNumException : public std::exception {
  virtual const char* what() const throw()
    {
        return "#grids not divisible by #devices.";
    }
}gne;


class noBoxException : public std::exception {
  virtual const char* what() const throw()
    {
        return "Box name unknow.";
    }
}nbe;


