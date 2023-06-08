#pragma once

// static polymorphism via CRTP copied from
// https://gist.github.com/12ff54e/7643d7361d7221e4d3d0918ec3e193d6
// @Dr. Zhong Qi
template <class T, class...>
struct CRTP {
    T& self() { return static_cast<T&>(*this); }
    const T& self() const { return static_cast<const T&>(*this); }
};


