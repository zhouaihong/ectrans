#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

struct Error {
  double l2=0.,linf=0.;
  bool finite=true;
  int zero_reference_fields=0,zero_reference_mismatches=0;
};

inline Error error(const std::vector<double> &a,const std::vector<double> &b,int rhs) {
  if(rhs<=0 || a.empty() || a.size()!=b.size() || a.size()%rhs)
    throw std::runtime_error("error shapes");
  Error out;
  for(int field=0;field<rhs;++field) {
    double peak=0.,actual_peak=0.;
    bool finite=true;
    for(size_t j=field;j<a.size();j+=rhs) {
      finite=finite && std::isfinite(a[j]) && std::isfinite(b[j]);
      peak=std::max(peak,std::abs(b[j]));
      actual_peak=std::max(actual_peak,std::abs(a[j]));
    }
    if(!finite) { out.finite=false;continue; }
    if(peak==0.) {
      // A zero input can legitimately produce an exactly zero parity output.
      ++out.zero_reference_fields;
      if(actual_peak!=0.)++out.zero_reference_mismatches;
      continue;
    }
    long double norm=0.,diff=0.,delta=0.;
    for(size_t j=field;j<a.size();j+=rhs) {
      const long double r=static_cast<long double>(b[j])/peak;
      const long double d=(static_cast<long double>(a[j])-b[j])/peak;
      norm+=r*r;diff+=d*d;delta=std::max(delta,std::abs(d));
    }
    double l2=static_cast<double>(std::sqrt(diff/norm));
    double linf=static_cast<double>(delta);
    if(!std::isfinite(l2)||!std::isfinite(linf)) { out.finite=false;continue; }
    out.l2=std::max(out.l2,l2);out.linf=std::max(out.linf,linf);
  }
  return out;
}

inline bool valid(const Error &e) {
  return e.finite && e.zero_reference_mismatches==0 && e.l2<=1e-13 && e.linf<=1e-13;
}
