#include "relative_error.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

void require(bool pass,const char *name) {
  if(!pass)throw std::runtime_error(name);
}
template<class F> void rejects(F fn,const char *name) {
  bool failed=false;
  try { fn(); } catch(const std::runtime_error &) { failed=true; }
  require(failed,name);
}

int main(int argc,char **argv) {
  try {
    const std::vector<double> b={0.,3.,0.,4.};
    auto e=error(b,b,2);
    require(valid(e)&&e.zero_reference_fields==1,"mixed zero reference");
    require(valid(error({0.,0.},{0.,0.},1)),"all zero reference");
    require(valid(error({-0.,0.},{0.,-0.},1)),"signed zero");
    auto a=b;a[0]=1e-300;e=error(a,b,2);
    require(!valid(e)&&e.zero_reference_mismatches==1,"tiny leakage in zero field");
    require(!valid(error({1.,0.},{0.,0.},1)),"nonzero output with zero reference");
    e=error({1.25},{1.},1);
    require(e.l2==.25&&e.linf==.25&&!valid(e),"nonzero relative error");
    require(valid(error({1.+1e-14},{1.},1)),"unchanged passing threshold");
    require(!valid(error({1.+1e-12},{1.},1)),"unchanged failing threshold");
    require(valid(error({1e-300,2e-300},{1e-300,2e-300},1)),"small nonzero reference");
    require(valid(error({1e300,2e300},{1e300,2e300},1)),"large finite reference");
    require(!valid(error({std::numeric_limits<double>::quiet_NaN()},{0.},1)),"nan rejected");
    require(!valid(error({0.},{std::numeric_limits<double>::infinity()},1)),"infinity rejected");
    rejects([]{error({1.},{1.},0);},"zero rhs rejected");
    rejects([]{error({1.},{1.},-1);},"negative rhs rejected");
    rejects([]{error({1.,2.},{1.},1);},"different sizes rejected");
    rejects([]{error({1.,2.,3.},{1.,2.,3.},2);},"incomplete field rejected");
    rejects([]{error({},{},1);},"empty reference rejected");
    std::cout<<"error_test: 17 cases passed\n";
    if(argc==2) {
      std::ifstream in(argv[1],std::ios::binary);
      int32_t h[7];
      require(bool(in.read(reinterpret_cast<char*>(h),sizeof(h)))&&h[0]==0x52534d31,"fixture header");
      int zero_fields=0;
      for(int p=0;p<2;++p) {
        const int cols=(h[2]-h[1]-p)/2+1;
        std::vector<double> x(size_t(h[4])*cols);
        require(bool(in.read(reinterpret_cast<char*>(x.data()),x.size()*sizeof(double))),"fixture input");
        auto measured=error(x,x,h[4]);
        require(valid(measured),"actual fixture identity");
        zero_fields+=measured.zero_reference_fields;
        in.seekg((int64_t(h[3])*cols+int64_t(h[5])*cols+int64_t(h[3])*h[4])*8,std::ios::cur);
        std::cout<<"fixture parity="<<p<<" zero_fields="<<measured.zero_reference_fields<<" pass\n";
      }
      require(zero_fields==3,"m64 regression must exercise the three zero fields");
    }
    return 0;
  } catch(const std::exception &e) {std::cerr<<"error_test: "<<e.what()<<'\n';return 1;}
}
