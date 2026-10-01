#include "../math20260927/cardinal_gpu/legacy_map.hpp"
#include "../math20260925/wmma_gpu/map_wmma.cuh"
#include "block_transpose.hpp"
#include <chrono>
#include <functional>
#include <sstream>

extern "C" void *comparison_allocator_new();
extern "C" void comparison_allocator_delete(void*);
extern "C" void clean_gemm(int);
extern "C" void hipblas_dgemm_wrapper_grouped_async(int,int,char,char,int,
    const int*,const int*,double,const double*,int,const int64_t*,const double*,
    const int*,const int64_t*,double,double*,int,const int64_t*,int,intptr_t,void*);

int pad8(int n) { return (n+7)/8*8; }

// Call the archived library, including its padding, DAG and bucket selection.
struct Archived {
  struct Host {
    std::vector<int> n,k,ldb;
    std::vector<int64_t> ox,ob,og;
    std::vector<double> x,b,g;
  } host[2];
  struct Device {
    Buffer<double> x,b,g,y,z;
    explicit Device(const Host &h) : x(h.x),b(h.b),g(h.g),y(h.g.size()),z(h.x.size()) {}
  };
  std::unique_ptr<Device> device[2];
  const std::vector<Fixture> &fixtures;
  int rhs,ld,resol;
  void *allocator;
  cudaStream_t streams[2];
  cudaEvent_t ready,done[2];
  explicit Archived(const std::vector<Fixture> &f) : fixtures(f),rhs(f.at(0).h.rhs),
      ld(pad8(rhs)),resol(9001),allocator(comparison_allocator_new()) {
    for (int p=0;p<2;++p) {
      auto &h=host[p];
      for (const auto &v:f) {
        if(v.h.rhs!=rhs)throw std::runtime_error("mixed rhs");
        const auto &part=v.part[p];
        int rows=v.h.rows,cols=part.cols,n=pad8(rows),k=pad8(cols);
        h.n.push_back(rows);h.k.push_back(cols);h.ldb.push_back(n);
        h.ox.push_back(h.x.size());h.ob.push_back(h.b.size());h.og.push_back(h.g.size());
        h.x.resize(h.x.size()+size_t(ld)*k,0.);
        h.b.resize(h.b.size()+size_t(n)*k,0.);
        h.g.resize(h.g.size()+size_t(ld)*n,0.);
        for(int j=0;j<cols;++j) {
          std::copy_n(part.x.data()+size_t(j)*rhs,rhs,h.x.data()+h.ox.back()+size_t(j)*ld);
          std::copy_n(part.p.data()+size_t(j)*rows,rows,h.b.data()+h.ob.back()+size_t(j)*n);
        }
        for(int j=0;j<rows;++j)
          std::copy_n(part.g.data()+size_t(j)*rhs,rhs,h.g.data()+h.og.back()+size_t(j)*ld);
      }
      device[p]=std::make_unique<Device>(h);
      CHECK(cudaStreamCreateWithFlags(&streams[p],cudaStreamNonBlocking));
      CHECK(cudaEventCreateWithFlags(&done[p],cudaEventDisableTiming));
    }
    CHECK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
  }
  ~Archived() {
    cudaDeviceSynchronize();clean_gemm(resol);comparison_allocator_delete(allocator);
    for(int p=0;p<2;++p){cudaEventDestroy(done[p]);cudaStreamDestroy(streams[p]);}
    cudaEventDestroy(ready);
  }
  void apply(bool trans,cudaStream_t join) {
    CHECK(cudaEventRecord(ready,join));
    for(int p=0;p<2;++p) {
      auto &h=host[p];auto &d=*device[p];
      CHECK(cudaStreamWaitEvent(streams[p],ready));
      hipblas_dgemm_wrapper_grouped_async(resol,(trans?22:12)-p,'N',trans?'N':'T',rhs,
          trans?h.k.data():h.n.data(),trans?h.n.data():h.k.data(),1.,
          trans?d.g.ptr:d.x.ptr,ld,trans?h.og.data():h.ox.data(),d.b.ptr,h.ldb.data(),
          h.ob.data(),0.,trans?d.z.ptr:d.y.ptr,ld,trans?h.ox.data():h.og.data(),
          int(fixtures.size()),reinterpret_cast<intptr_t>(streams[p]),allocator);
      CHECK(cudaEventRecord(done[p],streams[p]));
      CHECK(cudaStreamWaitEvent(join,done[p]));
    }
  }
  std::vector<double> output(size_t i,int p,bool trans) const {
    int rows=trans?fixtures[i].part[p].cols:fixtures[i].h.rows;
    const auto &h=host[p];const auto &d=*device[p];
    std::vector<double> out(size_t(rhs)*rows);
    CHECK(cudaMemcpy2D(out.data(),rhs*sizeof(double),
      (trans?d.z.ptr:d.y.ptr)+(trans?h.ox[i]:h.og[i]),ld*sizeof(double),
      rhs*sizeof(double),rows,cudaMemcpyDeviceToHost));
    return out;
  }
};

struct Candidate {
  const Fixture &f;
  Part e,o;
  Buffer<double> padded_g0,padded_g1;
  Buffer<int> indices;
  HostTiles fp,rp;
  BatchedMap fm,rm;
  BlockTranspose block;
  Buffer<int> fs,rs;
  explicit Candidate(const Fixture &v):f(v),e(v.h,v.part[0]),o(v.h,v.part[1]),
    padded_g0(padded_grid(v.h,v.part[0])),padded_g1(padded_grid(v.h,v.part[1])),
    indices(v.indices),fp(v,32,false),rp(v,32,true),fm(fp,v.h.rhs,e.sampled.ptr,o.sampled.ptr),
    rm(rp,v.h.rhs,e.g.ptr,o.g.ptr),block(fm,padded_g0.ptr,padded_g1.ptr),fs(fp.starts),rs(rp.starts) {}
  void apply(bool trans,int mode,cublasHandle_t blas,cudaStream_t stream) {
    const auto &h=f.h;
    const double one=1.,zero=0.;
    if(trans) {
      if(mode==0) {
        for(Part *p:{&e,&o})gather_transpose<<<dim3((h.rhs+127)/128,h.samples),128,0,stream>>>(
          p->g.ptr,p->gathered.ptr,p->offsets.ptr,p->reverse_indices.ptr,p->reverse_w.ptr,h.rhs,h.samples);
      } else if(mode==1)rm.apply(blas);
      else if(mode==2)block.apply(blas,stream);
      else launch_map(mode==3?8:mode==4?16:32,stream,e.g.ptr,o.g.ptr,rm.weights.ptr,rs.ptr,
          rm.even.ptr,rm.odd.ptr,h.rhs,rp.span,rp.groups);
    }
    for(int parity=0;parity<2;++parity) {
      auto &p=parity?o:e;
      const double *in=trans?(mode==0?p.gathered.ptr:mode==2?(parity?block.odd.ptr:block.even.ptr):
          (parity?rm.odd.ptr:rm.even.ptr)):p.x.ptr;
      CHECK(cublasDgemm(blas,CUBLAS_OP_N,trans?CUBLAS_OP_N:CUBLAS_OP_T,h.rhs,
          trans?p.cols:h.samples,trans?h.samples:p.cols,&one,in,h.rhs,p.a.ptr,h.samples,
          &zero,trans?p.back.ptr:p.sampled.ptr,h.rhs));
    }
    if(!trans) {
      if(mode==0)interpolate<<<dim3((h.rhs+127)/128,h.rows),128,0,stream>>>(
          e.sampled.ptr,o.sampled.ptr,e.out.ptr,o.out.ptr,indices.ptr,e.w.ptr,o.w.ptr,h.rhs,h.rows,h.width);
      else if(mode<=2)fm.apply(blas);
      else launch_map(mode==3?8:mode==4?16:32,stream,e.sampled.ptr,o.sampled.ptr,fm.weights.ptr,fs.ptr,
          fm.even.ptr,fm.odd.ptr,h.rhs,fp.span,fp.groups);
    }
    CHECK(cudaGetLastError());
  }
  std::vector<double> output(int p,bool trans,int mode) {
    auto &d=p?o:e;
    if(trans)return d.back.host();
    if(mode==0)return d.out.host();
    return fm.output(p);
  }
};

struct Measurement {double gpu_ms,host_ms;};
Measurement measure(cudaStream_t stream,const std::function<void()> &op,int count=100) {
  cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));
  CHECK(cudaStreamSynchronize(stream));
  auto begin=std::chrono::steady_clock::now();
  CHECK(cudaEventRecord(a,stream));
  for(int i=0;i<count;++i)op();
  CHECK(cudaEventRecord(b,stream));CHECK(cudaEventSynchronize(b));
  double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/count;
  float gpu;CHECK(cudaEventElapsedTime(&gpu,a,b));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));
  return {gpu/count,host};
}
std::vector<double> join(std::vector<double> a,const std::vector<double> &b) {
  a.insert(a.end(),b.begin(),b.end());return a;
}

void run(const std::vector<Fixture> &f,const std::string &name,bool timed=true) {
  cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  cublasHandle_t blas;CHECK(cublasCreate(&blas));CHECK(cublasSetStream(blas,stream));
  CHECK(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH));
  CHECK(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));
  {
    Archived baseline(f);
    std::vector<std::unique_ptr<Candidate>> candidates;
    for(const auto &v:f)candidates.push_back(std::make_unique<Candidate>(v));
    for(int i=0;i<3;++i){baseline.apply(false,stream);baseline.apply(true,stream);}
    CHECK(cudaStreamSynchronize(stream));
    // The archived implementation must agree with ordinary fp64 GEMM first.
    for(size_t i=0;i<f.size();++i) {
      DenseReference dense(f[i]);const double one=1.,zero=0.;
      for(int p=0;p<2;++p)for(int trans=0;trans<2;++trans) {
        auto &d=*dense.part[p];const auto &h=f[i].h;
        CHECK(cublasDgemm(blas,CUBLAS_OP_N,trans?CUBLAS_OP_N:CUBLAS_OP_T,h.rhs,
          trans?d.cols:h.rows,trans?h.rows:d.cols,&one,trans?d.g.ptr:d.x.ptr,h.rhs,
          d.p.ptr,h.rows,&zero,trans?d.back.ptr:d.out.ptr,h.rhs));
        CHECK(cudaStreamSynchronize(stream));
        auto err=error(baseline.output(i,p,trans),trans?d.back.host():d.out.host(),h.rhs);
        if(!valid(err))throw std::runtime_error("archived library differs from fp64 reference");
      }
    }
    const char *names[]={"sparse","tile32","block-transpose","wmma8","wmma16","wmma32"};
    for(int mode=0;mode<6;++mode) {
      auto candidate=[&](bool trans){for(auto &c:candidates)c->apply(trans,mode,blas,stream);};
      candidate(false);candidate(true);CHECK(cudaStreamSynchronize(stream));
      bool passed=true;
      for(size_t i=0;i<f.size();++i) {
        auto &c=*candidates[i];
        auto ef=error(hemispheres(c.output(0,false,mode),c.output(1,false,mode)),
          hemispheres(baseline.output(i,0,false),baseline.output(i,1,false)),f[i].h.rhs);
        auto ea=error(join(c.output(0,true,mode),c.output(1,true,mode)),
          join(baseline.output(i,0,true),baseline.output(i,1,true)),f[i].h.rhs);
        bool ok=valid(ef)&&valid(ea);passed=passed&&ok;
        std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"mode\":\"%s\",\"m\":%d,\"groups\":%zu,"
          "\"forward_l2\":%.17g,\"forward_linf\":%.17g,\"adjoint_l2\":%.17g,\"adjoint_linf\":%.17g,\"pass\":%s}\n",
          name.c_str(),names[mode],f[i].h.m,f.size(),ef.l2,ef.linf,ea.l2,ea.linf,ok?"true":"false");
      }
      if(!passed || !timed)continue;
      Graph forward(stream,[&]{candidate(false);}),adjoint(stream,[&]{candidate(true);});
      for(int round=0;round<8;++round)for(int trans=0;trans<2;++trans) {
        auto base=[&]{baseline.apply(trans,stream);};
        auto trial=[&]{CHECK(cudaGraphLaunch(trans?adjoint.exec:forward.exec,stream));};
        Measurement a,b;
        if(round%2){b=measure(stream,trial);a=measure(stream,base);}
        else {a=measure(stream,base);b=measure(stream,trial);}
        std::printf("{\"kind\":\"comparison\",\"case\":\"%s\",\"mode\":\"%s\",\"groups\":%zu,"
          "\"direction\":\"%s\",\"repeat\":%d,\"d487_gpu_ms\":%.9g,\"candidate_gpu_ms\":%.9g,"
          "\"d487_host_ms\":%.9g,\"candidate_host_ms\":%.9g}\n",name.c_str(),names[mode],f.size(),
          trans?"adjoint":"forward",round+1,a.gpu_ms,b.gpu_ms,a.host_ms,b.host_ms);
      }
    }
  }
  CHECK(cublasDestroy(blas));CHECK(cudaStreamDestroy(stream));
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IOLBF,0);
  try {
    bool inspect=argc>1 && std::string(argv[1])=="--inspect";
    bool smoke=argc>1 && std::string(argv[1])=="--smoke";
    if(argc!=((inspect||smoke)?3:2))throw std::runtime_error("expected manifest");
    std::ifstream manifest(argv[(inspect||smoke)?2:1]);
    if(!manifest)throw std::runtime_error("missing manifest");
    std::string line;
    while(std::getline(manifest,line)) {
      if(line.empty()||line[0]=='#')continue;
      std::istringstream row(line);std::string name,path;
      row>>name;std::vector<Fixture> fixtures;
      while(row>>path)fixtures.emplace_back(path.c_str());
      if(fixtures.empty())throw std::runtime_error("empty case");
      std::printf("case=%s fixtures=%zu scope=archived-d487-selected-legendre-blocks\n",name.c_str(),fixtures.size());
      if(!inspect)run(fixtures,name,!smoke);
      if(smoke)break;
    }
    return 0;
  } catch(const std::exception &e) {std::fprintf(stderr,"compare: %s\n",e.what());return 1;}
}
