#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <string>
#include <vector>

using Group = void (*)(int,int,char,char,int,const int*,const int*,double,
    const double*,int,const int64_t*,const double*,const int*,const int64_t*,
    double,double*,int,const int64_t*,int,intptr_t,void*);

namespace {
std::mutex mutex;
std::set<int> seen;
void check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr,"capture cuda: %s\n",cudaGetErrorString(e));
    std::abort();
  }
}
const char *rank() {
  const char *v=std::getenv("OMPI_COMM_WORLD_RANK");
  return v ? v : "0";
}
void capture(int id,char ta,char tb,int m,const int *n,const int *k,
    const double *b,const int *ldb,const int64_t *ob,int count,int lda,int ldc) {
  const char *root=std::getenv("D487_CAPTURE_DIR");
  if (!root || (id!=11 && id!=12 && id!=21 && id!=22)) return;
  std::lock_guard<std::mutex> lock(mutex);
  if (!seen.insert(id).second) return;
  check(cudaDeviceSynchronize());
  const std::string prefix=std::string(root)+"/rank"+rank()+"-blas"+std::to_string(id);
  std::ofstream out(prefix+".txt");
  out << "blas=" << id << " transa=" << ta << " transb=" << tb
      << " rhs=" << m << " lda=" << lda << " ldc=" << ldc << " groups=" << count << '\n';
  const int orders[]={0,1,64,65,224,225,256,257,288,289,384,512,640};
  for (int i=0;i<count;++i) {
    out << i << ' ' << n[i] << ' ' << k[i] << ' ' << ldb[i] << ' ' << ob[i] << '\n';
    if (n[i]<=0 || k[i]<=0) continue;
    const int cols=tb=='T' ? k[i] : n[i];
    bool wanted=false;
    for (int order:orders)
      if (cols==(1280-order)/2+1 || cols==(1279-order)/2+1) wanted=true;
    if (!wanted) continue;
    const int stored_cols=(cols+7)/8*8;
    std::vector<double> host(size_t(ldb[i])*stored_cols);
    check(cudaMemcpy(host.data(),b+ob[i],host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    std::ofstream data(prefix+"-group"+std::to_string(i)+".bin",std::ios::binary);
    int32_t header[]={id,i,m,n[i],k[i],ldb[i],stored_cols};
    data.write(reinterpret_cast<const char*>(header),sizeof(header));
    data.write(reinterpret_cast<const char*>(host.data()),host.size()*sizeof(double));
    if (!data) std::abort();
  }
  if (!out) std::abort();
  std::fprintf(stderr,"d487 capture rank=%s blas=%d rhs=%d groups=%d\n",rank(),id,m,count);
}
void dispatch(const char *name,int r,int id,char ta,char tb,int m,const int *n,
    const int *k,double alpha,const double *a,int lda,const int64_t *oa,
    const double *b,const int *ldb,const int64_t *ob,double beta,double *c,
    int ldc,const int64_t *oc,int count,intptr_t stream,void *allocator) {
  auto fn=reinterpret_cast<Group>(dlsym(RTLD_NEXT,name));
  if (!fn) { std::fprintf(stderr,"d487 symbol %s: %s\n",name,dlerror()); std::abort(); }
  capture(id,ta,tb,m,n,k,b,ldb,ob,count,lda,ldc);
  fn(r,id,ta,tb,m,n,k,alpha,a,lda,oa,b,ldb,ob,beta,c,ldc,oc,count,stream,allocator);
}
}

#define WRAPPER(name) \
extern "C" void name(int r,int id,char ta,char tb,int m,const int *n,const int *k, \
    double alpha,const double *a,int lda,const int64_t *oa,const double *b, \
    const int *ldb,const int64_t *ob,double beta,double *c,int ldc,const int64_t *oc, \
    int count,intptr_t stream,void *allocator) { \
  dispatch(#name,r,id,ta,tb,m,n,k,alpha,a,lda,oa,b,ldb,ob,beta,c,ldc,oc,count,stream,allocator); \
}
WRAPPER(hipblas_dgemm_wrapper_grouped)
WRAPPER(hipblas_dgemm_wrapper_grouped_async)
