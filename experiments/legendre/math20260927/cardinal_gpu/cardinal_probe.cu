#include "legacy_map.hpp"
#include "layer_map.cuh"
#include <array>
#include <functional>

struct Cardinal {
  int m, degree, rows, rhs, count, storage, arena;
  std::array<int, 4> offsets;
  std::vector<double> a[2], forward[2], adjoint[2];
  std::vector<HostLayer> layers[2];
  explicit Cardinal(const char *path) {
    std::ifstream in(path, std::ios::binary);
    auto h = read_array<int>(in, 8);
    if (h[0] != 0x43464d31 || h[1] < 0 || h[1] > 1279 || h[2] != 1280 ||
        h[3] < 1 || h[3] > 1280 || h[4] != 242 || h[5] < 1 || h[5] > 1280 ||
        h[6] < h[5] || h[6] > 1280 || h[7] < 1 || h[7] > 100000)
      throw std::runtime_error("cardinal header");
    m=h[1]; degree=h[2]; rows=h[3]; rhs=h[4]; count=h[5]; storage=h[6]; arena=h[7];
    auto positions = read_array<int>(in, 4);
    std::copy(positions.begin(), positions.end(), offsets.begin());
    for (int p = 0; p < 2; ++p) {
      if (offsets[p] < 0 || offsets[p] + storage > arena || offsets[p+2] < 0 || offsets[p+2] + rows > arena)
        throw std::runtime_error("cardinal node offsets");
      int cols = (degree-m-p)/2+1;
      a[p] = read_array<double>(in, size_t(storage)*cols);
      forward[p] = read_array<double>(in, size_t(rows)*rhs);
      adjoint[p] = read_array<double>(in, size_t(cols)*rhs);
      for (auto *v : {&a[p], &forward[p], &adjoint[p]})
        for (double x : *v) if (!std::isfinite(x)) throw std::runtime_error("nonfinite cardinal data");
      for (int j = 0; j < cols; ++j)
        for (int k = count; k < storage; ++k)
          if (a[p][size_t(j)*storage+k] != 0.) throw std::runtime_error("anchor padding");
    }
    for (int trans = 0; trans < 2; ++trans) {
      std::vector<bool> ready(arena, false), written(arena, false);
      if (!trans) for (int p = 0; p < 2; ++p)
        std::fill(ready.begin()+offsets[p], ready.begin()+offsets[p]+storage, true);
      int n = read_array<int>(in, 1)[0];
      if (n != 8) throw std::runtime_error("layer count");
      for (int level = 0; level < n; ++level) {
        HostLayer layer;
        std::vector<int> writes, reads;
        int nt = read_array<int>(in, 1)[0];
        if (nt < 1 || nt > 2000) throw std::runtime_error("layer task count");
        for (int t = 0; t < nt; ++t) {
          auto d = read_array<int>(in, 3);
          if (d[0] < 0 || d[1] < 1 || d[1] > 2560 || d[0]+d[1] > arena || d[2] < 1 || d[2] > 10000)
            throw std::runtime_error("layer shape");
          auto ids = read_array<int>(in, d[2]);
          for (int id : ids) {
            if (id >= arena || id < -2*rows || (id < 0 && !trans) || (id >= 0 && !ready[id]))
              throw std::runtime_error("layer reads an unavailable row");
            if (id >= 0) reads.push_back(id);
          }
          auto weights = read_array<double>(in, size_t(d[1])*d[2]);
          for (double x : weights) if (!std::isfinite(x)) throw std::runtime_error("nonfinite weights");
          layer.tasks.push_back({d[0],d[1],d[2],int(layer.indices.size()),int(layer.weights.size())});
          layer.indices.insert(layer.indices.end(), ids.begin(), ids.end());
          layer.weights.insert(layer.weights.end(), weights.begin(), weights.end());
          for (int j = 0; j < d[1]; j += 8) layer.tiles.push_back({t,j});
          for (int j = d[0]; j < d[0]+d[1]; ++j) {
            if (written[j]) throw std::runtime_error("multiple output writers");
            written[j] = true;
            writes.push_back(j);
          }
        }
        // The same kernel must not overwrite any of its source rows.
        std::sort(writes.begin(), writes.end());
        for (int r : reads)
          if (std::binary_search(writes.begin(), writes.end(), r)) throw std::runtime_error("in-layer read/write hazard");
        for (int j : writes) ready[j] = true;
        layers[trans].push_back(std::move(layer));
      }
      for (int p = 0; p < 2; ++p) {
        const int start=offsets[p+(trans?0:2)], length=trans?count:rows;
        for (int j = start; j < start+length; ++j)
          if (!ready[j] || !written[j]) throw std::runtime_error("missing map output");
        if (trans) for (int j = offsets[p]+count; j < offsets[p]+storage; ++j)
          if (written[j]) throw std::runtime_error("map writes anchor padding");
      }
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing cardinal bytes");
  }
};

std::vector<double> concatenate(std::vector<double> a, const std::vector<double> &b) {
  a.insert(a.end(), b.begin(), b.end()); return a;
}

struct DeviceCardinal {
  const Cardinal &h;
  Buffer<double> work;
  std::unique_ptr<Buffer<double>> a[2], back[2];
  std::vector<std::unique_ptr<DeviceLayer>> layers[2];
  explicit DeviceCardinal(const Cardinal &host) : h(host), work(size_t(h.arena)*h.rhs) {
    CHECK(cudaMemset(work.ptr, 0, work.size*sizeof(double)));
    for (int p = 0; p < 2; ++p) {
      a[p] = std::make_unique<Buffer<double>>(h.a[p]);
      back[p] = std::make_unique<Buffer<double>>(size_t(h.rhs)*((h.degree-h.m-p)/2+1));
      for (const auto &l : h.layers[p]) layers[p].push_back(std::make_unique<DeviceLayer>(l));
    }
  }
  void map(bool trans, int mode, cudaStream_t stream, DenseReference &ref) {
    for (auto &l : layers[trans])
      l->apply(mode, stream, work.ptr, ref.part[0]->g.ptr, ref.part[1]->g.ptr, h.rhs, h.rows);
  }
  void anchor(bool trans, cublasHandle_t blas, DenseReference &ref) {
    const double one=1.,zero=0.;
    for (int p=0; p<2; ++p) {
      auto &d=*ref.part[p];
      double *sampled=work.ptr+size_t(h.offsets[p])*h.rhs;
      if (trans)
        CHECK(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,h.rhs,d.cols,h.storage,&one,
          sampled,h.rhs,a[p]->ptr,h.storage,&zero,back[p]->ptr,h.rhs));
      else
        CHECK(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,h.rhs,h.storage,d.cols,&one,
          d.x.ptr,h.rhs,a[p]->ptr,h.storage,&zero,sampled,h.rhs));
    }
  }
  std::vector<double> output(int p) const {
    std::vector<double> values(size_t(h.rows)*h.rhs);
    CHECK(cudaMemcpy(values.data(),work.ptr+size_t(h.offsets[p+2])*h.rhs,
      values.size()*sizeof(double),cudaMemcpyDeviceToHost));
    return values;
  }
};

void print_error(const char *label, const char *target, int m, const Error &f, const Error &a) {
  std::printf("{\"kind\":\"validation\",\"case\":\"%s\",\"reference\":\"%s\",\"m\":%d,"
    "\"forward_l2\":%.17g,\"forward_linf\":%.17g,\"adjoint_l2\":%.17g,\"adjoint_linf\":%.17g,\"pass\":%s}\n",
    label,target,m,f.l2,f.linf,a.l2,a.linf,valid(f)&&valid(a)?"true":"false");
  if (!valid(f)||!valid(a)) throw std::runtime_error("numerical gate failed");
}

void run_cardinal(const Fixture &f, const Cardinal &h) {
  if (f.h.m!=h.m || f.h.rows!=h.rows || f.h.rhs!=h.rhs) throw std::runtime_error("paired fixture mismatch");
  cudaStream_t stream;
  cublasHandle_t blas;
  CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  CHECK(cublasCreate(&blas)); CHECK(cublasSetStream(blas,stream));
  CHECK(cublasSetMathMode(blas,CUBLAS_DEFAULT_MATH));
  CHECK(cublasSetAtomicsMode(blas,CUBLAS_ATOMICS_NOT_ALLOWED));
  {
    DenseReference ref(f);
    DeviceCardinal device(h);
    HostTiles fp(f,32,false),rp(f,32,true);
    Buffer<double> a0(f.part[0].a),a1(f.part[1].a);
    Buffer<double> samples0(size_t(h.rhs)*f.h.samples),samples1(size_t(h.rhs)*f.h.samples);
    Buffer<double> back0(f.part[0].x.size()),back1(f.part[1].x.size());
    BatchedMap fm(fp,h.rhs,samples0.ptr,samples1.ptr),rm(rp,h.rhs,ref.part[0]->g.ptr,ref.part[1]->g.ptr);
    const double one=1.,zero=0.;
    auto dense=[&](bool trans) {
      for (int p=0;p<2;++p) {
        auto &d=*ref.part[p];
        CHECK(cublasDgemm(blas,CUBLAS_OP_N,trans?CUBLAS_OP_N:CUBLAS_OP_T,h.rhs,
          trans?d.cols:h.rows,trans?h.rows:d.cols,&one,trans?d.g.ptr:d.x.ptr,h.rhs,
          d.p.ptr,h.rows,&zero,trans?d.back.ptr:d.out.ptr,h.rhs));
      }
    };
    auto phase=[&](bool trans) {
      if (trans) rm.apply(blas);
      for (int p=0;p<2;++p) {
        auto &d=*ref.part[p];
        CHECK(cublasDgemm(blas,CUBLAS_OP_N,trans?CUBLAS_OP_N:CUBLAS_OP_T,h.rhs,
          trans?d.cols:f.h.samples,trans?f.h.samples:d.cols,&one,
          trans?(p?rm.odd.ptr:rm.even.ptr):d.x.ptr,h.rhs,p?a1.ptr:a0.ptr,f.h.samples,&zero,
          trans?(p?back1.ptr:back0.ptr):(p?samples1.ptr:samples0.ptr),h.rhs));
      }
      if (!trans) fm.apply(blas);
    };
    auto candidate=[&](bool trans,int mode) {
      if (!trans) device.anchor(false,blas,ref);
      device.map(trans,mode,stream,ref);
      if (trans) device.anchor(true,blas,ref);
    };
    dense(false); dense(true); CHECK(cudaStreamSynchronize(stream));
    auto exact_f=hemispheres(ref.part[0]->out.host(),ref.part[1]->out.host());
    auto exact_a=concatenate(ref.part[0]->back.host(),ref.part[1]->back.host());
    auto cpu_f=hemispheres(h.forward[0],h.forward[1]);
    auto cpu_a=concatenate(h.adjoint[0],h.adjoint[1]);
    constexpr int modes=6,phase_mode=5;
    const char *names[]={"simt","wmma8","wmma32","wmma32_n16","wmma32_n32","phase32"};
    auto validate=[&](int mode) {
      if (mode==phase_mode) { phase(false); phase(true); }
      else { candidate(false,mode); candidate(true,mode); }
      CHECK(cudaStreamSynchronize(stream));
      auto y=mode==phase_mode?hemispheres(fm.output(0),fm.output(1)):hemispheres(device.output(0),device.output(1));
      auto z=mode==phase_mode?concatenate(back0.host(),back1.host()):concatenate(device.back[0]->host(),device.back[1]->host());
      print_error(names[mode],"dense_gpu",h.m,error(y,exact_f,h.rhs),error(z,exact_a,h.rhs));
      if (mode<phase_mode) print_error(names[mode],"cpu_fmm",h.m,error(y,cpu_f,h.rhs),error(z,cpu_a,h.rhs));
    };
    for (int mode=0;mode<modes;++mode) {
      for (int i=0;i<3;++i) { if(mode==phase_mode){phase(false);phase(true);}else{candidate(false,mode);candidate(true,mode);} }
      validate(mode);
    }
    Graph df(stream,[&]{dense(false);}),da(stream,[&]{dense(true);});
    std::unique_ptr<Graph> forward[modes],adjoint[modes];
    for (int mode=0;mode<modes;++mode) {
      forward[mode]=std::make_unique<Graph>(stream,[&]{if(mode==phase_mode)phase(false);else candidate(false,mode);});
      adjoint[mode]=std::make_unique<Graph>(stream,[&]{if(mode==phase_mode)phase(true);else candidate(true,mode);});
    }
    for (int round=0;round<8;++round) for (int v=0;v<modes;++v) {
      int mode=(round+v)%modes;
      double fd,fc,ad,ac;
      if(round%2==0){fd=timing(df.exec,stream);fc=timing(forward[mode]->exec,stream);ad=timing(da.exec,stream);ac=timing(adjoint[mode]->exec,stream);}
      else{fc=timing(forward[mode]->exec,stream);fd=timing(df.exec,stream);ac=timing(adjoint[mode]->exec,stream);ad=timing(da.exec,stream);}
      std::printf("{\"kind\":\"operator_diagnostic\",\"case\":\"%s\",\"m\":%d,\"round\":%d,"
        "\"dense_forward_ms\":%.9g,\"candidate_forward_ms\":%.9g,\"dense_adjoint_ms\":%.9g,\"candidate_adjoint_ms\":%.9g}\n",
        names[mode],h.m,round+1,fd,fc,ad,ac);
    }
    // Repeated graph replay must not retain contributions from a prior invocation.
    for (int mode=0;mode<modes;++mode) validate(mode);
    for (int mode=0;mode<phase_mode;++mode) {
      Graph af(stream,[&]{device.anchor(false,blas,ref);}),aa(stream,[&]{device.anchor(true,blas,ref);});
      Graph mf(stream,[&]{device.map(false,mode,stream,ref);}),ma(stream,[&]{device.map(true,mode,stream,ref);});
      std::printf("{\"kind\":\"stages\",\"case\":\"%s\",\"m\":%d,\"anchor_forward_ms\":%.9g,"
        "\"map_forward_ms\":%.9g,\"map_adjoint_ms\":%.9g,\"anchor_adjoint_ms\":%.9g}\n",
        names[mode],h.m,timing(af.exec,stream),timing(mf.exec,stream),timing(ma.exec,stream),timing(aa.exec,stream));
    }
  }
  CHECK(cublasDestroy(blas)); CHECK(cudaStreamDestroy(stream));
}

void selftest() {
  int tests=0;
  const std::pair<int,int> shapes[]={{1,1},{7,3},{8,4},{9,17},{17,33},{37,80},{2,192}};
  for (int rhs : {1,7,8,9,31,32,33,242}) for (auto shape : shapes) {
    int n=shape.first,k=shape.second,rows=k,arena=k+n;
    HostLayer host;
    host.tasks.push_back({k,n,k,0,0});
    for(int j=0;j<n;j+=8)host.tiles.push_back({0,j});
    std::vector<double> initial(size_t(arena)*rhs,0.),g0(size_t(rows)*rhs),g1(g0.size()),exact(size_t(n)*rhs);
    for(size_t j=0;j<g0.size();++j){g0[j]=std::sin(double(j)*.31+.23);g1[j]=std::cos(double(j)*.19+.37);initial[j]=std::sin(double(j)*.73+.47);}
    for(int q=0;q<k;++q)host.indices.push_back(q%3==0?q:(q%3==1?-1-q:-1-rows-q));
    for(int j=0;j<n;++j)for(int q=0;q<k;++q)host.weights.push_back(std::cos(double(j*k+q)*.17));
    for(int j=0;j<n;++j)for(int f=0;f<rhs;++f){long double sum=0.;for(int q=0;q<k;++q){int id=host.indices[q];double x=id>=0?initial[size_t(id)*rhs+f]:(id>=-rows?g0[size_t(-1-id)*rhs+f]:g1[size_t(-1-id-rows)*rhs+f]);sum+=static_cast<long double>(host.weights[j*k+q])*x;}exact[size_t(j)*rhs+f]=double(sum);}
    DeviceLayer layer(host);
    Buffer<double> work(initial),d0(g0),d1(g1);
    for(int mode=0;mode<5;++mode){
      for(int replay=0;replay<3;++replay)layer.apply(mode,nullptr,work.ptr,d0.ptr,d1.ptr,rhs,rows);
      CHECK(cudaDeviceSynchronize());
      auto all=work.host();
      if(!std::equal(initial.begin(),initial.begin()+size_t(k)*rhs,all.begin()))throw std::runtime_error("selftest overwrote input");
      std::vector<double> actual(all.begin()+size_t(k)*rhs,all.end());
      if(!valid(error(actual,exact,rhs)))throw std::runtime_error("layer selftest failed");
      ++tests;
    }
  }
  std::printf("{\"kind\":\"selftest\",\"cases\":%d,\"pass\":true}\n",tests);
}

int main(int argc,char **argv) {
  std::setvbuf(stdout,nullptr,_IOLBF,0);
  try {
    if(argc==2 && std::string(argv[1])=="--selftest"){selftest();return 0;}
    bool inspect=argc>1 && std::string(argv[1])=="--inspect";
    int start=inspect?2:1;
    if(argc<=start || (argc-start)%2)throw std::runtime_error("expected base/cardinal pairs");
    for(int i=start;i<argc;i+=2){
      Fixture f(argv[i]); Cardinal h(argv[i+1]);
      if(f.h.m!=h.m || f.h.rows!=h.rows || f.h.rhs!=h.rhs)throw std::runtime_error("fixture pair mismatch");
      if(inspect)std::printf("fixture m=%d layers=%zu/%zu arena_mib=%.3f ok\n",h.m,h.layers[0].size(),h.layers[1].size(),double(h.arena)*h.rhs*8/1048576.);
      else run_cardinal(f,h);
    }
    return 0;
  } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
