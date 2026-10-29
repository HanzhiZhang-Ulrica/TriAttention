#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cfloat>
#include <cstdint>
#include <type_traits>
#include <algorithm>

namespace {
constexpr int BM=64, BN=64, D=128, NT=128;
constexpr int QS=144, VS=136;
constexpr int QBytes=BM*QS;
constexpr int KPairWords=4*4*2*32*2;
constexpr int K4Words=2*4*2*32*2;
constexpr int Q4S=80;
constexpr int V16Words=4*8*32*4;
constexpr int V8Words=16*32*4;
constexpr int V4Words=16*32*2;
constexpr int MaxTasks=2048;
constexpr int SharedBytes=QBytes+KPairWords*4+V16Words*4+MaxTasks*2;
static_assert(BM*VS*2<=QBytes+KPairWords*4+V16Words*4 && SharedBytes<=48*1024);
static_assert(BM*Q4S<=(V16Words-V8Words)*4);
constexpr int Magic=0x4B400000;
enum Pv : int {PvF16=0, PvI8=1, PvI4=2};

__device__ __forceinline__ void cp16(void* dst,const void* src,bool valid) {
  const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(a),"l"(src),"r"(valid?16:0));
}
__device__ __forceinline__ void cp4(void* dst,const void* src,bool valid) {
  const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" :: "r"(a),"l"(src),"r"(valid?4:0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;"); }
template<int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

__device__ __forceinline__ void imma8(int* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void imma4(int* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]) : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void hmma_pv(float* c,const uint32_t* a,const uint32_t* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%8,%9,%10,%11}, {%12,%13}, {%0,%1,%2,%3};\n"
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%14,%15}, {%4,%5,%6,%7};"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]));
}
__device__ __forceinline__ void rowsum(float* d,const uint32_t* p) {
  float u0,u1;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%8}, {%0,0.,%2,0.};"
      : "+f"(d[0]),"=f"(u0),"+f"(d[1]),"=f"(u1) : "r"(p[0]),"r"(p[1]),"r"(p[2]),"r"(p[3]),"r"(0x3c003c00));
}
__device__ __forceinline__ void imma_u8s8(int (&d)[4],const uint32_t (&a)[2][4],uint4 b) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
      : "r"(a[0][0]),"r"(a[0][1]),"r"(a[0][2]),"r"(a[0][3]),"r"(b.x),"r"(b.y),"r"(Magic));
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
      : "r"(a[1][0]),"r"(a[1][1]),"r"(a[1][2]),"r"(a[1][3]),"r"(b.z),"r"(b.w));
}
__device__ __forceinline__ void imma_u4s4(int (&d)[4],const uint32_t (&a)[4],uint2 b) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
      : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b.x),"r"(b.y),"r"(Magic));
}
__device__ __forceinline__ uint32_t pack_half2(float a,float b) {
  const half2 v=__floats2half2_rn(a,b);
  return *reinterpret_cast<const uint32_t*>(&v);
}

constexpr float PMagicHalf=49152.5f;
__device__ __forceinline__ float pmagic_dither(float score) {
  return __int_as_float((__float_as_int(score)&0xff)|0x47400000);
}

__device__ __forceinline__ uint32_t pack_bytes(uint32_t w0,uint32_t w1,uint32_t w2,uint32_t w3) {
  return __byte_perm(__byte_perm(w0,w1,0x0051),__byte_perm(w2,w3,0x0051),0x5410);
}
__device__ __forceinline__ uint32_t pack_nibbles(uint32_t lo,uint32_t hi) {
  return (lo&0x0f0f0f0fu)|((hi&0x0f0f0f0fu)<<4);
}
__device__ __forceinline__ int nibble_sum(uint32_t x) {
  return __dp4a((x&0x0f0f0f0fu)+((x>>4)&0x0f0f0f0fu),0x01010101u,0u);
}

__host__ __device__ __forceinline__ int key_in_col(int sub,int j) { return sub*2+(j<2?j:j+6); }

__device__ __constant__ uint32_t kRotationSigns[4]={0x9a3b5e71u,0x2c6d8f13u,0xe41b7a95u,0x5f02c3d8u};
__device__ __forceinline__ void rotate128(float (&x)[4],int lane) {
#pragma unroll
  for(int i=0;i<4;++i) {
    const int c=lane*4+i;
    if((kRotationSigns[c>>5]>>(c&31))&1u) x[i]=-x[i];
  }
  float a=x[0]+x[1],b=x[0]-x[1],c2=x[2]+x[3],d=x[2]-x[3];
  x[0]=a+c2;x[1]=b+d;x[2]=a-c2;x[3]=b-d;
#pragma unroll
  for(int bit=1;bit<32;bit<<=1) {
    const bool upper=lane&bit;
#pragma unroll
    for(int i=0;i<4;++i) {
      const float y=__shfl_xor_sync(0xffffffffu,x[i],bit);
      x[i]=upper?y-x[i]:x[i]+y;
    }
  }
#pragma unroll
  for(int i=0;i<4;++i) x[i]*=0.08838834764831845f;
}

__device__ __forceinline__ void stageq(uint8_t* dst,const int8_t* q8,int64_t head,int tokens,int qb,bool full) {
  const int tid=threadIdx.x&(NT-1);
#pragma unroll
  for(int i=tid;i<BM*D/16;i+=NT) {
    const int row=i/(D/16),col=(i%(D/16))*16;
    const bool valid=full || qb+row<tokens;
    cp16(dst+row*QS+col,reinterpret_cast<const uint8_t*>(q8)+(valid?(head*tokens+qb+row)*D+col:0),valid);
  }
  commit();wait<0>();__syncthreads();
}
__device__ __forceinline__ void stagek(uint8_t* dst,const uint32_t* k,int64_t head,int tokens,int key) {
  const int tid=threadIdx.x&(NT-1);
  const uint32_t* source=k+(head*((tokens+63)/64)+key/64)*KPairWords;
#pragma unroll
  for(int i=tid;i<KPairWords/4;i+=NT) cp16(dst+i*16,source+i*4,true);
  commit();
}
__device__ __forceinline__ void stageq4(uint8_t* dst,const uint8_t* q4,int64_t head,int tokens,int qb,bool full) {
  const int tid=threadIdx.x&(NT-1);
#pragma unroll
  for(int i=tid;i<BM*4;i+=NT) {
    const int row=i/4,col=(i%4)*16;
    const bool valid=full || qb+row<tokens;
    cp16(dst+row*Q4S+col,q4+(valid?(head*tokens+qb+row)*(D/2)+col:0),valid);
  }
  commit();wait<0>();__syncthreads();
}
__device__ __forceinline__ void stagek4(uint8_t* dst,const uint32_t* k,int64_t head,int tokens,int key) {
  const int tid=threadIdx.x&(NT-1);
  const uint32_t* source=k+(head*((tokens+63)/64)+key/64)*K4Words;
#pragma unroll
  for(int i=tid;i<K4Words/4;i+=NT) cp16(dst+i*16,source+i*4,true);
  commit();
}
template<int Words> __device__ __forceinline__ void stagev(uint32_t* dst,const uint32_t* source) {
  const int tid=threadIdx.x&(NT-1);
#pragma unroll
  for(int i=tid;i<Words/4;i+=NT) cp16(dst+i*4,source+i*4,true);
  commit();
}

template<bool Masked>
__device__ __forceinline__ void qk8(const uint8_t* q,const uint8_t* k,int tokens,int query_begin,
    int key_begin,float (&t)[4][8]) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,sub=lane%4;
  const int row0=warp*16+group,row1=row0+8;
  int c[4][8];
#pragma unroll
  for(int col=0;col<4;++col)
#pragma unroll
    for(int j=0;j<8;++j) c[col][j]=Magic;
#pragma unroll
  for(int feature=0;feature<D;feature+=32) {
    const int byte=feature+sub*4;
    const uint32_t a0=*reinterpret_cast<const uint32_t*>(q+row0*QS+byte);
    const uint32_t a1=*reinterpret_cast<const uint32_t*>(q+row1*QS+byte);
    const uint32_t a2=*reinterpret_cast<const uint32_t*>(q+row0*QS+byte+16);
    const uint32_t a3=*reinterpret_cast<const uint32_t*>(q+row1*QS+byte+16);
#pragma unroll
    for(int col=0;col<4;++col) {
      const uint2* pair=reinterpret_cast<const uint2*>(k)+((feature/32*4+col)*2*32+lane);
      const uint2 b01=pair[0],b23=pair[32];
      imma8(c[col],a0,a1,a2,a3,b01.x,b01.y);
      imma8(c[col]+4,a0,a1,a2,a3,b23.x,b23.y);
    }
  }
#pragma unroll
  for(int col=0;col<4;++col)
#pragma unroll
    for(int j=0;j<8;++j) {
      float value=__int_as_float(c[col][j]);
      if constexpr(Masked) {
        const int row=(j%4<2)?row0:row1;
        const int key=key_begin+col*16+(j>=4?8:0)+sub*2+(j%2);
        if(key>=tokens || key>query_begin+row || query_begin+row>=tokens) value=-CUDART_INF_F;
      }
      t[col][j]=value;
    }
}

template<bool Masked>
__device__ __forceinline__ void qk4(const uint8_t* q,const uint8_t* k,int tokens,int query_begin,
    int key_begin,float (&t)[4][8]) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,sub=lane%4;
  const int row0=warp*16+group,row1=row0+8;
  int c[4][8];
#pragma unroll
  for(int col=0;col<4;++col)
#pragma unroll
    for(int j=0;j<8;++j) c[col][j]=Magic;
#pragma unroll
  for(int step=0;step<2;++step) {
    const int byte=step*32+sub*4;
    const uint32_t a0=*reinterpret_cast<const uint32_t*>(q+row0*Q4S+byte);
    const uint32_t a1=*reinterpret_cast<const uint32_t*>(q+row1*Q4S+byte);
    const uint32_t a2=*reinterpret_cast<const uint32_t*>(q+row0*Q4S+byte+16);
    const uint32_t a3=*reinterpret_cast<const uint32_t*>(q+row1*Q4S+byte+16);
#pragma unroll
    for(int col=0;col<4;++col) {
      const uint2* pair=reinterpret_cast<const uint2*>(k)+((step*4+col)*2*32+lane);
      const uint2 b01=pair[0],b23=pair[32];
      imma4(c[col],a0,a1,a2,a3,b01.x,b01.y);
      imma4(c[col]+4,a0,a1,a2,a3,b23.x,b23.y);
    }
  }
#pragma unroll
  for(int col=0;col<4;++col)
#pragma unroll
    for(int j=0;j<8;++j) {
      float value=__int_as_float(c[col][j]);
      if constexpr(Masked) {
        const int row=(j%4<2)?row0:row1;
        const int key=key_begin+col*16+(j>=4?8:0)+sub*2+(j%2);
        if(key>=tokens || key>query_begin+row || query_begin+row>=tokens) value=-CUDART_INF_F;
      }
      t[col][j]=value;
    }
}
__device__ __forceinline__ float row_max(const float (&t)[4][8],int r) {
  float m=-CUDART_INF_F;
#pragma unroll
  for(int col=0;col<4;++col)
    m=fmaxf(m,fmaxf(fmaxf(t[col][2*r],t[col][2*r+1]),fmaxf(t[col][2*r+4],t[col][2*r+5])));
  m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,2));
  return fmaxf(m,__shfl_xor_sync(0xffffffffu,m,1));
}

__device__ __forceinline__ float scaled_row_max(const float (&t)[4][8],int r,float qs,const float (&ks)[4]) {
  float m=-CUDART_INF_F;
#pragma unroll
  for(int col=0;col<4;++col) {
    const float x=fmaxf(fmaxf(t[col][2*r],t[col][2*r+1]),fmaxf(t[col][2*r+4],t[col][2*r+5]));
    if(x>-CUDART_INF_F) m=fmaxf(m,qs*ks[col]*__fsub_rn(x,12582912.f));
  }
  m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,2));
  return fmaxf(m,__shfl_xor_sync(0xffffffffu,m,1));
}

__device__ __forceinline__ float tile_base_margin(float slope,float base) {
  return fmaf(slope,0.75f,fabsf(base)*2.4e-7f);
}

template<typename Out,bool Full>
__device__ __forceinline__ void run_task(const int32_t* bands,const int32_t* entries,const int32_t* groups,
    const uint32_t* v16,const uint32_t* v8,const uint32_t* v4,const float* vstat,
    const int8_t* q8,const uint32_t* k8,const float* qa8,const float* ka8,
    const uint8_t* q4,const uint32_t* k4,const float* qa4,const float* ka4,Out* output,
    int heads_q,int heads_kv,int tokens,int qb,int qhead,int64_t osh,int64_t ost,float sm_log2) {
  const int query_begin=qb*BM,kvhead=qhead/(heads_q/heads_kv),tiles=(tokens+63)/64;
  const int64_t qh=qhead,kh=kvhead;
  extern __shared__ __align__(16) uint8_t storage[];
  uint8_t* qt=storage;
  uint8_t* kt=qt+QBytes;
  uint32_t* vt=reinterpret_cast<uint32_t*>(kt+KPairWords*4);
  uint16_t* cols=reinterpret_cast<uint16_t*>(vt+V16Words);
  uint8_t* q4t=reinterpret_cast<uint8_t*>(vt+V8Words);
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,sub=lane%4;
  const int row0=warp*16+lane/4,row1=row0+8;
  float out[16][4]={};

  float m[2]={-FLT_MAX,-FLT_MAX},denom[2]={},corr[2]={};
  int g[5];
  if(bands!=nullptr) {
    const int w8=bands[qhead*2],w4=bands[qhead*2+1];
    const int n16=qb>0?2:1;
    const int lo8=max(1,qb-w8+1),n8=max(0,qb-lo8);
    const int lo4=max(1,qb-w4+1),n4=max(0,qb-w8-lo4+1);
    for(int i=threadIdx.x;i<n16+n8+n4;i+=NT)
      cols[i]=i<n16?(i==0?0:qb):i<n16+n8?lo8+i-n16:lo4+i-n16-n8;
    g[0]=0;g[1]=n16;g[2]=g[3]=n16+n8;g[4]=n16+n8+n4;
  } else {
    const int* t=groups+(qhead*((tokens+63)/64)+qb)*4;
#pragma unroll
    for(int i=0;i<5;++i) g[i]=t[i];
    for(int i=threadIdx.x;i<g[4]-g[0];i+=NT) cols[i]=entries[g[0]+i];
  }
  const int first=g[0];
  stageq(qt,q8,qh,tokens,query_begin,Full);
  float qs[2];

  qs[0]=(Full || query_begin+row0<tokens)?qa8[qh*tokens+query_begin+row0]*sm_log2:FLT_MIN;
  qs[1]=(Full || query_begin+row1<tokens)?qa8[qh*tokens+query_begin+row1]*sm_log2:FLT_MIN;
  const float* kscale8=ka8+kh*tiles*4;
  int pv8_done=0;

  auto pass=[&](auto qk_tag,auto pv_tag,int begin,int end) {
  constexpr int qkbits=decltype(qk_tag)::value;
  constexpr int pv=decltype(pv_tag)::value;
  if(begin==end) return;
  float qsp[2];
  if constexpr(qkbits==8) {
    qsp[0]=qs[0];qsp[1]=qs[1];
  } else {
    stageq4(q4t,q4,qh,tokens,query_begin,Full);
    qsp[0]=(Full || query_begin+row0<tokens)?qa4[qh*tokens+query_begin+row0]*sm_log2:FLT_MIN;
    qsp[1]=(Full || query_begin+row1<tokens)?qa4[qh*tokens+query_begin+row1]*sm_log2:FLT_MIN;
  }
  const float* kscale=qkbits==8?kscale8:ka4+kh*tiles*4;
  auto stage_k=[&](int c) { if constexpr(qkbits==8) stagek(kt,k8,kh,tokens,c*BN); else stagek4(kt,k4,kh,tokens,c*BN); };

  auto exparg=[&](float slope,float x,float b) {
    if constexpr(qkbits==4) return fmaf(slope,__fsub_rn(x,12582912.f),-b);
    else return fmaf(slope,x,fmaf(-slope,12582912.f,-b));
  };
  constexpr int words=pv==PvF16?V16Words:pv==PvI8?V8Words:V4Words;
  const uint32_t* vh=(pv==PvF16?v16:pv==PvI8?v8:v4)+kh*tiles*words;
  auto stage_v=[&](int col) { stagev<words>(vt,vh+col*words); };
  int col=cols[begin];
  stage_k(col);
  stage_v(col);
  wait<1>();__syncthreads();
  auto iteration=[&](auto masked_tag,bool has_next,int next,bool flush) {
    constexpr bool last=decltype(masked_tag)::value;
    const int key=col*BN;
    float t[4][8];
    if constexpr(qkbits==8) qk8<last>(qt,kt,tokens,query_begin,key,t);
    else qk4<last>(q4t,kt,tokens,query_begin,key,t);

    float tile[2],alpha[2];
    float ks[4];
#pragma unroll
    for(int c=0;c<4;++c) ks[c]=__ldg(kscale+col*4+c);
#pragma unroll
    for(int r=0;r<2;++r) {
      const float mx=scaled_row_max(t,r,qsp[r],ks);
      tile[r]=mx;
      const float n=fmaxf(m[r],tile[r]);
      alpha[r]=exp2f(m[r]-n);
      m[r]=n;
    }
    wait<0>();__syncthreads();
    if(has_next) stage_k(next);

    if(flush || !__all_sync(0xffffffffu,alpha[0]==1.f && alpha[1]==1.f)) {
      const float shift[2]={-corr[0]*alpha[0],-corr[1]*alpha[1]};
#pragma unroll
      for(int c=0;c<16;++c)
#pragma unroll
        for(int j=0;j<4;++j) out[c][j]=fmaf(out[c][j],alpha[j/2],shift[j/2]);
      corr[0]=corr[1]=0.f;
    }
    if constexpr(pv==PvF16) {

      denom[0]*=alpha[0];denom[1]*=alpha[1];
      uint32_t prob[4][4];
#pragma unroll
      for(int col=0;col<4;++col) {
        float e[8];
#pragma unroll
        for(int j=0;j<8;++j) {
          const int r=(j%4<2)?0:1;
          const float slope=qsp[r]*ks[col];
          e[j]=exp2f(exparg(slope,t[col][j],m[r]));
        }
        prob[col][0]=pack_half2(e[0],e[1]);prob[col][1]=pack_half2(e[2],e[3]);
        prob[col][2]=pack_half2(e[4],e[5]);prob[col][3]=pack_half2(e[6],e[7]);
      }
#pragma unroll
      for(int key16=0;key16<4;++key16) {
        rowsum(denom,prob[key16]);
#pragma unroll
        for(int col16=0;col16<8;++col16) {
          const uint4 b=*reinterpret_cast<const uint4*>(vt+((key16*8+col16)*32+lane)*4);
          const uint32_t bw[4]={b.x,b.y,b.z,b.w};
          hmma_pv(&out[col16*2][0],prob[key16],bw);
        }
      }
    } else {

      float base[2],bq_den[2],bq_num[2];
      constexpr float levels=pv==PvI8?255.f:15.f;
      const float ks_max=fmaxf(fmaxf(ks[0],ks[1]),fmaxf(ks[2],ks[3]));
#pragma unroll
      for(int r=0;r<2;++r) {
        base[r]=tile[r]>-CUDART_INF_F?tile[r]:m[r];
        base[r]+=tile_base_margin(qsp[r]*ks_max,base[r]);
        bq_den[r]=exp2f(base[r]-m[r])/levels;
        bq_num[r]=pv==PvI8?bq_den[r]:bq_den[r]*(127.f/7.f);
      }

      int rs[2];
      auto quantize=[&](int col,uint32_t (&w)[8]) {
#pragma unroll
        for(int j=0;j<8;++j)
          w[j]=__float_as_uint(fmaf(exp2f(exparg(qsp[(j%4<2)?0:1]*ks[col],t[col][j],base[(j%4<2)?0:1])),levels,
              PMagicHalf));
      };
      if constexpr(pv==PvI8) {
        uint32_t a[2][4];
#pragma unroll
        for(int col=0;col<4;++col) {
          uint32_t w[8];
          quantize(col,w);
          a[col/2][(col%2)*2]=pack_bytes(w[0],w[1],w[4],w[5]);
          a[col/2][(col%2)*2+1]=pack_bytes(w[2],w[3],w[6],w[7]);
        }
        rs[0]=__dp4a(a[0][0],0x01010101u,__dp4a(a[0][2],0x01010101u,__dp4a(a[1][0],0x01010101u,__dp4a(a[1][2],0x01010101u,0u))));
        rs[1]=__dp4a(a[0][1],0x01010101u,__dp4a(a[0][3],0x01010101u,__dp4a(a[1][1],0x01010101u,__dp4a(a[1][3],0x01010101u,0u))));
#pragma unroll
        for(int n8=0;n8<16;++n8) {
          int d[4];
          imma_u8s8(d,a,*reinterpret_cast<const uint4*>(vt+(n8*32+lane)*4));
#pragma unroll
          for(int j=0;j<4;++j) out[n8][j]=fmaf(__int_as_float(d[j]),bq_num[j/2],out[n8][j]);
        }
      } else {

        uint32_t a[4];
#pragma unroll
        for(int h=0;h<2;++h) {
          uint32_t lo[8],hi[8];
          quantize(2*h,lo);
          quantize(2*h+1,hi);
          a[h*2]=pack_nibbles(pack_bytes(lo[0],lo[1],lo[4],lo[5]),pack_bytes(hi[0],hi[1],hi[4],hi[5]));
          a[h*2+1]=pack_nibbles(pack_bytes(lo[2],lo[3],lo[6],lo[7]),pack_bytes(hi[2],hi[3],hi[6],hi[7]));
        }
        rs[0]=nibble_sum(a[0])+nibble_sum(a[2]);
        rs[1]=nibble_sum(a[1])+nibble_sum(a[3]);
#pragma unroll
        for(int n8=0;n8<16;++n8) {
          int d[4];
          imma_u4s4(d,a,*reinterpret_cast<const uint2*>(vt+(n8*32+lane)*2));
#pragma unroll
          for(int j=0;j<4;++j) out[n8][j]=fmaf(__fsub_rn(__int_as_float(d[j]),12582912.f),bq_num[j/2],out[n8][j]);
        }
      }
#pragma unroll
      for(int r=0;r<2;++r) {
        rs[r]+=__shfl_xor_sync(0xffffffffu,rs[r],1);
        rs[r]+=__shfl_xor_sync(0xffffffffu,rs[r],2);
        denom[r]=fmaf(static_cast<float>(rs[r]),bq_den[r],denom[r]*alpha[r]);
        if constexpr(pv==PvI8) corr[r]=fmaf(12582912.f,bq_num[r],corr[r]);
      }
    }
    wait<0>();__syncthreads();
    if(has_next) stage_v(next);
    col=next;
  };
  for(int i=begin;i<end-1;++i) iteration(std::false_type{},true,cols[i+1],pv==PvI8 && (pv8_done+i-begin)%16==15);
  if(cols[end-1]==qb) iteration(std::true_type{},false,0,false);
  else iteration(std::false_type{},false,0,false);
  if(pv==PvI8) pv8_done+=end-begin;
  };
  using Q8=std::integral_constant<int,8>;
  using Q4=std::integral_constant<int,4>;
  pass(Q8{},std::integral_constant<int,PvF16>{},0,g[1]-first);
  pass(Q8{},std::integral_constant<int,PvI8>{},g[1]-first,g[2]-first);
  pass(Q8{},std::integral_constant<int,PvI4>{},g[2]-first,g[3]-first);
  pass(Q4{},std::integral_constant<int,PvI8>{},g[3]-first,g[4]-first);
  __syncthreads();
  Out* ot=reinterpret_cast<Out*>(storage);
  const float inv[2]={denom[0]>0.f?1.f/denom[0]:0.f,denom[1]>0.f?1.f/denom[1]:0.f};
  const float* mean=vstat+kh*3*D;
  const float* scale=mean+D;
#pragma unroll
  for(int n8=0;n8<16;++n8) {
    const int f=n8*8+sub*2;
#pragma unroll
    for(int r=0;r<2;++r) {
      const Out vals[2]={static_cast<Out>(fmaf((out[n8][r*2]-corr[r])*inv[r],scale[f],mean[f])),
                         static_cast<Out>(fmaf((out[n8][r*2+1]-corr[r])*inv[r],scale[f+1],mean[f+1]))};
      *reinterpret_cast<uint32_t*>(ot+(r?row1:row0)*VS+f)=*reinterpret_cast<const uint32_t*>(vals);
    }
  }
  __syncthreads();
  for(int i=threadIdx.x;i<BM*D/8;i+=NT) {
    const int row=i/(D/8),col=(i%(D/8))*8;
    if(Full || query_begin+row<tokens)
      *reinterpret_cast<uint4*>(output+(int64_t)qhead*osh+(int64_t)(query_begin+row)*ost+col)=
          *reinterpret_cast<const uint4*>(ot+row*VS+col);
  }
}

template<typename Out>
__global__ __maxnreg__(128) void unified_kernel(const int32_t* order,const int32_t* bands,const int32_t* groups,
    const int32_t* entries,
    const uint32_t* v16,const uint32_t* v8,const uint32_t* v4,const float* vstat,
    const int8_t* q8,const uint32_t* k8,const float* qa8,const float* ka8,
    const uint8_t* q4,const uint32_t* k4,const float* qa4,const float* ka4,Out* output,
    int heads_q,int heads_kv,int tokens,int64_t osh,int64_t ost,float sm_log2) {
  const int task=order[blockIdx.x];
  const int rows=(tokens+63)/64,qhead=task/rows,qb=task%rows;
  if((qb+1)*BM<=tokens)
    run_task<Out,true>(bands,entries,groups,v16,v8,v4,vstat,q8,k8,qa8,ka8,q4,k4,qa4,ka4,output,heads_q,heads_kv,tokens,qb,qhead,
        osh,ost,sm_log2);
  else
    run_task<Out,false>(bands,entries,groups,v16,v8,v4,vstat,q8,k8,qa8,ka8,q4,k4,qa4,ka4,output,heads_q,heads_kv,tokens,qb,qhead,
        osh,ost,sm_log2);
}

#include "common_device.cuh"
}

torch::Tensor attention(torch::Tensor order,c10::optional<torch::Tensor> bands,c10::optional<torch::Tensor> groups,
    c10::optional<torch::Tensor> entries,
    c10::optional<torch::Tensor> v16p,c10::optional<torch::Tensor> v8p,c10::optional<torch::Tensor> v4p,
    torch::Tensor vstat,torch::Tensor q8,torch::Tensor k8p,torch::Tensor qa8,torch::Tensor ka8,
    c10::optional<torch::Tensor> q4,c10::optional<torch::Tensor> k4p,c10::optional<torch::Tensor> qa4,
    c10::optional<torch::Tensor> ka4,torch::Tensor out,double scale) {
  TORCH_CHECK(q8.scalar_type()==at::kChar && q8.dim()==4 && q8.size(0)==1 && q8.size(3)==D && q8.is_contiguous(),"Q8 [1,Hq,T,128]");
  const int h=q8.size(1),t=q8.size(2),hk=vstat.size(1);
  TORCH_CHECK((t+63)/64<=MaxTasks,"at most 131072 tokens");
  TORCH_CHECK(h%hk==0 && order.scalar_type()==at::kInt && order.numel()==(int64_t)h*((t+63)/64),"CTA order");
  TORCH_CHECK(bands.has_value()!=groups.has_value() && groups.has_value()==entries.has_value() &&
      (!bands.has_value() || (bands->scalar_type()==at::kInt && bands->is_contiguous() && bands->numel()==(int64_t)h*2)) &&
      (!groups.has_value() || (groups->scalar_type()==at::kInt && entries->scalar_type()==at::kInt &&
                               groups->numel()==(int64_t)h*((t+63)/64)*4+1)),"bands [Hq,2] or task tables");
  TORCH_CHECK(q4.has_value()==k4p.has_value() && q4.has_value()==qa4.has_value() && q4.has_value()==ka4.has_value(),
      "INT4 Q/K operands (all or none)");
  TORCH_CHECK(out.dim()==4 && out.size(0)==1 && out.stride(3)==1 && out.stride(2)%8==0 && out.stride(1)%8==0 &&
      reinterpret_cast<uintptr_t>(out.data_ptr())%16==0,"output [1,Hq,T,128] aligned");
  c10::cuda::CUDAGuard guard(q8.device());
  const auto stream=at::cuda::getCurrentCUDAStream();
  const int32_t* bp=bands.has_value()?bands->data_ptr<int32_t>():nullptr;
  const int32_t* gp=groups.has_value()?groups->data_ptr<int32_t>():nullptr;
  const int32_t* ep=entries.has_value()?entries->data_ptr<int32_t>():nullptr;
#define LAUNCH(TYPE) unified_kernel<TYPE><<<order.numel(),NT,SharedBytes,stream>>>(order.data_ptr<int32_t>(), \
    bp,gp,ep,optional_ptr(v16p),optional_ptr(v8p),optional_ptr(v4p), \
    vstat.data_ptr<float>(),q8.data_ptr<int8_t>(),k8p.data_ptr<uint32_t>(),qa8.data_ptr<float>(),ka8.data_ptr<float>(), \
    q4.has_value()?reinterpret_cast<const uint8_t*>(q4->data_ptr()):nullptr,optional_ptr(k4p), \
    q4.has_value()?qa4->data_ptr<float>():nullptr,q4.has_value()?ka4->data_ptr<float>():nullptr, \
    out.data_ptr<TYPE>(),h,hk,t,out.stride(1),out.stride(2),(float)(scale*1.4426950408889634))
  if(out.scalar_type()==at::kHalf) LAUNCH(c10::Half); else LAUNCH(c10::BFloat16);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

#include "common_host.cuh"
