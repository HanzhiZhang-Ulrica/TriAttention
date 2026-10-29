#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cfloat>
#include <cstdint>
#include <cstring>
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

constexpr int KStages=2, KSlot=KPairWords*4, VRingBytes=32768;
constexpr int OffKU=KStages*KSlot, OffV=OffKU+KPairWords*4, OffQ=OffV+VRingBytes, OffCols=OffQ+QBytes;
constexpr int OffBar=OffCols+MaxTasks*2, NBars=KStages+8;
constexpr int SharedBytes=OffBar+NBars*8+1024;
static_assert(OffV%1024==0 && VRingBytes/(V4Words*4)<=8 && BM*VS*2<=OffQ && BM*Q4S<=QBytes && OffBar%8==0);
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

constexpr int KLbo=128, KSbo=1024;
__host__ __device__ __forceinline__ int kcore(int key,int kb) { return (key>>3)*KSbo+kb*KLbo+(key&7)*16; }

__device__ __forceinline__ uint32_t s4x4_to_s8(uint32_t x) {
  const uint32_t w=(x&0xfu)|((x&0xf0u)<<4)|((x&0xf00u)<<8)|((x&0xf000u)<<12);
  return __vsub4(w^0x08080808u,0x08080808u);
}

__host__ __device__ __forceinline__ int perm_row(int k) { return 16*((k>>1)&3)+8*((k>>3)&1)+2*(k>>4)+(k&1); }
template<bool Perm=false>
__device__ __forceinline__ void unpackk4(uint8_t* dst,const uint8_t* raw) {
  const int tid=threadIdx.x&(NT-1);
#pragma unroll
  for(int i=tid;i<K4Words/2;i+=NT) {
    const uint2 w=reinterpret_cast<const uint2*>(raw)[i];
    const int lane=i&31,pair=(i>>5)&1,col=(i>>6)&3,step=i>>8;
    const int key=col*16+pair*8+lane/4;
    *reinterpret_cast<uint4*>(dst+kcore(Perm?perm_row(key):key,step*4+lane%4))=
        make_uint4(s4x4_to_s8(w.x),s4x4_to_s8(w.x>>16),s4x4_to_s8(w.y),s4x4_to_s8(w.y>>16));
  }
}

__device__ __forceinline__ void stageq4_desc(uint8_t* dst,const uint8_t* q4,int64_t head,int tokens,int qb,bool full) {
  const int tid=threadIdx.x&(NT-1);
  for(int i=tid;i<BM*8;i+=NT) {
    const int row=i>>3,kb=i&7,b2=(kb>>2)*32+(kb&3)*4;
    const bool valid=full || qb+row<tokens;
    uint32_t x=0,y=0;
    if(valid) {
      const uint8_t* src=q4+(head*tokens+qb+row)*(D/2);
      x=*reinterpret_cast<const uint32_t*>(src+b2);
      y=*reinterpret_cast<const uint32_t*>(src+b2+16);
    }
    *reinterpret_cast<uint4*>(dst+kcore(row,kb))=make_uint4(s4x4_to_s8(x),s4x4_to_s8(x>>16),s4x4_to_s8(y),s4x4_to_s8(y>>16));
  }
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* bar,int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)),"r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint64_t* bar,uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)),"r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar,uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nWAIT:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT;\n}"
      :: "r"(smem_u32(bar)),"r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_load(void* dst,const void* src,uint32_t bytes,uint64_t* bar) {
  mbar_expect(bar,bytes);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
      :: "r"(smem_u32(dst)),"l"(src),"r"(bytes),"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_k8(void* dst,const CUtensorMap* map,int tile,uint64_t* bar) {
  mbar_expect(bar,KPairWords*4);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
      :: "r"(smem_u32(dst)),"l"(reinterpret_cast<uint64_t>(map)),"r"(0),"r"(0),"r"(4*tile),"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void tma_load_v8(void* dst,const CUtensorMap* map,int row,uint64_t* bar) {
  mbar_expect(bar,V8Words*4);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
      :: "r"(smem_u32(dst)),"l"(reinterpret_cast<uint64_t>(map)),"r"(0),"r"(row),"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void tma_k8p_block(void* dst,const CUtensorMap* map,int tile,int b,uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
      :: "r"(smem_u32(dst)+2048*b),"l"(reinterpret_cast<uint64_t>(map)),"r"(0),"r"(0),"r"(0),"r"(0),"r"(32*(4*tile+b)),
         "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tma_load_k8p(void* dst,const CUtensorMap* map,int tile,uint64_t* bar) {
  mbar_expect(bar,KPairWords*4);
#pragma unroll
  for(int b=0;b<4;++b)
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(smem_u32(dst)+2048*b),"l"(reinterpret_cast<uint64_t>(map)),"r"(0),"r"(0),"r"(0),"r"(0),"r"(32*(4*tile+b)),
           "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void publish() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
}
template<int Words> __device__ __forceinline__ void stagev(uint32_t* dst,const uint32_t* source) {
  const int tid=threadIdx.x&(NT-1);
#pragma unroll
  for(int i=tid;i<Words/4;i+=NT) cp16(dst+i*4,source+i*4,true);
  commit();
}

__device__ __forceinline__ uint64_t kdesc(const uint8_t* tile) {
  const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(tile));
  return static_cast<uint64_t>((a&0x3FFFF)>>4)|(static_cast<uint64_t>(KLbo>>4)<<16)|(static_cast<uint64_t>(KSbo>>4)<<32);
}

__device__ __forceinline__ uint64_t kdesc_sw32(const uint8_t* tile) {
  const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(tile));
  return static_cast<uint64_t>((a&0x3FFFF)>>4)|(1ull<<16)|(static_cast<uint64_t>(256>>4)<<32)|(3ull<<62);
}

__device__ __forceinline__ uint64_t desc_add(uint64_t d,uint32_t off16) {
  uint32_t lo,hi;
  asm("mov.b64 {%0,%1}, %2;" : "=r"(lo),"=r"(hi) : "l"(d));
  uint64_t r;
  asm("mov.b64 %0, {%1,%2};" : "=l"(r) : "r"(lo+off16),"r"(hi));
  return r;
}

__shared__ __align__(16) uint32_t magic4[4];
__device__ __forceinline__ void init_magic(int (&c)[32]) {
  const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(magic4));
#pragma unroll
  for(int k=0;k<8;++k)
    asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(c[4*k]),"=r"(c[4*k+1]),"=r"(c[4*k+2]),"=r"(c[4*k+3]) : "r"(a));
}
__device__ __forceinline__ void fence_acc(int (&c)[32]) {
#pragma unroll
  for(int i=0;i<32;++i) asm volatile("" : "+r"(c[i]) :: "memory");
}

#define TRIATTN_WGMMA_M64N64K32(NAME,TYPES) \
__device__ __forceinline__ void NAME(int (&c)[32],const uint32_t (&a)[4],uint64_t desc) { \
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n" \
      "wgmma.mma_async.sync.aligned.m64n64k32.s32." TYPES " " \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p;\n}" \
      : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3]),"+r"(c[4]),"+r"(c[5]),"+r"(c[6]),"+r"(c[7]), \
        "+r"(c[8]),"+r"(c[9]),"+r"(c[10]),"+r"(c[11]),"+r"(c[12]),"+r"(c[13]),"+r"(c[14]),"+r"(c[15]), \
        "+r"(c[16]),"+r"(c[17]),"+r"(c[18]),"+r"(c[19]),"+r"(c[20]),"+r"(c[21]),"+r"(c[22]),"+r"(c[23]), \
        "+r"(c[24]),"+r"(c[25]),"+r"(c[26]),"+r"(c[27]),"+r"(c[28]),"+r"(c[29]),"+r"(c[30]),"+r"(c[31]) \
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(desc),"r"(1)); \
}
TRIATTN_WGMMA_M64N64K32(wgmma_s8,"s8.s8")
TRIATTN_WGMMA_M64N64K32(wgmma_u8s8,"u8.s8")
#undef TRIATTN_WGMMA_M64N64K32

__device__ __forceinline__ void stageq_desc(uint8_t* dst,const int8_t* q8,int64_t head,int tokens,int qb,bool full) {
  const int tid=threadIdx.x&(NT-1);
  for(int i=tid;i<BM*D/16;i+=NT) {
    const int row=i/(D/16),f0=(i%(D/16))*16;
    const bool valid=full || qb+row<tokens;
    uint4 v=make_uint4(0,0,0,0);
    if(valid) v=*reinterpret_cast<const uint4*>(q8+(head*tokens+qb+row)*D+f0);
    const uint32_t w[4]={v.x,v.y,v.z,v.w};
#pragma unroll
    for(int j=0;j<4;++j) {
      const int f=f0+4*j,c=f>>5,fl=f&31;
      const int p=16*((fl>>3)&1)+4*(2*((fl>>2)&1)+((fl>>4)&1));
      int off=c*2048+row*32+p;
      off^=((off>>7)&1)<<4;
      *reinterpret_cast<uint32_t*>(dst+off)=w[j];
    }
  }
}

__device__ __forceinline__ uint64_t vdesc_sw64(const uint8_t* tile) {
  const uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(tile));
  return static_cast<uint64_t>((a&0x3FFFF)>>4)|(1ull<<16)|(static_cast<uint64_t>(512>>4)<<32)|(2ull<<62);
}

__device__ __forceinline__ void pv8_wg(const uint32_t (&w0)[4],const uint32_t (&w1)[4],const uint8_t* vt,
    const float (&bq_num)[2],float (&out)[16][4]) {

  const uint32_t a[2][4]={{w0[0],w1[0],w0[1],w1[1]},{w0[2],w1[2],w0[3],w1[3]}};
  const uint64_t desc=vdesc_sw64(vt);
#pragma unroll
  for(int h=0;h<2;++h) {
    int c[32];
    init_magic(c);
    fence_acc(c);
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for(int k0=0;k0<2;++k0) wgmma_u8s8(c,a[k0],desc_add(desc,(4096*h+32*k0)>>4));
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    fence_acc(c);
#pragma unroll
    for(int i=0;i<8;++i)
#pragma unroll
      for(int j=0;j<4;++j) out[h*8+i][j]=fmaf(__int_as_float(c[i*4+j]),bq_num[j/2],out[h*8+i][j]);
  }
}

template<int Bits>
__device__ __forceinline__ void q_frags(const uint8_t* q,uint32_t (&a)[4][4]) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,sub=lane%4;
  const int row0=warp*16+lane/4,row1=row0+8;
#pragma unroll
  for(int s=0;s<4;++s)
#pragma unroll
    for(int j=0;j<4;++j) {
      const int r=(j&1)?row1:row0,h=j>>1;
      int off;
      if constexpr(Bits==8) {off=s*2048+r*32+16*h+4*sub;off^=((off>>7)&1)<<4;}
      else off=kcore(r,2*s+h)+4*sub;
      a[s][j]=*reinterpret_cast<const uint32_t*>(q+off);
    }
}
template<int Bits>
__device__ __forceinline__ void qk_issue_rs(const uint32_t (&a)[4][4],const uint8_t* k,int (&c)[32]) {
  init_magic(c);
  const uint64_t bdesc=Bits==8?kdesc_sw32(k):kdesc(k);
  constexpr int kstep=Bits==8?2048:2*KLbo;
  fence_acc(c);
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
  for(int s=0;s<4;++s) wgmma_s8(c,a[s],desc_add(bdesc,(kstep*s)>>4));
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template<bool Masked,bool Perm=false>
__device__ __forceinline__ void qk_finish(int (&c)[32],int tokens,int query_begin,int key_begin,float (&t)[4][8]) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,sub=lane%4;
  const int row0=warp*16+group,row1=row0+8;
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  fence_acc(c);
#pragma unroll
  for(int col=0;col<4;++col)
#pragma unroll
    for(int j=0;j<8;++j) {
      float value=__int_as_float(c[col*8+j]);
      if constexpr(Masked) {
        const int row=(j%4<2)?row0:row1;
        const int key=key_begin+(Perm?16*sub+2*col:col*16+sub*2)+(j>=4?8:0)+(j%2);
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

__device__ __forceinline__ float scaled_row_max1(const float (&t)[4][8],int r,float qs,float ks) {
  float x=-CUDART_INF_F;
#pragma unroll
  for(int col=0;col<4;++col) x=fmaxf(x,fmaxf(fmaxf(t[col][2*r],t[col][2*r+1]),fmaxf(t[col][2*r+4],t[col][2*r+5])));
  float m=x>-CUDART_INF_F?qs*ks*__fsub_rn(x,12582912.f):-CUDART_INF_F;
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
    int heads_q,int heads_kv,int tokens,int qb,int qhead,int64_t osh,int64_t ost,float sm_log2,const CUtensorMap* kmap,
    const CUtensorMap* vmap,const CUtensorMap* kmapp) {
  const int query_begin=qb*BM,kvhead=qhead/(heads_q/heads_kv),tiles=(tokens+63)/64;
  const int64_t qh=qhead,kh=kvhead;
  extern __shared__ __align__(16) uint8_t storage_raw[];
  uint8_t* storage=storage_raw+((1024-(smem_u32(storage_raw)&1023))&1023);
  uint8_t* kring=storage;
  uint8_t* kunp=storage+OffKU;
  uint8_t* vring=storage+OffV;
  uint8_t* qt=storage+OffQ;
  uint8_t* q4t=qt;
  uint16_t* cols=reinterpret_cast<uint16_t*>(storage+OffCols);
  uint64_t* kbar=reinterpret_cast<uint64_t*>(storage+OffBar);
  uint64_t* vbar=kbar+KStages;
  if(threadIdx.x<4) magic4[threadIdx.x]=static_cast<uint32_t>(Magic);
  if(threadIdx.x==0) {
    for(int i=0;i<NBars;++i) mbar_init(kbar+i,1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  uint32_t kphase=0,vphase=0;
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
  stageq_desc(qt,q8,qh,tokens,query_begin,Full);publish();
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
    stageq4_desc(q4t,q4,qh,tokens,query_begin,Full);publish();
    qsp[0]=(Full || query_begin+row0<tokens)?qa4[qh*tokens+query_begin+row0]*sm_log2:FLT_MIN;
    qsp[1]=(Full || query_begin+row1<tokens)?qa4[qh*tokens+query_begin+row1]*sm_log2:FLT_MIN;
  }
  const float* kscale=qkbits==8?kscale8:ka4+kh*tiles*4;
  uint32_t qa[4][4];
  q_frags<qkbits>(qkbits==8?qt:q4t,qa);

  auto exparg=[&](float slope,float x,float b) {
    if constexpr(qkbits==4) return fmaf(slope,__fsub_rn(x,12582912.f),-b);
    else return fmaf(slope,x,fmaf(-slope,12582912.f,-b));
  };
  constexpr int words=pv==PvF16?V16Words:pv==PvI8?V8Words:V4Words;
  const uint32_t* vh=(pv==PvF16?v16:pv==PvI8?v8:v4)+kh*tiles*words;
  constexpr int vbytes=words*4,vstages=VRingBytes/vbytes;
  const int n=end-begin;

  constexpr bool perm=pv==PvI8;
  auto issue_k=[&](int i) {
    const int c=cols[begin+i];
    if constexpr(qkbits==8 && perm) tma_load_k8p(kring+(i%KStages)*KSlot,kmapp,kh*tiles+c,kbar+i%KStages);
    else if constexpr(qkbits==8) tma_load_k8(kring+(i%KStages)*KSlot,kmap,kh*tiles+c,kbar+i%KStages);
    else bulk_load(kring+(i%KStages)*KSlot,k4+(kh*tiles+c)*K4Words,K4Words*4,kbar+i%KStages);
  };
  auto issue_k_part=[&](int i,int b) {
    uint64_t* bar=kbar+i%KStages;
    if(b==0) mbar_expect(bar,KPairWords*4);
    tma_k8p_block(kring+(i%KStages)*KSlot,kmapp,kh*tiles+cols[begin+i],b,bar);
  };
  auto issue_v=[&](int i) {
    if constexpr(pv==PvI8) tma_load_v8(vring+(i%vstages)*vbytes,vmap,(kh*tiles+cols[begin+i])*128,vbar+i%vstages);
    else bulk_load(vring+(i%vstages)*vbytes,vh+cols[begin+i]*words,vbytes,vbar+i%vstages);
  };
  if(threadIdx.x==0) {
    for(int i=0;i<min(KStages,n);++i) issue_k(i);
    for(int i=0;i<min(vstages,n);++i) issue_v(i);
  }
  auto iteration=[&](auto masked_tag,int i,bool flush) {
    constexpr bool last=decltype(masked_tag)::value;
    const int col=cols[begin+i];
    const int key=col*BN;
    const int kslot=i%KStages,vslot=i%vstages;
    mbar_wait(kbar+kslot,(kphase>>kslot)&1u);
    kphase^=1u<<kslot;
    const uint8_t* kt=kring+kslot*KSlot;

    float ks[4];
#pragma unroll
    for(int c=0;c<4;++c) ks[c]=__ldg(kscale+col*4+c);
    const float ksown=__ldg(kscale+col*4+(threadIdx.x&3));
    float t[4][8];
    int cq[32];
    if constexpr(qkbits==8) qk_issue_rs<8>(qa,kt,cq);
    else {
      unpackk4<perm>(kunp,kt);publish();
      if(threadIdx.x==0 && i+KStages<n) issue_k(i+KStages);
      qk_issue_rs<4>(qa,kunp,cq);
    }
    qk_finish<last,perm>(cq,tokens,query_begin,key,t);
    if constexpr(qkbits==8 && perm) {
      if(lane==0 && i+KStages<n) issue_k_part(i+KStages,warp);
    } else if constexpr(qkbits==8) if(threadIdx.x==0 && i+KStages<n) issue_k(i+KStages);

    float tile[2],alpha[2];
#pragma unroll
    for(int r=0;r<2;++r) {
      const float mx=perm?scaled_row_max1(t,r,qsp[r],ksown):scaled_row_max(t,r,qsp[r],ks);
      tile[r]=mx;
      const float n=fmaxf(m[r],tile[r]);
      alpha[r]=exp2f(m[r]-n);
      m[r]=n;
    }

    if(flush || !__all_sync(0xffffffffu,alpha[0]==1.f && alpha[1]==1.f)) {
      const float shift[2]={-corr[0]*alpha[0],-corr[1]*alpha[1]};
#pragma unroll
      for(int c=0;c<16;++c)
#pragma unroll
        for(int j=0;j<4;++j) out[c][j]=fmaf(out[c][j],alpha[j/2],shift[j/2]);
      corr[0]=corr[1]=0.f;
    }
    mbar_wait(vbar+vslot,(vphase>>vslot)&1u);
    vphase^=1u<<vslot;
    const uint32_t* vt=reinterpret_cast<const uint32_t*>(vring+vslot*vbytes);
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
          w[j]=__float_as_uint(fmaf(exp2f(exparg(qsp[(j%4<2)?0:1]*(perm?ksown:ks[col]),t[col][j],base[(j%4<2)?0:1])),levels,
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
        uint32_t w0[4]={a[0][0],a[0][2],a[1][0],a[1][2]},w1[4]={a[0][1],a[0][3],a[1][1],a[1][3]};
        pv8_wg(w0,w1,reinterpret_cast<const uint8_t*>(vt),bq_num,out);
        if(threadIdx.x==96 && i+vstages<n) issue_v(i+vstages);
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
    if constexpr(pv!=PvI8) {
      __syncthreads();
      if(threadIdx.x==0 && i+vstages<n) issue_v(i+vstages);
    }
  };
  for(int i=0;i<n-1;++i) iteration(std::false_type{},i,pv==PvI8 && (pv8_done+i)%16==15);
  if(cols[end-1]==qb) iteration(std::true_type{},n-1,false);
  else iteration(std::false_type{},n-1,false);
  if(pv==PvI8) pv8_done+=end-begin;
  __syncthreads();
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
__global__ __maxnreg__(168) void unified_kernel(const int32_t* order,const int32_t* bands,const int32_t* groups,
    const int32_t* entries,
    const uint32_t* v16,const uint32_t* v8,const uint32_t* v4,const float* vstat,
    const int8_t* q8,const uint32_t* k8,const float* qa8,const float* ka8,
    const uint8_t* q4,const uint32_t* k4,const float* qa4,const float* ka4,Out* output,
    int heads_q,int heads_kv,int tokens,int64_t osh,int64_t ost,float sm_log2,const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap,const __grid_constant__ CUtensorMap kmapp) {
  const int task=order[blockIdx.x];
  const int rows=(tokens+63)/64,qhead=task/rows,qb=task%rows;
  if((qb+1)*BM<=tokens)
    run_task<Out,true>(bands,entries,groups,v16,v8,v4,vstat,q8,k8,qa8,ka8,q4,k4,qa4,ka4,output,heads_q,heads_kv,tokens,qb,qhead,
        osh,ost,sm_log2,&kmap,&vmap,&kmapp);
  else
    run_task<Out,false>(bands,entries,groups,v16,v8,v4,vstat,q8,k8,qa8,ka8,q4,k4,qa4,ka4,output,heads_q,heads_kv,tokens,qb,qhead,
        osh,ost,sm_log2,&kmap,&vmap,&kmapp);
}

#include "common_device.cuh"
}

static PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder() {
  static PFN_cuTensorMapEncodeTiled_v12000 encode=nullptr;
  if(encode==nullptr) {
    void* fn=nullptr;
    cudaDriverEntryPointQueryResult query;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",&fn,cudaEnableDefault,&query));
    TORCH_CHECK(fn!=nullptr && query==cudaDriverEntryPointSuccess,"cuTensorMapEncodeTiled unavailable");
    encode=reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }
  return encode;
}

#define TRIATTN_L2PROMO (128==256?CU_TENSOR_MAP_L2_PROMOTION_L2_256B:128==64?CU_TENSOR_MAP_L2_PROMOTION_L2_64B:CU_TENSOR_MAP_L2_PROMOTION_L2_128B)
static CUtensorMap k8_tensor_map(void* base,int64_t tiles) {
  const auto encode=tensor_map_encoder();
  CUtensorMap map;
  const cuuint64_t dims[3]={32,64,static_cast<cuuint64_t>(4*tiles)};
  const cuuint64_t strides[2]={32,2048};
  const cuuint32_t box[3]={32,64,4},estride[3]={1,1,1};
  const CUresult r=encode(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,base,dims,strides,box,estride,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_32B,TRIATTN_L2PROMO,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r==CUDA_SUCCESS,"cuTensorMapEncodeTiled failed: ",static_cast<int>(r));
  return map;
}

static CUtensorMap k8p_tensor_map(void* base,int64_t tiles) {
  CUtensorMap map;
  const cuuint64_t dims[5]={32,2,4,2,static_cast<cuuint64_t>(32*4*tiles)};
  const cuuint64_t strides[4]={32,512,256,64};
  const cuuint32_t box[5]={32,2,4,2,4},estride[5]={1,1,1,1,1};
  const CUresult r=tensor_map_encoder()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,5,base,dims,strides,box,estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_32B,TRIATTN_L2PROMO,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r==CUDA_SUCCESS,"cuTensorMapEncodeTiled (K8 permuted) failed: ",static_cast<int>(r));
  return map;
}

static CUtensorMap v8_tensor_map(const c10::optional<torch::Tensor>& v8p) {
  CUtensorMap map;
  memset(&map,0,sizeof(map));
  if(!v8p.has_value() || v8p->numel()==0) return map;
  TORCH_CHECK(v8p->is_contiguous() && v8p->numel()%V8Words==0 && reinterpret_cast<uintptr_t>(v8p->data_ptr())%16==0,
      "V8 operands [*, V8Words]");
  const cuuint64_t dims[2]={64,static_cast<cuuint64_t>(v8p->numel()/V8Words*128)};
  const cuuint64_t strides[1]={64};
  const cuuint32_t box[2]={64,128},estride[2]={1,1};
  const CUresult r=tensor_map_encoder()(&map,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,v8p->data_ptr(),dims,strides,box,estride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_64B,TRIATTN_L2PROMO,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r==CUDA_SUCCESS,"cuTensorMapEncodeTiled (V8) failed: ",static_cast<int>(r));
  return map;
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
  const int tiles=(t+63)/64;
  TORCH_CHECK(k8p.is_contiguous() && k8p.numel()>=(int64_t)hk*tiles*KPairWords &&
      reinterpret_cast<uintptr_t>(k8p.data_ptr())%16==0,"K8 operands [Hkv, tiles, KPairWords]");
  const CUtensorMap kmap=k8_tensor_map(k8p.data_ptr(),(int64_t)hk*tiles);
  const CUtensorMap kmapp=k8p_tensor_map(k8p.data_ptr(),(int64_t)hk*tiles);
  const CUtensorMap vmap=v8_tensor_map(v8p);
  static bool configured=false;
  if(!configured) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(unified_kernel<c10::Half>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
    C10_CUDA_CHECK(cudaFuncSetAttribute(unified_kernel<c10::BFloat16>,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
    configured=true;
  }
#define LAUNCH(TYPE) unified_kernel<TYPE><<<order.numel(),NT,SharedBytes,stream>>>(order.data_ptr<int32_t>(), \
    bp,gp,ep,optional_ptr(v16p),optional_ptr(v8p),optional_ptr(v4p), \
    vstat.data_ptr<float>(),q8.data_ptr<int8_t>(),k8p.data_ptr<uint32_t>(),qa8.data_ptr<float>(),ka8.data_ptr<float>(), \
    q4.has_value()?reinterpret_cast<const uint8_t*>(q4->data_ptr()):nullptr,optional_ptr(k4p), \
    q4.has_value()?qa4->data_ptr<float>():nullptr,q4.has_value()?ka4->data_ptr<float>():nullptr, \
    out.data_ptr<TYPE>(),h,hk,t,out.stride(1),out.stride(2),(float)(scale*1.4426950408889634),kmap,vmap,kmapp)
  if(out.scalar_type()==at::kHalf) LAUNCH(c10::Half); else LAUNCH(c10::BFloat16);
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

#include "common_host.cuh"
