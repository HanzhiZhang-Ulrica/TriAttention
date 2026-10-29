__global__ void quantize_q_kernel(const __nv_bfloat16* q,int8_t* q8,float* qa,uint8_t* q4,float* qa4,int tokens,
    int64_t qsh,int64_t qst) {
  const int lane=threadIdx.x%32,row=blockIdx.x*(blockDim.x/32)+threadIdx.x/32,head=blockIdx.y;
  if(row>=tokens) return;
  const __nv_bfloat16* src=q+head*qsh+(int64_t)row*qst+lane*4;
  float x[4],m=0.f;
#pragma unroll
  for(int i=0;i<4;++i) x[i]=__bfloat162float(src[i]);
  rotate128(x,lane);
#pragma unroll
  for(int i=0;i<4;++i) m=fmaxf(m,fabsf(x[i]));
#pragma unroll
  for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,o));
  const float inv=m>0.f?127.f/m:0.f;
  uint32_t word=0;
#pragma unroll
  for(int i=0;i<4;++i) word|=(uint32_t)(__float2int_rn(x[i]*inv)&0xff)<<(8*i);
  reinterpret_cast<uint32_t*>(q8+((int64_t)head*tokens+row)*D)[lane]=word;
  if(lane==0) qa[(int64_t)head*tokens+row]=m/127.f;
  if(q4!=nullptr) {
    const float inv4=m>0.f?7.f/m:0.f;
    uint32_t nib=0;
#pragma unroll
    for(int i=0;i<4;++i) nib|=(uint32_t)(__float2int_rn(x[i]*inv4)&0xf)<<(4*i);
    reinterpret_cast<uint16_t*>(q4+((int64_t)head*tokens+row)*(D/2))[lane]=(uint16_t)nib;
    if(lane==0) qa4[(int64_t)head*tokens+row]=m/7.f;
  }
}

constexpr int StatBlocks=64;

__global__ void partial_kernel(const __nv_bfloat16* x,float* partial,int tokens,int64_t sh,int64_t st) {
  const int head=blockIdx.y,c=threadIdx.x;
  const __nv_bfloat16* base=x+(int64_t)head*sh+c;
  float sum=0.f,hi=-FLT_MAX,lo=FLT_MAX;
  for(int t=blockIdx.x;t<tokens;t+=StatBlocks) {
    const float v=__bfloat162float(base[(int64_t)t*st]);
    sum+=v;hi=fmaxf(hi,v);lo=fminf(lo,v);
  }
  float* o=partial+(((int64_t)head*StatBlocks+blockIdx.x)*3)*D+c;
  o[0]=sum;o[D]=hi;o[2*D]=lo;
}

__global__ void finalize_kernel(const float* partial,float* stat,int tokens,bool identity) {
  const int64_t h=blockIdx.x;const int c=threadIdx.x;
  float mean=0.f,scale=1.f;
  if(!identity) {
    const float* in=partial+h*StatBlocks*3*D+c;
    float sum=0.f,hi=-FLT_MAX,lo=FLT_MAX;
    for(int b=0;b<StatBlocks;++b) {sum+=in[b*3*D];hi=fmaxf(hi,in[b*3*D+D]);lo=fminf(lo,in[b*3*D+2*D]);}
    mean=sum/tokens;
    const float range=fmaxf(hi-mean,mean-lo);
    scale=range>0.f?range/127.f:1.f;
  }
  stat[h*3*D+c]=mean;stat[h*3*D+D+c]=scale;stat[h*3*D+2*D+c]=1.f/scale;
}

__global__ void quantize_k_kernel(const __nv_bfloat16* k,const float* kstat,uint32_t* packed,float* ka,
    uint32_t* packed4,float* ka4,int tokens,int64_t ksh,int64_t kst) {
  __shared__ float tile_k[BN][D+1];
  __shared__ float wmax[8];
  const int tile=blockIdx.x,head=blockIdx.y,tiles=(tokens+63)/64,tid=threadIdx.x;
  const float* mean=kstat+(int64_t)head*3*D;
  for(int i=tid;i<BN*D;i+=blockDim.x) {
    const int r=i/D,c=i%D,t=tile*64+r;
    tile_k[r][c]=t<tokens?__bfloat162float(k[(int64_t)head*ksh+(int64_t)t*kst+c])-mean[c]:0.f;
  }
  __syncthreads();
  {
    const int lane=tid%32,warp=tid/32;
    for(int r=warp;r<BN;r+=blockDim.x/32) {
      float x[4];
#pragma unroll
      for(int i=0;i<4;++i) x[i]=tile_k[r][lane*4+i];
      rotate128(x,lane);
#pragma unroll
      for(int i=0;i<4;++i) tile_k[r][lane*4+i]=x[i];
    }
  }
  __syncthreads();

  {
    const int col=tid/64;
    float cm=0.f;
    for(int i=tid%64;i<16*D;i+=64) cm=fmaxf(cm,fabsf(tile_k[col*16+i/D][i%D]));
#pragma unroll
    for(int o=16;o;o>>=1) cm=fmaxf(cm,__shfl_xor_sync(0xffffffffu,cm,o));
    if(tid%32==0) wmax[tid/32]=cm;
  }
  __syncthreads();
  float inv_col[4],inv4_col[4];
#pragma unroll
  for(int c=0;c<4;++c) {
    const float cm=fmaxf(wmax[2*c],wmax[2*c+1]);
    inv_col[c]=cm>0.f?127.f/cm:0.f;
    inv4_col[c]=cm>0.f?7.f/cm:0.f;

    if(tid==c) ka[((int64_t)head*tiles+tile)*4+c]=cm>0.f?cm/127.f:1.f;
    if(packed4!=nullptr && tid==c) ka4[((int64_t)head*tiles+tile)*4+c]=cm>0.f?cm/7.f:1.f;
  }
  uint32_t* target=packed+((int64_t)head*tiles+tile)*KPairWords;
  for(int i=tid;i<KPairWords/2;i+=blockDim.x) {
    const int lane=i%32,pair=(i/32)%2,col=(i/64)%4,chunk=i/256,group=lane/4,sub=lane%4;
    const int r=col*16+group+pair*8,byte=chunk*32+sub*4;
    uint32_t w[2];
#pragma unroll
    for(int h=0;h<2;++h) {
      uint32_t word=0;
#pragma unroll
      for(int b=0;b<4;++b) word|=(uint32_t)(__float2int_rn(tile_k[r][byte+h*16+b]*inv_col[col])&0xff)<<(8*b);
      w[h]=word;
    }
    reinterpret_cast<uint2*>(target)[i]=make_uint2(w[0],w[1]);
  }
  if(packed4==nullptr) return;

  uint32_t* target4=packed4+((int64_t)head*tiles+tile)*K4Words;
  for(int i=tid;i<K4Words/2;i+=blockDim.x) {
    const int lane=i%32,pair=(i/32)%2,col=(i/64)%4,step=i/256,group=lane/4,sub=lane%4;
    const int r=col*16+group+pair*8,f0=step*64+sub*8;
    uint32_t w[2];
#pragma unroll
    for(int h=0;h<2;++h) {
      uint32_t word=0;
#pragma unroll
      for(int e=0;e<8;++e) word|=(uint32_t)(__float2int_rn(tile_k[r][f0+h*32+e]*inv4_col[col])&0xf)<<(4*e);
      w[h]=word;
    }
    reinterpret_cast<uint2*>(target4)[i]=make_uint2(w[0],w[1]);
  }
}
__device__ __forceinline__ float vnorm(const __nv_bfloat16* v,const float* stat,int t,int c,int tokens,int64_t vst) {
  return t<tokens?(__bfloat162float(v[(int64_t)t*vst+c])-stat[c])*stat[2*D+c]:0.f;
}

__global__ void pack_v_kernel(const __nv_bfloat16* v,const float* vstat,uint32_t* v16,uint32_t* v8,uint32_t* v4,
    int tokens,int64_t vsh,int64_t vst) {
  const int tile=blockIdx.x,head=blockIdx.y,tiles=(tokens+63)/64;
  const __nv_bfloat16* base=v+(int64_t)head*vsh;
  const float* stat=vstat+(int64_t)head*3*D;
  const int64_t ht=(int64_t)head*tiles+tile;
  if(v16) for(int i=threadIdx.x;i<V16Words;i+=blockDim.x) {

    const int word=i%4,lane=(i/4)%32,col16=(i/128)%8,key16=i/1024,g=lane/4,s=lane%4;
    const int t=tile*64+key16*16+2*s+(word%2)*8,c=col16*16+g+(word/2)*8;
    v16[ht*V16Words+i]=(uint32_t)__half_as_ushort(__float2half_rn(vnorm(base,stat,t,c,tokens,vst)))|
        ((uint32_t)__half_as_ushort(__float2half_rn(vnorm(base,stat,t+1,c,tokens,vst)))<<16);
  }
  if(v8) for(int i=threadIdx.x;i<V8Words;i+=blockDim.x) {

    const int word=i%4,lane=(i/4)%32,n8=i/128,g=lane/4,s=lane%4,c=n8*8+g,col=(word/2)*2+(word%2);
    uint32_t packed=0;
#pragma unroll
    for(int b=0;b<4;++b) {
      const int q=max(-127,min(127,__float2int_rn(vnorm(base,stat,tile*64+col*16+key_in_col(s,b),c,tokens,vst))));
      packed|=(uint32_t)(q&0xff)<<(8*b);
    }
    v8[ht*V8Words+i]=packed;
  }
  if(v4) for(int i=threadIdx.x;i<V4Words;i+=blockDim.x) {

    const int h=i%2,lane=(i/2)%32,n8=i/64,g=lane/4,s=lane%4,c=n8*8+g;
    uint32_t packed=0;
#pragma unroll
    for(int j=0;j<8;++j) {
      const int col=2*h+j%2;
      const int q=max(-7,min(7,__float2int_rn(vnorm(base,stat,tile*64+col*16+key_in_col(s,j/2),c,tokens,vst)*(7.f/127.f))));
      packed|=(uint32_t)(q&0xf)<<(4*j);
    }
    v4[ht*V4Words+i]=packed;
  }
}

constexpr int MaxGroup=8;
constexpr int MergeThreads=4*NT;
constexpr int DecodeShared=16*QS+16*Q4S+2*KPairWords*4+2*V8Words*4;

template<int pv,int kq>
__device__ __forceinline__ void decode_tile(const uint8_t* qt,const uint32_t* kb,const uint32_t* vb,uint32_t* pbuf,
    float* rowmax,const float (&slope)[2],float& m,float& denom,float (&out)[4][4],float& corr,
    int key_begin,int prompt,int group_heads,unsigned rowmask) {
  const int lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,sub=lane%4;
  int c[8];
#pragma unroll
  for(int j=0;j<8;++j) c[j]=Magic;
  if constexpr(kq==8) {
#pragma unroll
    for(int chunk=0;chunk<4;++chunk) {
      const int byte=chunk*32+sub*4;
      const uint32_t a0=*reinterpret_cast<const uint32_t*>(qt+group*QS+byte);
      const uint32_t a2=*reinterpret_cast<const uint32_t*>(qt+group*QS+byte+16);
      const uint2* pair=reinterpret_cast<const uint2*>(kb)+((chunk*4+warp)*2*32+lane);
      const uint2 b01=pair[0],b23=pair[32];
      imma8(c,a0,0u,a2,0u,b01.x,b01.y);
      imma8(c+4,a0,0u,a2,0u,b23.x,b23.y);
    }
  } else {
#pragma unroll
    for(int step=0;step<2;++step) {
      const int byte=step*32+sub*4;
      const uint32_t a0=*reinterpret_cast<const uint32_t*>(qt+group*Q4S+byte);
      const uint32_t a2=*reinterpret_cast<const uint32_t*>(qt+group*Q4S+byte+16);
      const uint2* pair=reinterpret_cast<const uint2*>(kb)+((step*4+warp)*2*32+lane);
      const uint2 b01=pair[0],b23=pair[32];
      imma4(c,a0,0u,a2,0u,b01.x,b01.y);
      imma4(c+4,a0,0u,a2,0u,b23.x,b23.y);
    }
  }

  const bool row_on=group<group_heads && ((rowmask>>group)&1u);
  float x[4];
#pragma unroll
  for(int i=0;i<4;++i) {
    const int j=i<2?i:i+2;
    const int key=key_begin+warp*16+(j>=4?8:0)+sub*2+(j%2);
    x[i]=(row_on && key<prompt)?__int_as_float(c[j]):-CUDART_INF_F;
  }
  float mx=fmaxf(fmaxf(x[0],x[1]),fmaxf(x[2],x[3]));
  mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,1));
  mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,2));

  if(sub==0) rowmax[warp*8+group]=mx>-CUDART_INF_F?slope[0]*__fsub_rn(mx,12582912.f):-CUDART_INF_F;
  __syncthreads();
  const float tile=fmaxf(fmaxf(rowmax[group],rowmax[8+group]),fmaxf(rowmax[16+group],rowmax[24+group]));
  const float bias=-slope[0]*12582912.f;
  const float n=fmaxf(m,tile),alpha=exp2f(m-n);
  m=n;
  if(alpha!=1.f) {
#pragma unroll
    for(int i=0;i<4;++i) {out[i][0]*=alpha;out[i][1]*=alpha;}
    corr*=alpha;
  }
  constexpr float levels=pv==PvI8?255.f:15.f;

  const bool any=tile>-CUDART_INF_F;
  float base=any?tile:n;
  base+=tile_base_margin(slope[1],base);
  const float bq=any?exp2f(base-n)/levels:0.f,restore=pv==PvI8?bq:bq*(127.f/7.f);
  uint32_t w[4];
#pragma unroll
  for(int i=0;i<4;++i) {

    const float arg=kq==8?fmaf(slope[0],x[i],bias-base):fmaf(slope[0],__fsub_rn(x[i],12582912.f),-base);
    w[i]=__float_as_uint(fmaf(exp2f(arg),levels,pmagic_dither(x[i])));
  }
  const uint32_t word=pack_bytes(w[0],w[1],w[2],w[3]);
  pbuf[warp*32+lane]=word;
  float part=(float)__dp4a(word,0x01010101u,0u)*bq;
  part+=__shfl_xor_sync(0xffffffffu,part,1);
  part+=__shfl_xor_sync(0xffffffffu,part,2);
  denom=fmaf(denom,alpha,part);
  if constexpr(pv==PvI8) corr=fmaf(12582912.f,restore,corr);
  __syncthreads();
  if constexpr(pv==PvI8) {
    const uint32_t a[2][4]={{pbuf[lane],0u,pbuf[32+lane],0u},{pbuf[64+lane],0u,pbuf[96+lane],0u}};
#pragma unroll
    for(int i=0;i<4;++i) {
      int d[4];
      imma_u8s8(d,a,*reinterpret_cast<const uint4*>(vb+((warp*4+i)*32+lane)*4));
      out[i][0]=fmaf(__int_as_float(d[0]),restore,out[i][0]);
      out[i][1]=fmaf(__int_as_float(d[1]),restore,out[i][1]);
    }
  } else {
    const uint32_t a[4]={pack_nibbles(pbuf[lane],pbuf[32+lane]),0u,pack_nibbles(pbuf[64+lane],pbuf[96+lane]),0u};
#pragma unroll
    for(int i=0;i<4;++i) {
      int d[4];
      imma_u4s4(d,a,*reinterpret_cast<const uint2*>(vb+((warp*4+i)*32+lane)*2));
      out[i][0]=fmaf(__fsub_rn(__int_as_float(d[0]),12582912.f),restore,out[i][0]);
      out[i][1]=fmaf(__fsub_rn(__int_as_float(d[1]),12582912.f),restore,out[i][1]);
    }
  }
}

__global__ __launch_bounds__(NT) void decode_kernel(const __nv_bfloat16* q,const uint32_t* k8c,const uint32_t* k4c,
    const float* ka,const float* ka4,const int32_t* entry,const int32_t* kinfo,const int32_t* count,
    const int32_t* fmt,const int32_t* slot,const uint32_t* v8,const uint32_t* v4,
    float* pm,float* pl,float* po,int heads_q,int heads_kv,int prompt,int per,float sm_log2,int64_t qsh,int maxn) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* qt=smem;
  uint8_t* qt4=smem+16*QS;
  uint32_t* kbuf=reinterpret_cast<uint32_t*>(smem+16*QS+16*Q4S);
  uint32_t* vbuf=kbuf+2*KPairWords;
  __shared__ uint32_t pbuf[4*32*2];
  __shared__ float rowmax[32];
  __shared__ float qscale[16],qscale4[16];
  __shared__ float dsum[4][8];
  const int split=blockIdx.x,kh=blockIdx.y,tid=threadIdx.x&(NT-1),lane=tid%32,warp=tid/32;
  const int g=heads_q/heads_kv,tiles=(prompt+BN-1)/BN,first=split*per,last=min(count[kh],first+per);

  for(int r=warp;r<16;r+=4) {
    uint32_t word=0;
    float mxq=0.f,xv[4]={0.f,0.f,0.f,0.f};
    if(r<g) {
      const __nv_bfloat16* src=q+(kh*g+r)*qsh+lane*4;
#pragma unroll
      for(int i=0;i<4;++i) xv[i]=__bfloat162float(src[i]);
      rotate128(xv,lane);
#pragma unroll
      for(int i=0;i<4;++i) mxq=fmaxf(mxq,fabsf(xv[i]));
#pragma unroll
      for(int o=16;o;o>>=1) mxq=fmaxf(mxq,__shfl_xor_sync(0xffffffffu,mxq,o));
      const float inv=mxq>0.f?127.f/mxq:0.f;
#pragma unroll
      for(int i=0;i<4;++i) word|=(uint32_t)(__float2int_rn(xv[i]*inv)&0xff)<<(8*i);
    }
    reinterpret_cast<uint32_t*>(qt+r*QS)[lane]=word;
    if(lane==0) qscale[r]=r<g?(mxq>0.f?mxq/127.f*sm_log2:FLT_MIN):FLT_MIN;
    uint32_t nib=0;
    const float inv4=mxq>0.f?7.f/mxq:0.f;
#pragma unroll
    for(int i=0;i<4;++i) nib|=(uint32_t)(__float2int_rn(xv[i]*inv4)&0xf)<<(4*i);
    reinterpret_cast<uint16_t*>(qt4+r*Q4S)[lane]=r<g?(uint16_t)nib:(uint16_t)0;
    if(lane==0) qscale4[r]=r<g?(mxq>0.f?mxq/7.f*sm_log2:FLT_MIN):FLT_MIN;
  }
  auto stage=[&](int i,int buf) {
    const int tile=entry[kh*maxn+i],info=kinfo[kh*maxn+i],ks4=(info>>8)&1,kslot=(int)((unsigned)info>>9);
    const uint32_t* ks=ks4?k4c+(int64_t)kslot*K4Words:k8c+(int64_t)kslot*KPairWords;
    const int kwords=ks4?K4Words:KPairWords;
    for(int w=tid;w<kwords/4;w+=NT) cp16(kbuf+buf*KPairWords+w*4,ks+w*4,true);
    const int f=fmt[tile],sl=slot[kh*maxn+i];
    const uint32_t* vs=f==PvI8?v8+(int64_t)sl*V8Words:v4+(int64_t)sl*V4Words;
    const int words=f==PvI8?V8Words:V4Words;
    for(int w=tid;w<words/4;w+=NT) cp16(vbuf+buf*V8Words+w*4,vs+w*4,true);
    commit();
  };
  float m=-FLT_MAX,denom=0.f,out[4][4]={},corr=0.f;
  __syncthreads();
  const int group=lane/4;
  if(first<last) stage(first,0);
  for(int i=first;i<last;++i) {
    const int buf=(i-first)&1;
    if(i+1<last) {stage(i+1,buf^1);wait<1>();} else wait<0>();
    __syncthreads();
    const int tile=entry[kh*maxn+i],info=kinfo[kh*maxn+i];
    const bool k4tile=(info>>8)&1;
    const unsigned rowmask=info&0xff;

    const float* kt=(k4tile?ka4:ka)+((int64_t)kh*tiles+tile)*4;
    const float qs=k4tile?qscale4[group]:qscale[group];
    const float slope[2]={qs*kt[warp],qs*fmaxf(fmaxf(kt[0],kt[1]),fmaxf(kt[2],kt[3]))};
    const uint32_t* kb=kbuf+buf*KPairWords;
    const uint32_t* vb=vbuf+buf*V8Words;
    const int f=fmt[tile];
    if(k4tile) {
      if(f==PvI8) decode_tile<PvI8,4>(qt4,kb,vb,pbuf,rowmax,slope,m,denom,out,corr,tile*BN,prompt,g,rowmask);
      else decode_tile<PvI4,4>(qt4,kb,vb,pbuf,rowmax,slope,m,denom,out,corr,tile*BN,prompt,g,rowmask);
    } else {
      if(f==PvI8) decode_tile<PvI8,8>(qt,kb,vb,pbuf,rowmax,slope,m,denom,out,corr,tile*BN,prompt,g,rowmask);
      else decode_tile<PvI4,8>(qt,kb,vb,pbuf,rowmax,slope,m,denom,out,corr,tile*BN,prompt,g,rowmask);
    }
    __syncthreads();
  }

  if(lane%4==0) dsum[warp][group]=denom;
  __syncthreads();
  const int splits=gridDim.x,sub=lane%4;
  if(group<g) {
    const int64_t idx=(int64_t)(kh*g+group)*splits+split;
#pragma unroll
    for(int i=0;i<4;++i) {
      const int ch=warp*32+i*8+sub*2;
      po[idx*D+ch]=out[i][0]-corr;
      po[idx*D+ch+1]=out[i][1]-corr;
    }
    if(warp==0 && sub==0) {
      pm[idx]=m;
      pl[idx]=dsum[0][group]+dsum[1][group]+dsum[2][group]+dsum[3][group];
    }
  }
}

__global__ void decode_merge_kernel(const float* pm,const float* pl,const float* po,const float* vstat,const float* kstat,
    const __nv_bfloat16* q,__nv_bfloat16* ktail,__nv_bfloat16* vtail,const __nv_bfloat16* kcur,const __nv_bfloat16* vcur,
    const int64_t* step,__nv_bfloat16* out,int heads_q,int heads_kv,int splits,int max_tail,float sm_log2,int64_t qsh,
    int64_t tsh,int64_t tst,int64_t csh,int64_t osh) {

  constexpr int W=NT/32,WS=MergeThreads/32;
  __shared__ float ts[1024];
  __shared__ __align__(16) float ew[1024];
  __shared__ float wmax[WS],wl[W];
  __shared__ __align__(16) float wo[W][D];

  const int prior=min((int)*step,max_tail-1),tail=prior+1;
  const int hq=blockIdx.x,kh=hq/(heads_q/heads_kv),tid=threadIdx.x,lane=tid%32,warp=tid/32;
  const float* mean=vstat+(int64_t)kh*3*D;
  const float* scale=mean+D;
  if(hq%(heads_q/heads_kv)==0 && tid<D) {
    ktail[kh*tsh+prior*tst+tid]=kcur[kh*csh+tid];
    vtail[kh*tsh+prior*tst+tid]=vcur[kh*csh+tid];
  }

  float qv[4],kc[4];
#pragma unroll
  for(int i=0;i<4;++i) {qv[i]=__bfloat162float(q[hq*qsh+lane*4+i]);kc[i]=kstat[(int64_t)kh*3*D+lane*4+i];}

  constexpr int U=8;
  for(int t0=warp*U;t0<tail;t0+=WS*U) {
    float part[U];
#pragma unroll
    for(int u=0;u<U;++u) {
      const int t=min(t0+u,tail-1);
      const __nv_bfloat16* kv=t<prior?ktail+kh*tsh+t*tst:kcur+kh*csh;
      part[u]=0.f;
#pragma unroll
      for(int i=0;i<4;++i) part[u]=fmaf(qv[i],__bfloat162float(kv[lane*4+i])-kc[i],part[u]);
    }
#pragma unroll
    for(int o=16;o;o>>=1)
#pragma unroll
      for(int u=0;u<U;++u) part[u]+=__shfl_xor_sync(0xffffffffu,part[u],o);
    if(lane==0)
#pragma unroll
      for(int u=0;u<U;++u) if(t0+u<tail) ts[t0+u]=part[u]*sm_log2;
  }
  __syncthreads();

  float m=-FLT_MAX;
  for(int s=tid;s<splits;s+=MergeThreads) m=fmaxf(m,pm[(int64_t)hq*splits+s]);
  for(int t=tid;t<tail;t+=MergeThreads) m=fmaxf(m,ts[t]);
#pragma unroll
  for(int o=16;o;o>>=1) m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,o));
  if(lane==0) wmax[warp]=m;
  __syncthreads();
  m=wmax[0];
#pragma unroll
  for(int w=1;w<WS;++w) m=fmaxf(m,wmax[w]);
  for(int t=tid;t<tail;t+=MergeThreads) ew[t]=exp2f(ts[t]-m);

  float l=0.f,o[4]={0.f,0.f,0.f,0.f};
#pragma unroll 4
  for(int s=warp;s<splits && warp<W;s+=W) {
    const int64_t idx=(int64_t)hq*splits+s;
    const float w=exp2f(pm[idx]-m);
    l=fmaf(pl[idx],w,l);
    const float4 p=*reinterpret_cast<const float4*>(po+idx*D+lane*4);
    o[0]=fmaf(p.x,w,o[0]);o[1]=fmaf(p.y,w,o[1]);o[2]=fmaf(p.z,w,o[2]);o[3]=fmaf(p.w,w,o[3]);
  }
  if(warp<W) {
    if(lane==0) wl[warp]=l;
    *reinterpret_cast<float4*>(&wo[warp][lane*4])=make_float4(o[0],o[1],o[2],o[3]);
  }
  __syncthreads();
  if(tid>=D) return;
  float total=0.f,acc=0.f;
#pragma unroll
  for(int w=0;w<W;++w) {total+=wl[w];acc+=wo[w][tid];}
  acc=acc*scale[tid]+total*mean[tid];

  const __nv_bfloat16* vt=vtail+kh*tsh+tid;
  int t=0;
  for(;t+32<=prior;t+=32) {
    float vv[32],e[32];
#pragma unroll
    for(int u=0;u<32;++u) vv[u]=__bfloat162float(vt[(int64_t)(t+u)*tst]);
#pragma unroll
    for(int u=0;u<32;u+=4) {const float4 x=*reinterpret_cast<const float4*>(&ew[t+u]);e[u]=x.x;e[u+1]=x.y;e[u+2]=x.z;e[u+3]=x.w;}
#pragma unroll
    for(int u=0;u<32;++u) {total+=e[u];acc=fmaf(e[u],vv[u],acc);}
  }
  for(;t<prior;++t) {
    const float e=ew[t];
    total+=e;
    acc=fmaf(e,__bfloat162float(vt[(int64_t)t*tst]),acc);
  }
  total+=ew[prior];
  acc=fmaf(ew[prior],__bfloat162float(vcur[kh*csh+tid]),acc);
  out[hq*osh+tid]=__float2bfloat16(acc/total);
}

uint32_t* optional_ptr(const c10::optional<torch::Tensor>& x) {
  return x.has_value()?x->data_ptr<uint32_t>():nullptr;
}
