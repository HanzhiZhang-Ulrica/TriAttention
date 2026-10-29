void prepare_qk(torch::Tensor q,torch::Tensor k,torch::Tensor q8,torch::Tensor qa,torch::Tensor kp,torch::Tensor ka,
    torch::Tensor partial,torch::Tensor kstat,c10::optional<torch::Tensor> q4,c10::optional<torch::Tensor> qa4,
    c10::optional<torch::Tensor> k4p,c10::optional<torch::Tensor> ka4) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4 && q.size(0)==1 && q.size(3)==D && q.stride(3)==1 &&
      k.scalar_type()==at::kBFloat16 && k.dim()==4 && k.size(0)==1 && k.size(2)==q.size(2) && k.size(3)==D && k.stride(3)==1,
      "BF16 Q/K [1,H,T,128] with contiguous features");
  const int hq=q.size(1),hk=k.size(1),t=q.size(2),tiles=(t+63)/64;
  TORCH_CHECK(q8.numel()==(int64_t)hq*t*D && qa.numel()==(int64_t)hq*t && kp.numel()==(int64_t)hk*tiles*KPairWords &&
      ka.numel()==(int64_t)hk*tiles*4 && partial.numel()==(int64_t)hk*StatBlocks*3*D && kstat.numel()==(int64_t)hk*3*D,"Q/K buffers");
  const bool int4=q4.has_value();
  TORCH_CHECK(int4==qa4.has_value() && int4==k4p.has_value() && int4==ka4.has_value() &&
      (!int4 || (q4->numel()==(int64_t)hq*t*D/2 && qa4->numel()==(int64_t)hq*t && k4p->numel()==(int64_t)hk*tiles*K4Words &&
                 ka4->numel()==(int64_t)hk*tiles*4)),"INT4 Q/K buffers (all or none)");
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream=at::cuda::getCurrentCUDAStream();
  const auto* qp=reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  const auto* kp_in=reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  quantize_q_kernel<<<dim3((t+3)/4,hq),128,0,stream>>>(qp,q8.data_ptr<int8_t>(),qa.data_ptr<float>(),
      int4?reinterpret_cast<uint8_t*>(q4->data_ptr()):nullptr,int4?qa4->data_ptr<float>():nullptr,t,q.stride(1),q.stride(2));
  partial_kernel<<<dim3(StatBlocks,hk),D,0,stream>>>(kp_in,partial.data_ptr<float>(),t,k.stride(1),k.stride(2));
  finalize_kernel<<<hk,D,0,stream>>>(partial.data_ptr<float>(),kstat.data_ptr<float>(),t,false);
  quantize_k_kernel<<<dim3(tiles,hk),256,0,stream>>>(kp_in,kstat.data_ptr<float>(),kp.data_ptr<uint32_t>(),ka.data_ptr<float>(),
      int4?k4p->data_ptr<uint32_t>():nullptr,int4?ka4->data_ptr<float>():nullptr,t,k.stride(1),k.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void pack_v(torch::Tensor v,torch::Tensor partial,torch::Tensor vstat,c10::optional<torch::Tensor> v16p,
    c10::optional<torch::Tensor> v8p,c10::optional<torch::Tensor> v4p) {
  TORCH_CHECK(v.is_cuda() && v.scalar_type()==at::kBFloat16 && v.dim()==4 && v.size(0)==1 && v.size(3)==D && v.stride(3)==1,
      "BF16 V [1,Hkv,T,128]");
  const int hk=v.size(1),t=v.size(2),tiles=(t+63)/64;
  TORCH_CHECK(partial.numel()==(int64_t)hk*StatBlocks*3*D && vstat.numel()==(int64_t)hk*3*D,"V statistics buffers");
  auto check=[&](const c10::optional<torch::Tensor>& x,int64_t words){
    TORCH_CHECK(!x.has_value() || (x->scalar_type()==at::kUInt32 && x->is_contiguous() && x->numel()==(int64_t)hk*tiles*words),"packed V size");};
  check(v16p,V16Words);check(v8p,V8Words);check(v4p,V4Words);
  c10::cuda::CUDAGuard guard(v.device());
  const auto stream=at::cuda::getCurrentCUDAStream();
  const auto* vp=reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
  const bool identity=!v8p.has_value() && !v4p.has_value();
  if(!identity) partial_kernel<<<dim3(StatBlocks,hk),D,0,stream>>>(vp,partial.data_ptr<float>(),t,v.stride(1),v.stride(2));
  finalize_kernel<<<hk,D,0,stream>>>(partial.data_ptr<float>(),vstat.data_ptr<float>(),t,identity);
  pack_v_kernel<<<dim3(tiles,hk),256,0,stream>>>(vp,vstat.data_ptr<float>(),optional_ptr(v16p),optional_ptr(v8p),
      optional_ptr(v4p),t,v.stride(1),v.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor decode(torch::Tensor q,torch::Tensor k8c,c10::optional<torch::Tensor> k4c,torch::Tensor ka,
    c10::optional<torch::Tensor> ka4,torch::Tensor entry,torch::Tensor kinfo,torch::Tensor count,torch::Tensor fmt,torch::Tensor slot,
    torch::Tensor v8,torch::Tensor v4,torch::Tensor vstat,torch::Tensor kstat,torch::Tensor ktail,
    torch::Tensor vtail,torch::Tensor kcur,torch::Tensor vcur,torch::Tensor step,torch::Tensor out,torch::Tensor pm,
    torch::Tensor pl,torch::Tensor po,int64_t prompt,int64_t per,double scale) {
  TORCH_CHECK(q.scalar_type()==at::kBFloat16 && q.dim()==4 && q.size(2)==1 && q.size(3)==D && q.stride(3)==1,"q [1,Hq,1,128]");

  const int hq=q.size(1),hk=kstat.numel()/(3*D),tiles=(prompt+BN-1)/BN,max_tail=ktail.size(2);
  const int maxn=entry.size(1),splits=std::max<int>(1,(maxn+per-1)/per);
  TORCH_CHECK(hq%hk==0 && hq/hk<=MaxGroup && max_tail<=1024 && pm.numel()>=(int64_t)hq*splits && po.numel()>=(int64_t)hq*splits*D &&
      step.scalar_type()==at::kLong && step.is_cuda() && kcur.stride(3)==1 && vcur.stride(1)==kcur.stride(1) &&
      entry.scalar_type()==at::kInt && kinfo.scalar_type()==at::kInt && count.scalar_type()==at::kInt &&
      entry.size(0)==hk && kinfo.sizes()==entry.sizes() && slot.sizes()==entry.sizes() && slot.scalar_type()==at::kInt &&
      count.numel()==hk && fmt.numel()==tiles && ka.numel()==(int64_t)hk*tiles*4 &&
      k8c.dim()==2 && k8c.size(1)==KPairWords && v8.dim()==2 && v8.size(1)==V8Words && v4.dim()==2 && v4.size(1)==V4Words &&
      k4c.has_value()==ka4.has_value() &&
      (!k4c.has_value() || (k4c->dim()==2 && k4c->size(1)==K4Words && ka4->numel()==(int64_t)hk*tiles*4)),
      "decode geometry/buffers");
  const float sm_log2=(float)(scale*1.4426950408889634);
  c10::cuda::CUDAGuard guard(q.device());
  const auto stream=at::cuda::getCurrentCUDAStream();
  static bool configured=[]{
    C10_CUDA_CHECK(cudaFuncSetAttribute(decode_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,DecodeShared));
    return true;
  }();
  (void)configured;
  const auto* qp=reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  decode_kernel<<<dim3(splits,hk),NT,DecodeShared,stream>>>(qp,k8c.data_ptr<uint32_t>(),optional_ptr(k4c),
      ka.data_ptr<float>(),ka4.has_value()?ka4->data_ptr<float>():nullptr,entry.data_ptr<int32_t>(),
      kinfo.data_ptr<int32_t>(),count.data_ptr<int32_t>(),
      fmt.data_ptr<int32_t>(),slot.data_ptr<int32_t>(),v8.data_ptr<uint32_t>(),v4.data_ptr<uint32_t>(),
      pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),hq,hk,(int)prompt,(int)per,sm_log2,q.stride(1),maxn);
  decode_merge_kernel<<<hq,MergeThreads,0,stream>>>(pm.data_ptr<float>(),pl.data_ptr<float>(),po.data_ptr<float>(),vstat.data_ptr<float>(),
      kstat.data_ptr<float>(),qp,reinterpret_cast<__nv_bfloat16*>(ktail.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(vtail.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kcur.data_ptr<at::BFloat16>()),reinterpret_cast<const __nv_bfloat16*>(vcur.data_ptr<at::BFloat16>()),
      step.data_ptr<int64_t>(),reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),hq,hk,splits,max_tail,sm_log2,q.stride(1),
      ktail.stride(1),ktail.stride(2),kcur.stride(1),out.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t decode_occupancy() {
  int blocks=0;
  C10_CUDA_CHECK(cudaFuncSetAttribute(decode_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,DecodeShared));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,decode_kernel,NT,DecodeShared));
  return blocks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
  module.def("prepare_qk",&prepare_qk,"Per-token INT8 (and INT4) Q and centered per-block INT8 (and INT4) K operands");
  module.def("pack_v",&pack_v,"Normalized BF16 V' into FP16/INT8/INT4 PV operands");
  module.def("decode",&decode,"Single-token decode over prefill operands (tensor cores) plus a BF16 tail");
  module.def("decode_occupancy",&decode_occupancy,"Resident decode CTAs per SM");
  module.def("attention",&attention,"Unified per-tile skip / INT8 QK x FP16, INT8, INT4 PV / INT4 QK x INT8 PV");
}
