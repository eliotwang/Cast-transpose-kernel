import triton
import triton.language as tl


import torch

import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        #triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 1}, num_stages=2, num_warps=1),
        
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8), 
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        
                
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),            
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),         
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),          
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),            
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),           
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),          
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),           
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
                         
    ]

def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=4,num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=8, num_stages=3),            
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2}, num_warps=8, num_stages=3),         
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 4}, num_warps=4, num_stages=4),          
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'waves_per_eu': 2}, num_warps=4, num_stages=4),            
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 2}, num_warps=4, num_stages=4),           
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 3}, num_warps=4, num_stages=4),          
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'waves_per_eu': 3}, num_warps=4, num_stages=4),    
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8}, num_warps=2, num_stages=2),       
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'waves_per_eu': 3}, num_warps=1, num_stages=8),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'waves_per_eu': 8}, num_warps=1, num_stages=8),


        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'waves_per_eu': 8}, num_warps=8, num_stages=8),                 
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'waves_per_eu': 8}, num_warps=8, num_stages=8),           
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 8}, num_warps=8, num_stages=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'waves_per_eu': 3}, num_warps=8, num_stages=8), 
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8,  'waves_per_eu': 8}, num_warps=8, num_stages=8),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8,  'waves_per_eu': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8,  'waves_per_eu': 8}, num_warps=1, num_stages=4),
    ]
'''

def get_hip_autotune_config():
    return [
       triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8),      
    ]
'''
def get_autotune_config():
    #if is_cuda():
    #    return get_cuda_autotune_config()
    #else:
    return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N'],
)  
 


@triton.jit
def cast_transpose_kernel(
        input_ptr,
        scale_ptr,
        output_c_ptr,
        output_t_ptr,
        amax_ptr,
        M, N,
        stride_input,
        stride_output_c,
        stride_output_t,
        dtype: tl.constexpr, 
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr
):
    """Kernel for casting and transposing a tensor."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offsets_m[:, None] < M
    mask_n = offsets_n[None, :] < N

    mask_tm = offsets_m[None, :] < M
    mask_tn = offsets_n[:, None] < N

    input_ptrs = input_ptr + offsets_m[:, None] * stride_input + offsets_n[None, :] 
    
    input_ = tl.load(input_ptrs, mask=mask_m & mask_n)
    scale = tl.load(scale_ptr)
    input_ = input_.to(tl.float32)
    output_c = input_ * scale
   
    output_c = tl.cast(output_c, dtype) 
    tl.store(output_c_ptr + offsets_m[:, None] * stride_output_c + offsets_n[None, :], output_c, mask=mask_m & mask_n)

    output_t = output_c.trans()
   
    tl.store(output_t_ptr + offsets_n[:, None] * stride_output_t + offsets_m[None, :] , output_t, mask=mask_tn & mask_tm)
    amax = tl.max(tl.abs(input_))
    tl.atomic_max(amax_ptr, amax, sem='relaxed')
    #tl.store(output_t_ptr + offsets_n[None, :] * stride_output_t + offsets_m[:, None] , output_c, mask=mask_n & mask_m)
    
dtype_map = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float8_e5m2: tl.float8e5,
        torch.float8_e4m3fnuz: tl.float8e4b8
    }
  

def cast_transpose(input_tensor, scale_tensor, output_type: torch.dtype):
    M, N = input_tensor.shape
    output_c = torch.empty((M, N), device=input_tensor.device, dtype=output_type)
    output_t = torch.empty((N, M), device=input_tensor.device, dtype=output_type)
    amax = torch.zeros(1, device=input_tensor.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    

    dtype = dtype_map[output_type]
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    warm_up = 5
    for i in range(warm_up): 
        cast_transpose_kernel[grid](input_tensor, scale_tensor, output_c, output_t, amax, M, N,
                                 input_tensor.stride(0), output_c.stride(0), output_t.stride(0), dtype=dtype)
    
    freq = 10
    start_event.record()
    for i in range(freq):
        cast_transpose_kernel[grid](input_tensor, scale_tensor, output_c, output_t, amax, M, N,
                                 input_tensor.stride(0), output_c.stride(0), output_t.stride(0), dtype=dtype)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_us = start_event.elapsed_time(end_event) / freq * 1000
    print(f"GPU operation took {elapsed_time_us:.4f} us")

    return output_c, output_t, amax

def compare_vector(A, B):

    size_A = len(A)
    size_B = len(B)
    if size_A != size_B:
        return False

    for i in range(size_A):
        if abs(A[i] - B[i]) > 0.02:
            print("diff value is ", i, A[i], B[i])
            return False
    return True


def torch_cast_transpose(input_tensor, scale, output_type):
    scaled_tensor = scale * input_tensor
    casted_tensor = scaled_tensor.to(output_type)
    transposed_out = casted_tensor.transpose(0,1).contiguous()
    amax = torch.max(torch.abs(input_tensor.to(torch.float32)))
    return casted_tensor, transposed_out, amax


shapes = [(2048, 12288), (256, 65536), (65536, 128), (768, 1024)]
input_types = [torch.float32, torch.bfloat16, torch.float16]
#output_types = [torch.float32, torch.float16, torch.bfloat16, torch.float8_e5m2]#, torch.float8_e4m3fn]
output_types = [torch.float8_e5m2, torch.float8_e4m3fnuz]#
def test_cast_transpose():
    torch.manual_seed(0)
    for input_type in input_types:
        for output_type in output_types:
            for M, N in shapes:
                print(f"M = {M}, N = {N}, input_type = {input_type}, output_type = {output_type}")
                input_tensor = torch.randn((M, N), device='cuda', dtype=input_type)
                scale_tensor = torch.randn(1, dtype=torch.float32, device='cuda')

                output_c ,output_t, triton_amax = cast_transpose(input_tensor, scale_tensor, output_type)
                
                torch_c, torch_t, torch_amax = torch_cast_transpose(input_tensor, scale_tensor, output_type)
                #expected_output_c = input_tensor.to(output_type)
                #expected_output_t = expected_output_c.T
                #ref_outc = expected_output_c.to(torch.float32).view(-1)
                #outc = output_c.to(torch.float32).view(-1)
                #is_same = compare_vector(ref_outc, outc)       
                #if is_same == True:
                #    print("========PASS======")
                #else:
                #    print("========FAIL======")
                '''
                assert torch.allclose(output_c.to(torch.float32), torch_c.to(torch.float32), atol=1e-2), "Output C does not match expected output!"
                assert torch.allclose(output_t.to(torch.float32), torch_t.to(torch.float32), atol=1e-2), "Output T does not match expected output!"
                assert torch.allclose(torch_amax, triton_amax, atol=1e-2), "Triton amax does not match expected output!"
                print("✅ Triton outputs match expected outputs.")
                '''

if __name__  == "__main__":
    test_cast_transpose()