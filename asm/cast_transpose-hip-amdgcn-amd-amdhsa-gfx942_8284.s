	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.section	.text._ZL24set_hip_f8_bias_mode_bitb,"axG",@progbits,_ZL24set_hip_f8_bias_mode_bitb,comdat
	.globl	_ZL24set_hip_f8_bias_mode_bitb  ; -- Begin function _ZL24set_hip_f8_bias_mode_bitb
	.p2align	8
	.type	_ZL24set_hip_f8_bias_mode_bitb,@function
_ZL24set_hip_f8_bias_mode_bitb:         ; @_ZL24set_hip_f8_bias_mode_bitb
; %bb.0:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZL24set_hip_f8_bias_mode_bitb
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 4
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  0
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 0
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZL24set_hip_f8_bias_mode_bitb,"axG",@progbits,_ZL24set_hip_f8_bias_mode_bitb,comdat
.Lfunc_end0:
	.size	_ZL24set_hip_f8_bias_mode_bitb, .Lfunc_end0-_ZL24set_hip_f8_bias_mode_bitb
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 4
; NumSgprs: 6
; NumVgprs: 0
; NumAgprs: 0
; TotalNumVgprs: 0
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 6
; NumVGPRsForWavesPerEU: 1
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm ; -- Begin function _Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm
	.globl	_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm
	.p2align	8
	.type	_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm,@function
_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm: ; @_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x8
	s_waitcnt lgkmcnt(0)
	s_cmp_eq_u64 s[6:7], 0
	s_cselect_b64 s[4:5], -1, 0
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB1_2
; %bb.1:
	s_load_dword s3, s[6:7], 0x0
	s_waitcnt lgkmcnt(0)
	v_cmp_neq_f32_e64 s[4:5], s3, 1.0
.LBB1_2:                                ; %Flow177
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB1_20
; %bb.3:
	s_load_dwordx4 s[4:7], s[0:1], 0x30
	s_load_dwordx2 s[16:17], s[0:1], 0x20
	s_load_dwordx2 s[12:13], s[0:1], 0x0
	s_load_dwordx4 s[8:11], s[0:1], 0x10
	s_mov_b32 s3, 0
	s_waitcnt lgkmcnt(0)
	s_lshr_b64 s[14:15], s[6:7], 5
	v_mov_b64_e32 v[2:3], s[14:15]
	v_cmp_lt_u64_e32 vcc, s[2:3], v[2:3]
	s_mov_b64 s[18:19], 0
	s_cbranch_vccnz .LBB1_5
; %bb.4:
	v_mov_b32_e32 v1, s6
	v_alignbit_b32 v1, s7, v1, 5
	v_cvt_f32_u32_e32 v2, v1
	v_readfirstlane_b32 s18, v1
	s_sub_i32 s19, 0, s18
	v_rcp_iflag_f32_e32 v2, v2
	s_nop 0
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_nop 0
	v_readfirstlane_b32 s20, v2
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s2, s20
	s_mul_i32 s21, s19, s18
	s_sub_i32 s21, s2, s21
	s_add_i32 s20, s19, 1
	s_sub_i32 s22, s21, s18
	s_cmp_ge_u32 s21, s18
	s_cselect_b32 s19, s20, s19
	s_cselect_b32 s21, s22, s21
	s_add_i32 s20, s19, 1
	s_cmp_ge_u32 s21, s18
	s_cselect_b32 s18, s20, s19
	s_mov_b32 s19, s3
.LBB1_5:
	s_load_dwordx2 s[0:1], s[0:1], 0x28
	s_cmp_eq_u64 s[16:17], 0
	s_cbranch_scc1 .LBB1_7
; %bb.6:
	s_load_dword s3, s[16:17], 0x0
	s_branch .LBB1_8
.LBB1_7:
	s_mov_b32 s3, 1.0
.LBB1_8:                                ; %.preheader96
	s_mul_i32 s15, s18, s15
	s_mul_hi_u32 s16, s18, s14
	s_add_i32 s16, s16, s15
	s_mul_i32 s14, s18, s14
	s_sub_u32 s14, s2, s14
	s_subb_u32 s15, 0, s16
	v_and_b32_e32 v10, 31, v0
	s_lshl_b64 s[14:15], s[14:15], 5
	s_lshl_b64 s[16:17], s[18:19], 6
	v_lshrrev_b32_e32 v1, 5, v0
	v_lshlrev_b32_e32 v11, 1, v10
	v_or_b32_e32 v4, s14, v1
	v_or_b32_e32 v2, s16, v11
	v_mov_b32_e32 v3, s17
	v_mad_u64_u32 v[12:13], s[18:19], v4, s4, v[2:3]
	s_mul_i32 s2, s15, s4
	v_mul_lo_u32 v2, v4, s5
	v_add3_u32 v13, s2, v13, v2
	v_lshl_add_u64 v[4:5], v[12:13], 2, s[12:13]
	s_lshl_b64 s[12:13], s[4:5], 5
	global_load_dwordx2 v[2:3], v[4:5], off
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[12:13]
	global_load_dwordx2 v[4:5], v[6:7], off
	v_lshl_add_u64 v[8:9], v[6:7], 0, s[12:13]
	global_load_dwordx2 v[6:7], v[8:9], off
	v_lshl_add_u64 v[8:9], v[8:9], 0, s[12:13]
	global_load_dwordx2 v[8:9], v[8:9], off
	v_lshlrev_b32_e32 v14, 1, v1
	s_movk_i32 s2, 0x42
	v_or_b32_e32 v18, s16, v14
	v_mad_u32_u24 v28, v10, s2, v14
	v_mad_u32_u24 v11, v1, s2, v11
	s_mul_i32 s2, s17, s6
	v_mov_b32_e32 v15, s15
	v_or_b32_e32 v14, s14, v10
	v_mul_lo_u32 v19, v18, s7
	v_mad_u64_u32 v[16:17], s[12:13], v18, s6, 0
	v_or_b32_e32 v20, 16, v18
	v_or_b32_e32 v21, 32, v18
	v_or_b32_e32 v22, 48, v18
	v_lshl_add_u64 v[14:15], v[14:15], 1, s[10:11]
	v_add3_u32 v17, v17, v19, s2
	v_mul_lo_u32 v24, v20, s7
	v_mad_u64_u32 v[18:19], s[10:11], v20, s6, 0
	v_mul_lo_u32 v25, v21, s7
	v_mad_u64_u32 v[20:21], s[10:11], v21, s6, 0
	v_mul_lo_u32 v26, v22, s7
	v_mad_u64_u32 v[22:23], s[10:11], v22, s6, 0
	s_lshl_b64 s[4:5], s[4:5], 4
	v_add3_u32 v19, v19, v24, s2
	v_add3_u32 v21, v21, v25, s2
	v_add3_u32 v23, v23, v26, s2
	v_lshl_add_u64 v[12:13], v[12:13], 1, s[8:9]
	v_lshl_add_u64 v[16:17], v[16:17], 1, v[14:15]
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[14:15]
	v_lshl_add_u64 v[20:21], v[20:21], 1, v[14:15]
	v_lshl_add_u64 v[14:15], v[22:23], 1, v[14:15]
	v_lshl_add_u64 v[22:23], v[12:13], 0, s[4:5]
	v_lshl_add_u64 v[24:25], v[22:23], 0, s[4:5]
	v_lshl_add_u64 v[26:27], v[24:25], 0, s[4:5]
	s_waitcnt vmcnt(3) lgkmcnt(0)
	v_fma_mixlo_f16 v29, s3, v2, 0
	v_mov_b32_e32 v31, v29
	s_waitcnt vmcnt(2)
	v_fma_mixlo_f16 v32, s3, v4, 0
	ds_write_b16 v28, v29
	v_fma_mixhi_f16 v31, s3, v3, 0
	v_mov_b32_e32 v29, v32
	s_waitcnt vmcnt(1)
	v_fma_mixlo_f16 v34, s3, v6, 0
	global_store_dword v[12:13], v31, off
	v_fma_mixhi_f16 v29, s3, v5, 0
	s_waitcnt vmcnt(1)
	v_fma_mixlo_f16 v13, s3, v8, 0
	v_mov_b32_e32 v12, v34
	global_store_dword v[22:23], v29, off
	v_mov_b32_e32 v22, v13
	v_fma_mixhi_f16 v12, s3, v7, 0
	v_fma_mixhi_f16 v22, s3, v9, 0
	ds_write_b16 v28, v32 offset:16
	ds_write_b16 v28, v34 offset:32
	ds_write_b16 v28, v13 offset:48
	global_store_dword v[24:25], v12, off
	global_store_dword v[26:27], v22, off
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v12, v11
	ds_read_u16 v13, v11 offset:528
	ds_read_u16 v22, v11 offset:1056
	ds_read_u16 v23, v11 offset:1584
	v_fma_mixlo_f16 v30, s3, v3, 0
	v_fma_mixlo_f16 v33, s3, v5, 0
	v_fma_mixlo_f16 v35, s3, v7, 0
	v_fma_mixlo_f16 v24, s3, v9, 0
	s_waitcnt lgkmcnt(3)
	global_store_short v[16:17], v12, off
	s_waitcnt lgkmcnt(2)
	global_store_short v[18:19], v13, off
	s_waitcnt lgkmcnt(1)
	global_store_short v[20:21], v22, off
	s_waitcnt lgkmcnt(0)
	global_store_short v[14:15], v23, off
	s_barrier
	ds_write_b16 v28, v30
	ds_write_b16 v28, v33 offset:16
	ds_write_b16 v28, v35 offset:32
	ds_write_b16 v28, v24 offset:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_u16 v22, v11
	s_lshl_b64 s[2:3], s[6:7], 1
	v_lshl_add_u64 v[12:13], v[16:17], 0, s[2:3]
	ds_read_u16 v16, v11 offset:528
	ds_read_u16 v17, v11 offset:1056
	ds_read_u16 v11, v11 offset:1584
	s_cmp_eq_u64 s[0:1], 0
	s_waitcnt lgkmcnt(3)
	global_store_short v[12:13], v22, off
	v_lshl_add_u64 v[12:13], v[18:19], 0, s[2:3]
	s_waitcnt lgkmcnt(2)
	global_store_short v[12:13], v16, off
	v_lshl_add_u64 v[12:13], v[20:21], 0, s[2:3]
	s_waitcnt lgkmcnt(1)
	global_store_short v[12:13], v17, off
	v_lshl_add_u64 v[12:13], v[14:15], 0, s[2:3]
	s_waitcnt lgkmcnt(0)
	global_store_short v[12:13], v11, off
	s_barrier
	s_cbranch_scc1 .LBB1_20
; %bb.9:
	v_max_f32_e64 v2, |v2|, |v2|
	v_max_f32_e32 v2, 0, v2
	v_max3_f32 v2, |v4|, |v3|, v2
	v_max3_f32 v2, |v6|, |v5|, v2
	v_max3_f32 v2, |v8|, |v7|, v2
	v_max_f32_e32 v2, v2, v2
	v_max_f32_e64 v3, |v9|, |v9|
	v_max_f32_e32 v2, v3, v2
	v_mbcnt_lo_u32_b32 v3, -1, 0
	v_mbcnt_hi_u32_b32 v4, -1, v3
	v_and_b32_e32 v6, 31, v4
	v_cmp_gt_u32_e32 vcc, 16, v6
	s_mov_b64 s[2:3], exec
	s_nop 0
	v_cndmask_b32_e64 v3, 0, 1, vcc
	v_lshlrev_b32_e32 v3, 4, v3
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v3, v3, v2
	v_cmp_gt_u32_e32 vcc, 24, v6
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v3, v3, v3
	v_max_f32_e32 v2, v2, v3
	v_cndmask_b32_e64 v3, 0, 1, vcc
	v_lshlrev_b32_e32 v3, 3, v3
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v3, v3, v2
	v_cmp_gt_u32_e32 vcc, 28, v6
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v3, v3, v3
	v_max_f32_e32 v3, v2, v3
	v_cndmask_b32_e64 v2, 0, 1, vcc
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_lshl_u32 v2, v2, v4, 2
	ds_bpermute_b32 v5, v2, v3
	v_cmp_gt_u32_e32 vcc, 30, v6
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v5, v5, v5
	v_max_f32_e32 v5, v3, v5
	v_cndmask_b32_e64 v3, 0, 1, vcc
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_lshl_u32 v3, v3, v4, 2
	ds_bpermute_b32 v7, v3, v5
	v_cmp_ne_u32_e32 vcc, 31, v6
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v7, v7, v7
	v_addc_co_u32_e32 v4, vcc, 0, v4, vcc
	v_max_f32_e32 v5, v5, v7
	v_lshlrev_b32_e32 v4, 2, v4
	ds_bpermute_b32 v6, v4, v5
	v_cmp_eq_u32_e32 vcc, 0, v10
	s_cmp_lg_u64 vcc, 0
	s_cmov_b64 exec, vcc
	s_cbranch_scc0 .LBB1_11
; %bb.10:
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v6, v6, v6
	v_max_f32_e32 v5, v5, v5
	v_max_f32_e32 v5, v5, v6
	v_lshlrev_b32_e32 v6, 2, v1
	ds_write_b32 v6, v5 offset:2112
	s_or_b64 exec, exec, s[2:3]
.LBB1_11:
	v_cmp_eq_u32_e32 vcc, 0, v1
	s_mov_b64 s[2:3], exec
	v_mov_b32_e32 v1, 0
	s_cmp_lg_u64 vcc, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cmov_b64 exec, vcc
	s_cbranch_scc0 .LBB1_15
; %bb.12:
	v_cmp_gt_u32_e32 vcc, 8, v0
	s_mov_b64 s[4:5], exec
	v_mov_b32_e32 v1, 0
	s_cmp_lg_u64 vcc, 0
	s_cmov_b64 exec, vcc
	s_cbranch_scc0 .LBB1_14
; %bb.13:
	v_lshlrev_b32_e32 v1, 2, v0
	ds_read_b32 v1, v1 offset:2112
	s_or_b64 exec, exec, s[4:5]
.LBB1_14:
	s_waitcnt lgkmcnt(0)
	ds_bpermute_b32 v2, v2, v1
	v_max_f32_e32 v1, v1, v1
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v2, v2, v2
	v_max_f32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v3, v1
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v2, v2, v2
	v_max_f32_e32 v1, v1, v2
	ds_bpermute_b32 v2, v4, v1
	s_waitcnt lgkmcnt(0)
	v_max_f32_e32 v2, v2, v2
	v_max_f32_e32 v1, v1, v2
	s_or_b64 exec, exec, s[2:3]
.LBB1_15:                               ; %_ZN18transformer_engine10reduce_maxILi8EfEET0_S1_i.exit
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_cmp_lg_u64 vcc, 0
                                        ; kill: def $sgpr2_sgpr3 killed $exec
	s_cmov_b64 exec, vcc
	s_cbranch_scc0 .LBB1_20
; %bb.16:
	s_mov_b64 s[2:3], exec
	s_brev_b32 s4, 1
.LBB1_17:                               ; %ComputeLoop
                                        ; =>This Inner Loop Header: Depth=1
	s_ff1_i32_b64 s5, s[2:3]
	v_readlane_b32 s8, v1, s5
	s_lshl_b64 s[6:7], 1, s5
	s_max_i32 s4, s4, s8
	s_andn2_b64 s[2:3], s[2:3], s[6:7]
	s_cmp_lg_u64 s[2:3], 0
	s_cbranch_scc1 .LBB1_17
; %bb.18:                               ; %ComputeEnd
	v_mbcnt_lo_u32_b32 v0, exec_lo, 0
	v_mbcnt_hi_u32_b32 v0, exec_hi, v0
	v_cmp_eq_u32_e32 vcc, 0, v0
	s_xor_b64 s[2:3], vcc, exec
	s_cmp_lg_u64 vcc, 0
	s_cmov_b64 exec, vcc
	s_cbranch_scc0 .LBB1_20
; %bb.19:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, s4
	global_atomic_smax v0, v1, s[0:1]
.LBB1_20:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm
		.amdhsa_group_segment_fixed_size 2144
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length  0
		.amdhsa_user_sgpr_kernarg_preload_offset  0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 36
		.amdhsa_next_free_sgpr 23
		.amdhsa_accum_offset 36
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm, .Lfunc_end1-_Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1668
; NumSgprs: 29
; NumVgprs: 36
; NumAgprs: 0
; TotalNumVgprs: 36
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 2144 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 29
; NumVGPRsForWavesPerEU: 36
; AccumOffset: 36
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 8
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_5731b28403059ece,@object ; @__hip_cuid_5731b28403059ece
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5731b28403059ece
__hip_cuid_5731b28403059ece:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5731b28403059ece, 1

	.ident	"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.2.1 24355 77cf9ad00e298ed06e06aec0f81009510f545714)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5731b28403059ece
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           1
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 4
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _ZL24set_hip_f8_bias_mode_bitb
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .sgpr_spill_count: 0
    .symbol:         _ZL24set_hip_f8_bias_mode_bitb.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     0
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           8
        .value_kind:     by_value
      - .offset:         56
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 2144
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm
    .private_segment_fixed_size: 0
    .sgpr_count:     29
    .sgpr_spill_count: 0
    .symbol:         _Z31cast_transpose_optimized_kernelPKfS0_P6__halfS2_S0_Pfmm.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     36
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
