# SPDX-License-Identifier: Apache-2.0

# VLEN=1024. Transpose the 16 pair-major vectors of one N128 weight
# group by interleaving 1-, 2-, 4- and 8-byte blocks. Every resulting
# register holds 8 columns of 16 signed-nibble pairs.
# a0 source, a1 scratch, a2 number of G32 groups.
.text
.p2align 3
.hidden buddy_k3_transpose_groups
.globl buddy_k3_transpose_groups
buddy_k3_transpose_groups:
1:
    vl8re8.v v0, (a0)
    addi t0, a0, 1024
    vl8re8.v v8, (t0)
    vsetvli t0, zero, e8, m1, ta, ma
    smt.vpack.vv v16, v0, v1, 0
    smt.vpack.vv v18, v2, v3, 0
    smt.vpack.vv v20, v4, v5, 0
    smt.vpack.vv v22, v6, v7, 0
    smt.vpack.vv v24, v8, v9, 0
    smt.vpack.vv v26, v10, v11, 0
    smt.vpack.vv v28, v12, v13, 0
    smt.vpack.vv v30, v14, v15, 0
    vsetvli t0, zero, e16, m1, ta, ma
    smt.vpack.vv v0, v16, v18, 0
    smt.vpack.vv v2, v17, v19, 0
    smt.vpack.vv v4, v20, v22, 0
    smt.vpack.vv v6, v21, v23, 0
    smt.vpack.vv v8, v24, v26, 0
    smt.vpack.vv v10, v25, v27, 0
    smt.vpack.vv v12, v28, v30, 0
    smt.vpack.vv v14, v29, v31, 0
    vsetvli t0, zero, e32, m1, ta, ma
    smt.vpack.vv v16, v0, v4, 0
    smt.vpack.vv v18, v1, v5, 0
    smt.vpack.vv v20, v2, v6, 0
    smt.vpack.vv v22, v3, v7, 0
    smt.vpack.vv v24, v8, v12, 0
    smt.vpack.vv v26, v9, v13, 0
    smt.vpack.vv v28, v10, v14, 0
    smt.vpack.vv v30, v11, v15, 0
    vsetvli t0, zero, e64, m1, ta, ma
    smt.vpack.vv v0, v16, v24, 0
    smt.vpack.vv v2, v17, v25, 0
    smt.vpack.vv v4, v18, v26, 0
    smt.vpack.vv v6, v19, v27, 0
    smt.vpack.vv v8, v20, v28, 0
    smt.vpack.vv v10, v21, v29, 0
    smt.vpack.vv v12, v22, v30, 0
    smt.vpack.vv v14, v23, v31, 0
    vs8r.v v0, (a1)
    addi t0, a1, 1024
    vs8r.v v8, (t0)
    li t0, 2048
    add t1, a0, t0
    add t2, a1, t0
    vl2re16.v v16, (t1)
    vs2r.v v16, (t2)
    addi t0, t0, 256
    add a0, a0, t0
    add a1, a1, t0
    addi a2, a2, -1
    bnez a2, 1b
    ret

# One M8 x N32 result. a0 packed weights, a1 activations, a2 weight
# scales, a3 four row-major 8x8 sums, a4 groups, a5 first K chunk.
# v0:7 hold four f32 tiles. v16:23 hold the four independent i32 dots.
.hidden buddy_k3_multiply32
.globl buddy_k3_multiply32
buddy_k3_multiply32:
    vsetvli t0, zero, e32, m8, ta, ma
    beqz a5, 2f
    vmv.v.i v0, 0
    j 3f
2:
    vl8re32.v v0, (a3)
3:
    vmv.v.i v16, 0
    vl4re8.v v12, (a0)
    vsetvli t0, zero, e8, m4, ta, ma
    vsra.vi v24, v12, 4
    vsll.vi v12, v12, 4
    vsra.vi v12, v12, 4
    vsetvli t0, zero, e8, m1, ta, ma
    vl2re16.v v10, (a1)
    vnsrl.wi v8, v10, 0
    vnsrl.wi v9, v10, 8
    vsetvli t0, zero, e32, m1, ta, ma
    smt.vmadot v16, v8, v12, i8
    smt.vmadot v18, v8, v13, i8
    smt.vmadot v20, v8, v14, i8
    smt.vmadot v22, v8, v15, i8
    smt.vmadot v16, v9, v24, i8
    smt.vmadot v18, v9, v25, i8
    smt.vmadot v20, v9, v26, i8
    smt.vmadot v22, v9, v27, i8
    vsetvli t0, zero, e16, m4, ta, ma
    # Restore the signed i16 group sum, including two's-complement wrap.
    vnsrl.wi v24, v16, 0
    vfwcvt.f.x.v v16, v24
    # Duplicate each set of eight column scales across the eight rows.
    li t0, 32
    vsetvli zero, t0, e16, m1, ta, ma
    vle16.v v24, (a2)
    vsetvli t0, zero, e16, m1, ta, ma
    smt.vpack.vv v26, v24, v24, 1
    smt.vpack.vv v28, v26, v26, 1
    smt.vpack.vv v24, v28, v28, 1
    smt.vpack.vv v26, v29, v29, 1
    vsetvli t0, zero, e16, m4, ta, ma
    vfwcvt.f.f.v v8, v24
    # Duplicate each activation scale across eight output columns.
    addi t1, a1, 256
    vsetivli zero, 8, e32, m1, ta, ma
    vle32.v v24, (t1)
    vsetvli t0, zero, e32, m1, ta, ma
    smt.vpack.vv v26, v24, v24, 0
    smt.vpack.vv v28, v26, v26, 0
    smt.vpack.vv v24, v28, v28, 0
    vmv2r.v v26, v24
    vmv2r.v v28, v24
    vmv2r.v v30, v24
    vsetvli t0, zero, e32, m8, ta, ma
    vfmul.vv v8, v8, v24
    vfmacc.vv v0, v8, v16
    li t0, 2304
    add a0, a0, t0
    add a2, a2, t0
    addi a1, a1, 288
    addi a4, a4, -1
    bnez a4, 3b
    vs8r.v v0, (a3)
    ret
