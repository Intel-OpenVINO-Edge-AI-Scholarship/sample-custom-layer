/*
 Copyright (C) 2018-2019 Intel Corporation
 SPDX-License-Identifier: Apache-2.0
*/

// ===============================================================================
// Generated file for Inference Engine extension for GPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// Refer to the section "Adding Your Own Kernels to the Inference Engine" in
// OpenVINO* documentation (either online or offline in
// <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
// to the corresponding section).
// ===============================================================================

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void pnorm_kernel(
     // insert pointers to inputs, outputs as arguments here
     // if your layer have one input and one output, argumants will be:
     const __global INPUT0_TYPE*  input0, __global OUTPUT0_TYPE* output
     )
{
    // Here kernel implementation should be added!

    const int size = sizeof(INPUT0_DIMS) / sizeof(INPUT0_DIMS[0]);
    // batch size
    const int B = INPUT0_DIMS[0];
    const int F = INPUT0_DIMS[1];
    const int Y = INPUT0_DIMS[2];
    // feature size
    const int X = INPUT0_DIMS[3];

    const int significant_ = 1;
    const int to_significant_ = 5;
    const float const_avg_ratio_ = 0.2;
    const int p = -1;
    const int group = 1;

    const int IF = F + INPUT0_LOWER_PADDING[1] + INPUT0_UPPER_PADDING[1];
    const int IY = Y + INPUT0_LOWER_PADDING[2] + INPUT0_UPPER_PADDING[2];
    const int IX = X + INPUT0_LOWER_PADDING[3] + INPUT0_UPPER_PADDING[3];

    const int OF = OUTPUT0_DIMS[1] + OUTPUT0_LOWER_PADDING[1] + OUTPUT0_UPPER_PADDING[1];
    const int OY = OUTPUT0_DIMS[2] + OUTPUT0_LOWER_PADDING[2] + OUTPUT0_UPPER_PADDING[2];
    const int OX = OUTPUT0_DIMS[3] + OUTPUT0_LOWER_PADDING[3] + OUTPUT0_UPPER_PADDING[3];

    int in_padding = INPUT0_LOWER_PADDING[2]*OX + INPUT0_LOWER_PADDING[3];
    int out_padding = OUTPUT0_LOWER_PADDING[2]*OX + OUTPUT0_LOWER_PADDING[3];

    int in_offset = get_global_id(0)*IF*IY*IX + get_global_id(1)*IX + in_padding;
    int out_offset = get_global_id(0)*OF*OY*OX + get_global_id(1)*OX + out_padding;

    // defining accumulator feature variables
    ACCUMULATOR_TYPE stationary = 0;
    ACCUMULATOR_TYPE alternate = 0;
    ACCUMULATOR_TYPE large = 0;
    ACCUMULATOR_TYPE weight1 = 0;
    ACCUMULATOR_TYPE weight2 = 0;

    // definining all the layer attributes

    float mul = 1000.0;
    int qty = 10;
    
    for (int i = 0; i < X; i++)
    {
        int index = in_offset + f*IY*IX + i;
        int out_index = out_offset + f*OY*OX + i;

        for(int j = significant_; j <= to_significant_; j++) {
            int4 q = pown(qty, j);
            float4 mul = convert_float4(q);
            int reci1 = convert_int_rte(1/input[index]*mul);
            int reci2 = convert_int_rte(1/input[index+1]*mul);
            int reci3 = convert_int_rte(1/input[index+2]*mul);
            int reci4 = convert_int_rte(1/input[index+3]*mul);
            int reci5 = convert_int_rte(1/input[index+4]*mul);
            float freci1 = convert_float(reci1) / mul;
            float freci2 = convert_float(reci2) / mul;
            float freci3 = convert_float(reci3) / mul;
            float freci4 = convert_float(reci4) / mul;
            float freci5 = convert_float(reci5) / mul;

            if (i == 0) {
                stationary += freci;
            } else if(i == 1) {
                alternate += freci;
            } else if(i == 2) {
                large += freci;
            } else if(i == 3) {
                weight1 += freci;
            } else if(i == 4) {
                weight2 += freci;
            }
        }

        if (i == 0) {
            output[out_index] = stationary;
        } else if(i == 1) {
            output[out_index] = alternate;
        } else if(i == 2) {
            output[out_index] = large;
        } else if(i == 3) {
            output[out_index] = weight1;
        } else if(i == 4) {
            output[out_index] = weight2;
        }
    }
}
