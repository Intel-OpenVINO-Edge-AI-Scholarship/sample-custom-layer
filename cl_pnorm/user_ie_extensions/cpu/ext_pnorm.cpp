/*
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

// ===============================================================================
// Generated file for Inference Engine extension for CPU plugin
//
// IMPLEMENT YOUR KERNEL HERE.
//
// You need to edit this file in order to:
//  1. initialize parameters (in constructor)
//  2. implement inference logic (in execute() method)
//
// Refer to the section "Adding Your Own Kernels to the Inference Engine" in
// OpenVINO* documentation (either online or offline in
// <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
// to the corresponding section).
// ===============================================================================

#include "ext_list.hpp"
#include "ext_base.hpp"
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cassert>
#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#include <immintrin.h>
#endif
#include "ie_parallel.hpp"
#include "tbb/blocked_range.h"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class PNormLayer : public CNNLayer {
public:
  /**
   * @brief A default constructor. Constructs a WeightableLayer instance and initiates layer parameters with the given values
   * @param prms Initial layer parameters
   */
  explicit PNormLayer(const LayerParams &prms) : CNNLayer(prms) {}

  /**
   * @brief A virtual destructor
   */
  virtual ~PNormLayer() = default;

  /**
   * @brief Constructs a WeightableLayer instance and initiates layer parameters with the given values
   */
  using CNNLayer::CNNLayer;

  /**
   * @brief Layer name
   */
  int significant_ = 1;
  /**
   * @brief Layer type
   */
  int to_significant_ = 5;
  /**
   * @brief Layer name
   */
  float const_avg_ratio_ = 0.167;
  /**
   * @brief Layer name
   */
  int p;
  /**
   * @brief Layer name
   */
  int group;

};

struct pNormArray {
  double stationary;
  double alternate;
  double large;
  double weight1;
  double weight2;

  pNormArray& operator+=(const pNormArray& y)
  {
    return *this;
  }

  pNormArray& operator +(const pNormArray& y)
  {
    stationary += y.stationary;
    alternate += y.alternate;
    large += y.large;
    weight1 += y.weight1;
    weight2 += y.weight2;

    return *this;
  }

  pNormArray operator /(const pNormArray& y)
  {
    return pNormArray{stationary/y.stationary, alternate/y.alternate, 
    large/y.large, weight1/y.weight1, weight2/y.weight2};
  }

  pNormArray operator *(const pNormArray& y)
  {
    return pNormArray{stationary*y.stationary, alternate*y.alternate, 
    large*y.large, weight1*y.weight1, weight2*y.weight2};
  }
};

class pNormImpl: public ExtLayerBase {
public:
    explicit pNormImpl(const CNNLayer* layer) {
        try {
            // LayerSetUp
            // Read parameters from IR and/or initialise them here.
            // Implemented functions for reading parameters are:

            significant_ = layer->GetParamAsInt("significant");
            to_significant_ = layer->GetParamAsInt("to_significant");
            const_avg_ratio_ = layer->GetParamAsFloat("const_avg_ratio");
            
            // set configuration: specify data format for layer
            // more information about data formats you can find in "Inference Engine Memory primitives" in OpenVINO* documentation
            // (either online or offline in <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
            // to the corresponding section). 
            addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
    }

    pNormArray roundf(const pNormArray x, float value)
    {
      float mul = (float) powf(10.0, value);
      return pNormArray{std::roundf(x.stationary * mul) / mul, std::roundf(x.alternate * mul) / mul, 
      std::roundf(x.large * mul) / mul, std::roundf(x.weight1 * mul) / mul, 
      std::roundf(x.weight2 * mul) / mul};
    }

    StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                       ResponseDesc *resp) noexcept override {
        // Add here implementation for layer inference
        // Examples of implementations you can find in Inerence Engine tool samples/extenstions folder
        
        float* src_data = inputs[0]->buffer();
        float* dst_data = outputs[0]->buffer();

        // Get the dimensions from the input (output dimensions are the same)  
        SizeVector dims = inputs[0]->getTensorDesc().getDims();

        // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
        int batch_size = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
        int features = static_cast<int>((dims.size() > 1) ? dims[1] : 1);

        // significant digits used for calculation of pNorm
        std::vector<pNormArray> pNormVec(batch_size);

        pNormArray single{1.0, 1.0, 1.0, 1.0};
        pNormArray ratio{const_avg_ratio_, 
        const_avg_ratio_, const_avg_ratio_, 
        const_avg_ratio_, const_avg_ratio_};
        
        // do rounding operations
        parallel_for(0, batch_size, [&](int i) {
          pNormArray pNorm{src_data[i*features], src_data[i*features+1], src_data[i*features+2], 
          src_data[i*features + 3], src_data[i*features + 4]};
          pNorm = parallel_sum(to_significant_- significant_ + 1, 
          pNorm, [&](size_t s)->pNormArray {
            pNorm = single / pNorm;
            pNorm = roundf(pNorm, (float) (s + significant_));
            return pNorm;
          });
          pNorm = pNorm * ratio;
          pNormVec[i] = pNorm;
        });

        // write data
        for(int i = 0; i < batch_size; i++) {
          dst_data[i*features+0] = pNormVec[i].stationary;
          dst_data[i*features+1] = pNormVec[i].alternate;
          dst_data[i*features+2] = pNormVec[i].large;
          dst_data[i*features+3] = pNormVec[i].weight1;
          dst_data[i*features+4] = pNormVec[i].weight2;
        }

    }

// attributes of the layer
private:
    int significant_ = 1;
    int to_significant_ = 5;
    float const_avg_ratio_ = 0.167;
    int p;
    int group;
};

REG_FACTORY_FOR(ImplFactory<pNormImpl>, PNORM);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
