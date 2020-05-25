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
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

class LOGNORMLayer : public CNNLayer {
public:
  /**
   * @brief A default constructor. Constructs a WeightableLayer instance and initiates layer parameters with the given values
   * @param prms Initial layer parameters
   */
  explicit LOGNORMLayer(const LayerParams &prms) : CNNLayer(prms) {}

  /**
   * @brief A virtual destructor
   */
  virtual ~LOGNORMLayer() = default;

  /**
   * @brief Constructs a WeightableLayer instance and initiates layer parameters with the given values
   */
  using CNNLayer::CNNLayer;

  /**
   * @brief Layer name
   */
  float scale_ = 1000.0;
  /**
   * @brief Layer type
   */
  float negate_ = 1.0;

};

class LOGNORMImpl: public ExtLayerBase {
public:
    explicit LOGNORMImpl(const CNNLayer* layer) {
        try {
            // LayerSetUp
            // Read parameters from IR and/or initialise them here.
            // Implemented functions for reading parameters are:
            
            scale_ = layer->GetParamAsInt("scale");
            negate_ = layer->GetParamAsInt("negate");
            
            // set configuration: specify data format for layer
            // more information about data formats you can find in "Inference Engine Memory primitives" in OpenVINO* documentation
            // (either online or offline in <INSTALL_DIR>/deployment_tools/documentation/docs/index.html an then navigate
            // to the corresponding section). 
            addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
        } catch (InferenceEngine::details::InferenceEngineException &ex) {
            errorMsg = ex.what();
        }
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

        std::vector<float> feature_mean(batch_size);
        feature_mean.fill(0.0f)
        std::vector<float> feature_min(batch_size);
        feature_min.fill(0.0f)

        parallel_for(0, batch_size, [&](int ithr) {
            float sum = 0.0f;
            sum = parallel_sum(features, 
            sum, [&](size_t s)->float {
                return s;
            });
            feature_mean[ithr] = sum / features;
        });

        parallel_for(0, batch_size, [&](int ithr) {
            float min_f = src_data[ithr*batch_size];
            for(int i = 1; i < features; i++) {
                min_f = std::min(min_f, src_data[ithr*batch_size+i]);
            }
            feature_min[ithr] = min_f;
        });
        
        // Perform (in parallel) the hyperbolic cosine given by: 
        //    lognorm(x) = scale * e^(x-mean) + (x-mean)^2 + 
        //                x.min() * log((x-mean)^2)
        parallel_for2d(batch_size, features, [&](int b, int f) {
            float factor = (std::exp(src_data[b*features+f]-feature_mean[f]) + 
            std::pow(src_data[b*features+f]-feature_mean[f],2) + 
            feature_min[b] * std::log(std::pow(src_data[b*features+f]-feature_mean[f],2)));
            dst_data[b*features+f] = scale_* factor;
        });
        return OK;
    }

// attributes of the layer
private:
    float scale_ = 1000.0;
    float negate_ = 1.0;
};

REG_FACTORY_FOR(ImplFactory<LOGNORMImpl>, LOGNORM);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
