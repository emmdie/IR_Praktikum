/*
 * Copyright (c) 1993-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#pragma once

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/device_mdspan.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief multi-gpu mean squared error
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam TPB threads-per-block
 * @param out the output data (device pointer)
 * @param in1 the first input data
 * @param in1Desc descriptor for the first input data
 * @param in2 the second input data
 * @param in2Desc descriptor for the second input data
 * @param comm the communicator object
 * @param stream cuda stream where to launch work
 * @param root worker ID which is supposed to be considered as root
 * @param broadcastResult if false, only root process will have the result,
 *   else all ranks
 * @{
 */
void meanSquaredError(double* out,
                      const Matrix::Data<double>& in1,
                      const Matrix::PartDescriptor& in1Desc,
                      const Matrix::Data<double>& in2,
                      const Matrix::PartDescriptor& in2Desc,
                      const raft::comms::comms_t& comm,
                      cudaStream_t stream,
                      int root             = 0,
                      bool broadcastResult = true);
void meanSquaredError(float* out,
                      const Matrix::Data<float>& in1,
                      const Matrix::PartDescriptor& in1Desc,
                      const Matrix::Data<float>& in2,
                      const Matrix::PartDescriptor& in2Desc,
                      const raft::comms::comms_t& comm,
                      cudaStream_t stream,
                      int root             = 0,
                      bool broadcastResult = true);
/** @} */

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
