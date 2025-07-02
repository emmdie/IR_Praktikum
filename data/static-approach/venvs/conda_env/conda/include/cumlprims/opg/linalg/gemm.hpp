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
#include <raft/core/handle.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * @brief A multi gpu generalized matrix multiplication function.
 * This function performs
 * Z = X * Y
 * The X and Y matrix are distributed in blocks on different ranks.
 * First Y matrix is duplicated at each rank. It is multiplied with blocks of X
 * local to the rank.
 * \param      h          cuML handle object.
 * \param[out] outZParts  Result of the multiplication with size M x N.
 *                        Distributed across ranks according to inXDesc/outZDesc
 *                        descriptor.
 * \param[out] outZDesc   Descriptor for outZParts matrix. It has to be
 *                        same as inXDesc.
 * \param[in]  inX        Input matrix X with dimensions M x K. Distributed
 *                        across ranks according to inXDesc descriptor.
 * \param[in]  inXDesc    Descriptor for X matrix.
 * \param[in]  inY        Input matrix Y with dimensions K x N. Distributed
 *                        across ranks according to inYDesc descriptor.
 * \param[in]  inYDesc    Descriptor for Y matrix.
 * \param[in]  myRank     Rank of calling process.
 * \param[in]  stream     cuda stream on which work is launched.
 */

void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<float>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<float>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<float>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream);

void gemm(const raft::handle_t& h,
          std::vector<Matrix::Data<double>*>& outZParts,
          Matrix::PartDescriptor& outZDesc,
          std::vector<Matrix::Data<double>*>& inXParts,
          Matrix::PartDescriptor& inXDesc,
          std::vector<Matrix::Data<double>*>& inYParts,
          Matrix::PartDescriptor& inYDesc,
          int myRank,
          cudaStream_t stream);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
