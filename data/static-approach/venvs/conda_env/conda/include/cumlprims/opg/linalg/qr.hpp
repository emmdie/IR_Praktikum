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
#include <raft/core/handle.hpp>

namespace MLCommon {
namespace LinAlg {
namespace opg {

/**
 * \brief    Multi-GPU QR decomposition for tall (rows > columns) matrices.
 *
 * This function implements multi-GPU QR decomposition for tall matrices
 * descibed in https://arxiv.org/abs/1301.1071.
 * \param       h               cuML handle object.
 * \param[out]  outQParts       Factor Q of input matrix QR decomposition. Q
                                matrix is distributed among ranks according to
                                desc description. The size of Q matrix is
                                same as size of 'in' matrix, i.e. M x N.
 * \param[out]  outR            Factor R of input matrix QR decomposition. R is
                                N x N matrix and is duplicated on all ranks.
                                Note that, only upper triangular part of outR,
                                including diagonal, is modified. Lower
                                triangular part is left as it is.
 * \param[in]   inParts         The tall input matrix, distributed among ranks.
                                The size of 'in' matrix is M x N, where M > N.
 * \param[in]   desc            Description of input matrix. The output matrix
                                'outQParts' follows the same description.
 * \param       myRank          MPI rank of the process.
 */

void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<float>*>& outQParts,
              float* outR,
              std::vector<Matrix::Data<float>*>& inXParts,
              Matrix::PartDescriptor& desc,
              int myRank);

void qrDecomp(const raft::handle_t& h,
              std::vector<Matrix::Data<double>*>& outQParts,
              double* outR,
              std::vector<Matrix::Data<double>*>& inXParts,
              Matrix::PartDescriptor& desc,
              int myRank);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
