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
 * \brief       Multi-GPU SVD decomposition for tall (rows >= columns) matrices
 * \param       h             cuML handle object.
 * \param[out]  sVector       A vector of size 1 x N with eigen values of
                              in matrix.
 * \param[out]  uParts        Parts of output U matrix from SVD decomposition
                              with size M x N. It is distributed among ranks.
                              Descriptor desc describes the matrix.
 * \param[out]  vMatrix       Full output V matrix from SVD decomposition with
                              size N x N. It is duplicated on all ranks.
 * \param[in]   genUMatrix    Currently ignored.
                              U matrix is generated only if this is true.
 * \param[in]   genVMatrix    Currently ignored.
                              V matrix is generated only if this is true.
 * \param[in]   tolerance     Error tolerance used for single GPU SVD.
                              Algorithm stops when the error is below
                              tolerance
 * \param[in]   maxSweeps     Number of sweeps in the single GPU SVD using
                              Jacobi algorithm. More sweeps provide better
                              accuracy.
 * \parms[in]   inParts       Parts of the tall input matrix, distributed among
                              ranks. The size of in matrix is M x N,
                              where M >> N.
 * \param[in]   desc          Discriptor of in matrix (inParts) and U matrix
                              (uParts).
 * \param       myRank        MPI rank of the process
 */

void svdQR(const raft::handle_t& h,
           float* sVector,
           std::vector<Matrix::Data<float>*>& uMatrixParts,
           float* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           float tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<float>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank);

void svdQR(const raft::handle_t& h,
           double* sVector,
           std::vector<Matrix::Data<double>*>& uMatrixParts,
           double* vMatrixParts,
           bool genUMatrix,
           bool genVMatrix,
           double tolerance,
           int maxSweeps,
           std::vector<Matrix::Data<double>*>& inParts,
           Matrix::PartDescriptor& desc,
           int myRank);

}  // end namespace opg
}  // end namespace LinAlg
}  // end namespace MLCommon
