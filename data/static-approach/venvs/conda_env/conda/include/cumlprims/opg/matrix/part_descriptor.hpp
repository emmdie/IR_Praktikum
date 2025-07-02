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

#include <stdint.h>

#include <cumlprims/opg/matrix/data.hpp>
#include <ostream>
#include <set>
#include <vector>

namespace MLCommon {
namespace Matrix {

/** Describes the data layout */
enum Layout {
  /** row major layout */
  LayoutRowMajor = 0,
  /** column major layout */
  LayoutColMajor
};

struct RankSizePair {
  RankSizePair() : rank(-1), size(0) {}

  RankSizePair(int _rank, size_t _size) : rank(_rank), size(_size) {}

  int rank;

  /**
   * Total number of rows
   */
  size_t size;
};

struct PartDescriptor {
  /** total number of rows */
  size_t M;
  /** total number of columns */
  size_t N;

  int rank;

  Layout layout;
  /** mapping of each block (in col-major order) to the device that owns it */
  std::vector<RankSizePair*> partsToRanks;

  /**
   * @brief For a given matrix and block-sizes construct the corresponding
   *  descriptor for it. This is useful when we are dealing with standard
   *  row/column-wise block-cyclic data distribution, as seen in other popular
   *  multi-node packages like magma etc.
   * @param _M total number of rows of this matrix
   * @param _N total number of columns
   * @param _partsToRanks mapping of ranks to parts and sizes
   */
  PartDescriptor(size_t _M,
                 size_t _N,
                 const std::vector<RankSizePair*>& _partsToRanks,
                 int rank,
                 Layout _layout = LayoutColMajor);

  /** total number of blocks across all workers */
  int totalBlocks() const { return partsToRanks.size(); }

  /** Count the total number of blocks owned by a given rank */
  int totalBlocksOwnedBy(int rank) const;

  std::set<int> uniqueRanks();

  std::vector<size_t> startIndices() const;

  std::vector<size_t> startIndices(int rank) const;

  /**
   * @brief Returns the vector of blocks (each identified by linearBLockIndex)
   * owned by the given rank
   */
  std::vector<RankSizePair*> blocksOwnedBy(int rank) const;

  /** Count the total number of matrix elements owned by a given rank */
  size_t totalElementsOwnedBy(int rank) const;

  friend std::ostream& operator<<(std::ostream& os, const PartDescriptor& desc);
  friend bool operator==(const PartDescriptor& a, const PartDescriptor& b);
};

/** Print matrix descriptor in human readable form */
std::ostream& operator<<(std::ostream& os, const PartDescriptor& desc);

/** compare 2 descriptor objects */
bool operator==(const PartDescriptor& a, const PartDescriptor& b);

};  // end namespace Matrix
};  // end namespace MLCommon
