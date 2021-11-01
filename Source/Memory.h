
/*
    Copyright notice:
    This project is based on pbrt, with necessary modifications.
    Any files that are originally part of pbrt will contain the following copyright notice.
*/

/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#pragma once

#include "DPHPC.h"

NAMESPACE_DPHPC_BEGIN

// Memory Declarations
#define ARENA_ALLOC(arena, Type) new ((arena).Alloc(sizeof(Type))) Type

void* AllocAligned(size_t size);
template <typename T>
T* AllocAligned(size_t count) {
	return (T*)AllocAligned(count * sizeof(T));
}

void FreeAligned(void*);

class alignas(L1_CACHE_LINE_SIZE)
MemoryArena {
public:
	// MemoryArena Public Methods
	MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) {}
	~MemoryArena() {
		FreeAligned(currentBlock);
		for (auto& block : usedBlocks) FreeAligned(block.second);
		for (auto& block : availableBlocks) FreeAligned(block.second);
	}
	void* Alloc(size_t nBytes) {
		// Round up _nBytes_ to minimum machine alignment
	#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
			// gcc bug: max_align_t wasn't in std:: until 4.9.0
		const int align = alignof(::max_align_t);
	#else
		const int align = alignof(std::max_align_t);
	#endif
		static_assert(IsPowerOf2(align), "Minimum alignment not a power of two");
		nBytes = (nBytes + align - 1) & ~(align - 1);
		if (currentBlockPos + nBytes > currentAllocSize) {
			// Add current block to _usedBlocks_ list
			if (currentBlock) {
				usedBlocks.push_back(
					std::make_pair(currentAllocSize, currentBlock));
				currentBlock = nullptr;
				currentAllocSize = 0;
			}

			// Get new block of memory for _MemoryArena_

			// Try to get memory block from _availableBlocks_
			for (auto iter = availableBlocks.begin();
				iter != availableBlocks.end(); ++iter) {
				if (iter->first >= nBytes) {
					currentAllocSize = iter->first;
					currentBlock = iter->second;
					availableBlocks.erase(iter);
					break;
				}
			}
			if (!currentBlock) {
				currentAllocSize = std::max(nBytes, blockSize);
				currentBlock = AllocAligned<uint8_t>(currentAllocSize);
			}
			currentBlockPos = 0;
		}
		void* ret = currentBlock + currentBlockPos;
		currentBlockPos += nBytes;
		return ret;
	}
	template <typename T>
	T* Alloc(size_t n = 1, bool runConstructor = true) {
		T* ret = (T*)Alloc(n * sizeof(T));
		if (runConstructor)
			for (size_t i = 0; i < n; ++i) new (&ret[i]) T();
		return ret;
	}
	void Reset() {
		currentBlockPos = 0;
		availableBlocks.splice(availableBlocks.begin(), usedBlocks);
	}
	size_t TotalAllocated() const {
		size_t total = currentAllocSize;
		for (const auto& alloc : usedBlocks) total += alloc.first;
		for (const auto& alloc : availableBlocks) total += alloc.first;
		return total;
	}

private:
	MemoryArena(const MemoryArena&) = delete;
	MemoryArena& operator=(const MemoryArena&) = delete;
	// MemoryArena Private Data
	const size_t blockSize;
	size_t currentBlockPos = 0, currentAllocSize = 0;
	uint8_t* currentBlock = nullptr;
	std::list<std::pair<size_t, uint8_t*>> usedBlocks, availableBlocks;
};


NAMESPACE_DPHPC_END