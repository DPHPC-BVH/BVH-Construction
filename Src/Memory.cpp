#include "Memory.h"
#include <malloc.h>


NAMESPACE_DPHPC_BEGIN
void* AllocAligned(size_t size) {
	
#if defined(OS_WIN)
	return _aligned_malloc(size, L1_CACHE_LINE_SIZE);
#else
	return memalign(L1_CACHE_LINE_SIZE, size);
#endif

}



void FreeAligned(void* ptr) {

	if (!ptr) return;
#if defined(OS_WIN)
	_aligned_free(ptr);
#else
	free(ptr);
#endif
	
}

NAMESPACE_DPHPC_END