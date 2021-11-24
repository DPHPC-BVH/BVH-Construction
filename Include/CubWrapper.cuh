#pragma once

#include "DPHPC.h"


NAMESPACE_DPHPC_BEGIN

void DeviceSort(unsigned int numberOfElements, unsigned int** dKeysIn, unsigned int** dKeysOut,
                 unsigned int** dValuesIn, unsigned int** dValuesOut);

NAMESPACE_DPHPC_END