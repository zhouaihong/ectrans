#pragma once

#include <cstddef>

using growing_allocator_free_function = void (*)(void *, std::size_t);

extern "C" void growing_allocator_register_free_c(
    void *, growing_allocator_free_function);
