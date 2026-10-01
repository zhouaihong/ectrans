module comparison_allocator
  use iso_c_binding
  use growing_allocator_mod
  implicit none
contains
  function comparison_allocator_new() result(ptr) bind(c)
    type(c_ptr) :: ptr
    type(growing_allocation_type), pointer :: alloc
    allocate(alloc)
    nullify(alloc%ptr)
    alloc%free_funcs_sz=0
    ptr=c_loc(alloc)
  end function

  subroutine comparison_allocator_delete(ptr) bind(c)
    type(c_ptr), value :: ptr
    type(growing_allocation_type), pointer :: alloc
    call c_f_pointer(ptr,alloc)
    call destroy_growing_allocator(alloc)
    deallocate(alloc)
  end subroutine
end module
