! (C) Copyright 2000- ECMWF.
! (C) Copyright 2000- Meteo-France.
! (C) Copyright 2024- NVIDIA.
! 
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

MODULE TPM_FIELDS_GPU

USE EC_PARKIND, ONLY: JPIM, JPIB, JPRD, JPRBT

IMPLICIT NONE

SAVE

TYPE FIELDS_GPU_TYPE
! scratch arrays for ltinv and ltdir and associated dimension variables
REAL(KIND=JPRBT),ALLOCATABLE :: ZAA(:)  !! JPRL for 1/2
REAL(KIND=JPRBT),ALLOCATABLE :: ZAS(:)  !! JPRL for 1/2

! for m=0 in ledir_mod:
REAL(KIND=JPRD),ALLOCATABLE :: ZAA0(:,:)
REAL(KIND=JPRD),ALLOCATABLE :: ZAS0(:,:)
REAL(KIND=JPRBT),ALLOCATABLE :: ZEPSNM(:,:)
REAL(KIND=JPRBT),ALLOCATABLE :: ZREC_A1(:),ZREC_A2(:),ZREC_A3(:)
REAL(KIND=JPRBT),ALLOCATABLE :: ZREC_S1(:),ZREC_S2(:),ZREC_S3(:)
REAL(KIND=JPRBT),ALLOCATABLE :: ZREC_MU2(:)
INTEGER(KIND=JPIM),ALLOCATABLE :: IREC_A_SEED(:),IREC_S_SEED(:)
INTEGER(KIND=JPIM),ALLOCATABLE :: IREC_TILE_MLOC(:)
INTEGER(KIND=JPIB),ALLOCATABLE :: IREC_TILE_ROW(:)
END TYPE FIELDS_GPU_TYPE

TYPE(FIELDS_GPU_TYPE),ALLOCATABLE,TARGET :: FIELDS_GPU_RESOL(:)
TYPE(FIELDS_GPU_TYPE),POINTER     :: FG

END MODULE TPM_FIELDS_GPU
