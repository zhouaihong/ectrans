#define ALIGN(I, A) (((I)+(A)-1)/(A)*(A))
#if defined CUDAGPU
#define ACC_GET_HIP_STREAM ACC_GET_CUDA_STREAM
#define OPENACC_LIB OPENACC
#endif

! (C) Copyright 2000- ECMWF.
! (C) Copyright 2000- Meteo-France.
! (C) Copyright 2022- NVIDIA.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

MODULE LEINV_MOD
  USE PARKIND_ECTRANS,        ONLY: JPIM, JPRB, JPRBT, JPRD, JPIB
  USE BUFFERED_ALLOCATOR_MOD, ONLY: BUFFERED_ALLOCATOR
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: LEINV_STRIDES, LEINV, LEINV_GEMM_ANTISYM, LEINV_GEMM_SYM

  INTEGER(KIND=JPIM) :: A = 8 !Alignment

CONTAINS
  SUBROUTINE LEINV_STRIDES(KF_LEG,IOUT_STRIDES0,IOUT_SIZE,IIN_STRIDES0,IIN_SIZE,&
                           IOUT0_STRIDES0,IOUT0_SIZE,IIN0_STRIDES0,IIN0_SIZE)
    USE TPM_DIM,   ONLY: R
    USE TPM_DISTR, ONLY: D
#ifndef TRANS_SINGLE
    USE BUTTERFLY_GPU_DP_MOD, ONLY: BUTTERFLY_WORK_COLUMNS
#endif

    IMPLICIT NONE

    INTEGER(KIND=JPIM), INTENT(IN)  :: KF_LEG

    INTEGER(KIND=JPIM), OPTIONAL :: IOUT_STRIDES0
    INTEGER(KIND=JPIB), OPTIONAL :: IOUT_SIZE
    INTEGER(KIND=JPIM), OPTIONAL :: IIN_STRIDES0
    INTEGER(KIND=JPIB), OPTIONAL :: IIN_SIZE
    INTEGER(KIND=JPIM), OPTIONAL :: IOUT0_STRIDES0, IOUT0_SIZE
    INTEGER(KIND=JPIM), OPTIONAL :: IIN0_STRIDES0, IIN0_SIZE

    ASSOCIATE(D_OFFSETS_GEMM1=>D%OFFSETS_GEMM1, D_OFFSETS_GEMM2=>D%OFFSETS_GEMM2)


    IF (PRESENT(IOUT0_STRIDES0)) &
      IOUT0_STRIDES0 = ALIGN(KF_LEG,A)
    IF (PRESENT(IOUT0_SIZE)) &
      IOUT0_SIZE = IOUT0_STRIDES0 * ALIGN(R%NDGNH,A)
    IF (PRESENT(IIN_STRIDES0)) &
      IIN_STRIDES0 = ALIGN(2*KF_LEG,A)
    IF (PRESENT(IIN_SIZE)) THEN
      IIN_SIZE = IIN_STRIDES0*D_OFFSETS_GEMM2(D%NUMP+1)
#ifndef TRANS_SINGLE
      IIN_SIZE = IIN_SIZE+IIN_STRIDES0*BUTTERFLY_WORK_COLUMNS()
#endif
    ENDIF
    IF (PRESENT(IOUT_STRIDES0)) &
      IOUT_STRIDES0 = ALIGN(2*KF_LEG,A)
    IF (PRESENT(IOUT_SIZE)) THEN
      IOUT_SIZE = IOUT_STRIDES0*D_OFFSETS_GEMM1(D%NUMP+1)
#ifndef TRANS_SINGLE
      IOUT_SIZE = IOUT_SIZE+IOUT_STRIDES0*BUTTERFLY_WORK_COLUMNS()
#endif
    ENDIF
    IF (PRESENT(IIN0_STRIDES0)) &
      IIN0_STRIDES0 = ALIGN(KF_LEG,A)
    IF (PRESENT(IIN0_SIZE)) &
      IIN0_SIZE = IIN0_STRIDES0 * ALIGN(MAX((R%NTMAX+2)/2,(R%NTMAX+3)/2),A)

    END ASSOCIATE
  END SUBROUTINE LEINV_STRIDES

  SUBROUTINE LEINV(ALLOCATOR,PIA,ZINP,ZINP0,ZOUTS,ZOUTA,ZOUTS0,ZOUTA0,KF_LEG)
    !**** *LEINV* - Inverse Legendre transform.

    !     Purpose.
    !     --------
    !        Inverse Legendre tranform of all variables(kernel).

    !**   Interface.
    !     ----------
    !        CALL LEINV(...)

    !        Explicit arguments :  KM - zonal wavenumber (input-c)
    !        --------------------  KFC - number of fields to tranform (input-c)
    !                              PIA - spectral fields
    !                              for zonal wavenumber KM (input)

    !        Implicit arguments :  None.
    !        --------------------

    !     Method.
    !     -------

    !     Externals.
    !     ----------

    !     Reference.
    !     ----------
    !        ECMWF Research Department documentation of the IFS

    !     Author.
    !     -------
    !      Nils Wedi + Mats Hamrud + George Modzynski
    !
    !     Modifications.
    !     --------------
    !        J.Hague : Oct 2012 DR_HOOK round calls to DGEMM:
    !      F. Vana  05-Mar-2015  Support for single precision
    !     ------------------------------------------------------------------

    USE TPM_GEN,                     ONLY: LSYNC_TRANS, NOUT, NCUR_RESOL
    USE YOMHOOK,                     ONLY: LHOOK, DR_HOOK, JPHOOK
    USE TPM_DIM,                     ONLY: R
    USE TPM_GEOMETRY,                ONLY: G
    USE TPM_FIELDS_GPU,              ONLY: FG
    USE TPM_DISTR,                   ONLY: D
#ifndef TRANS_SINGLE
    USE TPM_FLT,                     ONLY: S
    USE BUTTERFLY_GPU_DP_MOD,        ONLY: APPLY_BUTTERFLY_INVERSE
#endif
    USE HICBLAS_MOD,                 ONLY: HIP_DGEMM_BATCHED, &
      &                                    HIP_DGEMM_GROUPED, HIP_SGEMM_GROUPED
    USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_INTPTR_T, C_LOC
    USE MPL_MODULE,                  ONLY: MPL_BARRIER,MPL_ALL_MS_COMM
    USE TPM_STATS,                   ONLY: GSTATS => GSTATS_NVTX
#ifdef ACCGPU
    USE OPENACC_LIB, ONLY: ACC_GET_HIP_STREAM
#endif
#ifdef TRANS_SINGLE
#define HIP_GEMM HIP_SGEMM_GROUPED
#else
#define HIP_GEMM HIP_DGEMM_GROUPED
#endif

    IMPLICIT NONE

    REAL(KIND=JPRB),    INTENT(IN)  :: PIA(:,:,:)
    INTEGER(KIND=JPIM), INTENT(IN)  :: KF_LEG
    REAL(KIND=JPRBT), POINTER, INTENT(OUT) :: ZINP(:), ZOUTS(:), ZOUTA(:)
    REAL(KIND=JPRD), POINTER, INTENT(OUT) :: ZINP0(:), ZOUTS0(:), ZOUTA0(:)
    TYPE(BUFFERED_ALLOCATOR), INTENT(IN) :: ALLOCATOR

    !     LOCAL
    INTEGER(KIND=JPIM)  :: KS(D%NUMP), NS(D%NUMP)
    INTEGER(KIND=JPIB)  :: AOFFSETS(D%NUMP), BOFFSETS(D%NUMP), COFFSETS(D%NUMP)
    INTEGER(KIND=JPIM)  :: KM, KMLOC, IA, IS, JK, J, IMLOC0(1)
    INTEGER(KIND=JPIM)  :: IOUT_STRIDES0
    INTEGER(KIND=JPIB)  :: IOUT_SIZE
    INTEGER(KIND=JPIM)  :: IIN_STRIDES0
    INTEGER(KIND=JPIB)  :: IIN_SIZE
    INTEGER(KIND=JPIM)  :: IOUT0_STRIDES0, IOUT0_SIZE
    INTEGER(KIND=JPIM)  :: IIN0_STRIDES0, IIN0_SIZE

    REAL(KIND=JPHOOK) :: ZHOOK_HANDLE

    INTEGER(C_INTPTR_T) :: HIP_STREAM

    ASSOCIATE(D_NUMP=>D%NUMP, R_NSMAX=>R%NSMAX, G_NDGLU=>G%NDGLU, D_MYMS=>D%MYMS, D_OFFSETS_GEMM1=>D%OFFSETS_GEMM1,&
        D_OFFSETS_GEMM2=>D%OFFSETS_GEMM2, &
        ZAA=>FG%ZAA, ZAS=>FG%ZAS, ZAA0=>FG%ZAA0, ZAS0=>FG%ZAS0)

    !*       1.1      PREPARATIONS.
    IF (LHOOK) CALL DR_HOOK('LE_DGEMM',0,ZHOOK_HANDLE)

#ifdef ACCGPU
    HIP_STREAM = INT(ACC_GET_HIP_STREAM(1_C_INT), C_INTPTR_T)
#endif
#ifdef OMPGPU
    HIP_STREAM = 0_C_INTPTR_T
#endif

    !     ------------------------------------------------------------------

    !*       1.       PERFORM LEGENDRE TRANFORM.
    !                 --------------------------

    !*       1.1      PREPARATIONS.

    CALL LEINV_STRIDES(KF_LEG,IOUT_STRIDES0,IOUT_SIZE,IIN_STRIDES0,IIN_SIZE,&
                       IOUT0_STRIDES0,IOUT0_SIZE,IIN0_STRIDES0,IIN0_SIZE)


#ifdef OMPGPU
    !$OMP TARGET DATA &
    !$OMP&              MAP(PRESENT,ALLOC:D,D_MYMS,D_NUMP) &
    !$OMP&              MAP(PRESENT,ALLOC:ZINP,ZOUTS,ZOUTA,ZINP0,ZOUTS0,ZOUTA0) &
    !$OMP&              MAP(PRESENT,ALLOC:ZAA,ZAS,PIA) &
    !$OMP&              MAP(PRESENT,ALLOC:R,R_NSMAX,D_OFFSETS_GEMM2)
#endif
#ifdef ACCGPU
    !$ACC DATA PRESENT(D,D_MYMS,D_NUMP) &
    !$ACC&     PRESENT(ZINP,ZOUTS,ZOUTA,ZINP0,ZOUTS0,ZOUTA0) &
    !$ACC&     PRESENT(ZAA,ZAS,PIA) &
    !$ACC&     PRESENT(R,R_NSMAX,D_OFFSETS_GEMM2)
#endif

    ! READ 2:NSMAX+3

    !IF KM=0 and NSMAX is 6:
    !    IA=1
    !    DO=1,6/2+1 ... 1..4
    !       PIA_2=1+1+(J-1)*2 ...2+(0..3)*2 .... 2,4,6,8
    !IF KM=0 and NSMAX is 7:
    !    IA=2
    !    DO=1,7/2+1 ... 1..4
    !       PIA_2=2+1+(1..4-1)*2 ...3+(0..3)*2 .... 3,5,7,9

    CALL GSTATS(494,0)
#ifdef OMPGPU
    ! Directive incomplete -> putting more variables in SHARED() triggers internal compiler error
    ! ftn-7991: INTERNAL COMPILER ERROR:  "Too few arguments on the stack"
    !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO COLLAPSE(2) &
    !$OMP& PRIVATE(KM,IA,J) &
    !$OMP& SHARED(D,R,KF_LEG,ZINP,IIN_STRIDES0,IIN0_STRIDES0) MAP(TO:KF_LEG)
#endif
#ifdef ACCGPU
    !$ACC PARALLEL LOOP COLLAPSE(2) PRIVATE(KM,IA,J) &
    !$ACC& FIRSTPRIVATE(KF_LEG,IIN_STRIDES0,IIN0_STRIDES0) DEFAULT(NONE) &
#ifdef _CRAYFTN
    !$ACC&
#else
    !$ACC& ASYNC(1)
#endif
#endif
    DO KMLOC=1,D_NUMP
      DO JK=1,2*KF_LEG
        KM =  D_MYMS(KMLOC)
        IA  = 1+MOD(R_NSMAX-KM+2,2)
        IF(KM /= 0)THEN
#ifdef ACCGPU
          !$ACC LOOP SEQ
#endif
          DO J=1,(R_NSMAX-KM+2)/2
            ZINP(JK+(J-1)*IIN_STRIDES0+D_OFFSETS_GEMM2(KMLOC)*IIN_STRIDES0)=PIA(JK,IA+1+(J-1)*2,KMLOC)
          ENDDO
          ! those are only needed with tensor cores (zinp might contain NaNs!)
#if defined(USE_CUTLASS) && defined(USE_CUTLASS_3XTF32)
          !$ACC LOOP SEQ
          DO J=(R_NSMAX-KM+2)/2+1,ALIGN((R_NSMAX-KM+2)/2,A)
            ZINP(JK+(J-1)*IIN_STRIDES0+D_OFFSETS_GEMM2(KMLOC)*IIN_STRIDES0)=0
          ENDDO
#endif
        ELSEIF (MOD((JK-1),2) == 0) THEN
          ! every other field is sufficient because Im(KM=0) == 0
#ifdef ACCGPU
          !$ACC LOOP SEQ
#endif
          DO J=1,(R_NSMAX+2)/2
            ZINP0((JK-1)/2+1+(J-1)*IIN0_STRIDES0) = PIA(JK,IA+1+(J-1)*2,KMLOC)
          ENDDO
          ! those are only needed with tensor cores (zinp might contain NaNs!)
#if defined(USE_CUTLASS) && defined(USE_CUTLASS_3XTF32)
          !$ACC LOOP SEQ
          DO J=(R_NSMAX+2)/2+1,ALIGN((R_NSMAX+2)/2,A)
            ZINP0((JK-1)/2+1+(J-1)*IIN0_STRIDES0) = 0
          ENDDO
#endif
        ENDIF
      ENDDO
    ENDDO


    IF (LSYNC_TRANS) THEN
#ifdef ACCGPU
      !$ACC WAIT(1)
#endif
      CALL GSTATS(494,1)
      CALL GSTATS(440,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(440,1)
    ELSE
      CALL GSTATS(494,1)
    ENDIF
    CALL GSTATS(424,0)
    CALL GSTATS(471,0)

    IMLOC0 = FINDLOC(D_MYMS,0)
    IF (IMLOC0(1) > 0) THEN
      ! compute m=0 in double precision
#ifdef OMPGPU
      !$OMP TARGET DATA USE_DEVICE_ADDR(ZAA0,ZINP0,ZOUTA0)
#endif
#ifdef ACCGPU
      !$ACC HOST_DATA USE_DEVICE(ZAA0,ZINP0,ZOUTA0)
#endif
      CALL HIP_DGEMM_BATCHED( &
        & 'N', 'T', &
        & KF_LEG, G_NDGLU(0), (R_NSMAX+2)/2, &
        & 1.0_JPRD, &
        & C_LOC(ZINP0), IIN0_STRIDES0, 0, &
        & C_LOC(ZAA0), SIZE(ZAA0,1), 0, &
        & 0.0_JPRD, &
        & C_LOC(ZOUTA0), IOUT0_STRIDES0, 0, &
        & 1, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
      !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
      !$OMP END TARGET DATA
#endif
   ENDIF

#ifndef TRANS_SINGLE
    CALL APPLY_BUTTERFLY_INVERSE(S%PLAN_A,ZINP,ZOUTA,IIN_STRIDES0,&
      & D_OFFSETS_GEMM2(D_NUMP+1),D_OFFSETS_GEMM1(D_NUMP+1),1,1110,ALLOCATOR)
#endif

    DO KMLOC=1,D_NUMP
      KM = D_MYMS(KMLOC)
      KS(KMLOC) = (R_NSMAX-KM+2)/2
      NS(KMLOC) = G_NDGLU(KM)
      AOFFSETS(KMLOC) = IIN_STRIDES0*D_OFFSETS_GEMM2(KMLOC)
      BOFFSETS(KMLOC) = D%OFFSETS_GEMM_MATRIX(KMLOC)
      COFFSETS(KMLOC) = IOUT_STRIDES0*D_OFFSETS_GEMM1(KMLOC)
#ifndef TRANS_SINGLE
      IF (S%PLAN_A%ENABLED .AND. S%PLAN_A%USE_HYBRID(KMLOC)) THEN
        NS(KMLOC)=0
        KS(KMLOC)=0
      ENDIF
#endif
    ENDDO
    IF(IMLOC0(1) > 0) THEN
      NS(IMLOC0(1)) = 0
      KS(IMLOC0(1)) = 0
    ENDIF
#ifdef OMPGPU
      !$OMP TARGET DATA USE_DEVICE_ADDR(ZAA,ZINP,ZOUTA)
#endif
#ifdef ACCGPU
      !$ACC HOST_DATA USE_DEVICE(ZAA,ZINP,ZOUTA)
#endif
    CALL HIP_GEMM( &
        & NCUR_RESOL, 11, & ! unique identifier
        & 'N', 'T', &
        & 2*KF_LEG, NS(:), KS(:), &
        & 1.0_JPRBT, &
        & C_LOC(ZINP), IIN_STRIDES0, AOFFSETS, &
        & C_LOC(ZAA), D%LEGENDRE_MATRIX_STRIDES, BOFFSETS, &
        & 0.0_JPRBT, &
        & C_LOC(ZOUTA), IOUT_STRIDES0, COFFSETS, &
        & D_NUMP, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
      !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
      !$OMP END TARGET DATA
#endif
    CALL GSTATS(471,1)

    IF (LSYNC_TRANS) THEN
#ifdef ACCGPU
      CALL GSTATS(478,0)
      !$ACC WAIT(1)
      CALL GSTATS(478,1)
#endif
      CALL GSTATS(444,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(444,1)
    ENDIF
    CALL GSTATS(424,1)

    ! 2. +++++++++++++ symmetric
    !IF KM=0 and NSMAX is 6:
    !    IS=2
    !    DO=1,4
    !       PIA_2=2+1+(0..3)*2 ... 3+(0..3)*2 ... 3,5,7,9
    !IF KM=0 and NSMAX is 7:
    !    IS=1
    !    DO=1,5
    !       PIA_2=1+1+(1..5-1)*2 ...2+(0..4)*2 .... 2,4,6,8,10

    CALL GSTATS(495,0)
#ifdef OMPGPU
    ! Directive incomplete -> putting more variables in SHARED() triggers internal compiler error
    ! ftn-7991: INTERNAL COMPILER ERROR:  "Too few arguments on the stack"
    !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO COLLAPSE(2) &
    !$OMP& PRIVATE(KM,IS,J) &
    !$OMP& SHARED(D,R,KF_LEG,ZINP,IIN_STRIDES0,IIN0_STRIDES0) MAP(TO:KF_LEG)
#endif
#ifdef ACCGPU
    !$ACC PARALLEL LOOP COLLAPSE(2) PRIVATE(KM,IS,J) &
    !$ACC& FIRSTPRIVATE(KF_LEG,IIN_STRIDES0,IIN0_STRIDES0) DEFAULT(NONE) &
#ifndef _CRAYFTN
    !$ACC& ASYNC(1)
#else
    !$ACC&
#endif
#endif
    DO KMLOC=1,D_NUMP
      DO JK=1,2*KF_LEG
        KM =  D_MYMS(KMLOC)
        IS  = 1+MOD(R_NSMAX-KM+1,2)
        IF(KM /= 0) THEN
#ifdef ACCGPU
          !$ACC LOOP SEQ
#endif
          DO J=1,(R_NSMAX-KM+3)/2
            ZINP(JK+(J-1)*IIN_STRIDES0+D_OFFSETS_GEMM2(KMLOC)*IIN_STRIDES0)=PIA(JK,IS+1+(J-1)*2,KMLOC)
          ENDDO
#if defined(USE_CUTLASS) && defined(USE_CUTLASS_3XTF32)
          ! those are only needed with tensor cores (zinp might contain NaNs!)
          !$ACC LOOP SEQ
          DO J=(R_NSMAX-KM+3)/2+1,ALIGN((R_NSMAX-KM+3)/2,A)
            ZINP(JK+(J-1)*IIN_STRIDES0+D_OFFSETS_GEMM2(KMLOC)*IIN_STRIDES0)=0
          ENDDO
#endif
        ELSEIF (MOD((JK-1),2) == 0) THEN
#ifdef ACCGPU
          !$ACC LOOP SEQ
#endif
          DO J=1,(R_NSMAX+3)/2
            ZINP0((JK-1)/2+1+(J-1)*IIN0_STRIDES0) = PIA(JK,IS+1+(J-1)*2,KMLOC)
          ENDDO
          ! those are only needed with tensor cores (zinp might contain NaNs!)
#if defined(USE_CUTLASS) && defined(USE_CUTLASS_3XTF32)
          !$ACC LOOP SEQ
          DO J=(R_NSMAX+3)/2+1,ALIGN((R_NSMAX+3)/2,A)
            ZINP0((JK-1)/2+1+(J-1)*IIN0_STRIDES0) = 0
          ENDDO
#endif
        ENDIF
      ENDDO
    ENDDO

    IF (LSYNC_TRANS) THEN
#ifdef ACCGPU
      !$ACC WAIT(1)
#endif
      CALL GSTATS(495,1)
      CALL GSTATS(440,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(440,1)
    ELSE
      CALL GSTATS(495,1)
    ENDIF
    CALL GSTATS(424,0)
    CALL GSTATS(473,0)

    IF (IMLOC0(1) > 0) THEN
#ifdef OMPGPU
      !$OMP TARGET DATA USE_DEVICE_ADDR(ZAS0,ZINP0,ZOUTS0)
#endif
#ifdef ACCGPU
      !$ACC HOST_DATA USE_DEVICE(ZAS0,ZINP0,ZOUTS0)
#endif
      CALL HIP_DGEMM_BATCHED( &
        & 'N', 'T', &
        & KF_LEG, G_NDGLU(0), (R_NSMAX+3)/2, &
        & 1.0_JPRD, &
        & C_LOC(ZINP0), IIN0_STRIDES0, 0, &
        & C_LOC(ZAS0), SIZE(ZAS0,1), 0, &
        & 0.0_JPRD, &
        & C_LOC(ZOUTS0), IOUT0_STRIDES0, 0, &
        & 1, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
      !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
      !$OMP END TARGET DATA
#endif
    ENDIF

#ifndef TRANS_SINGLE
    CALL APPLY_BUTTERFLY_INVERSE(S%PLAN_S,ZINP,ZOUTS,IIN_STRIDES0,&
      & D_OFFSETS_GEMM2(D_NUMP+1),D_OFFSETS_GEMM1(D_NUMP+1),1,1210,ALLOCATOR)
#endif

    DO KMLOC=1,D_NUMP
      KM = D_MYMS(KMLOC)
      KS(KMLOC) = (R_NSMAX-KM+3)/2
      NS(KMLOC) = G_NDGLU(KM)
      AOFFSETS(KMLOC) = IIN_STRIDES0*D_OFFSETS_GEMM2(KMLOC)
      BOFFSETS(KMLOC) = D%OFFSETS_GEMM_MATRIX(KMLOC)
      COFFSETS(KMLOC) = IOUT_STRIDES0*D_OFFSETS_GEMM1(KMLOC)
#ifndef TRANS_SINGLE
      IF (S%PLAN_S%ENABLED .AND. S%PLAN_S%USE_HYBRID(KMLOC)) THEN
        NS(KMLOC)=0
        KS(KMLOC)=0
      ENDIF
#endif
    ENDDO
    IF(IMLOC0(1) > 0) THEN
      NS(IMLOC0(1)) = 0
      KS(IMLOC0(1)) = 0
    ENDIF
#ifdef OMPGPU
    !$OMP TARGET DATA USE_DEVICE_ADDR(ZAS,ZINP,ZOUTS)
#endif
#ifdef ACCGPU
    !$ACC HOST_DATA USE_DEVICE(ZAS,ZINP,ZOUTS)
#endif
    CALL HIP_GEMM( &
      & NCUR_RESOL, 12, & ! unique identifier
      & 'N', 'T', &
      & 2*KF_LEG, NS(:), KS(:), &
      & 1.0_JPRBT, &
      & C_LOC(ZINP), IIN_STRIDES0, AOFFSETS, &
      & C_LOC(ZAS), D%LEGENDRE_MATRIX_STRIDES, BOFFSETS, &
      & 0.0_JPRBT, &
      & C_LOC(ZOUTS), IOUT_STRIDES0, COFFSETS, &
      & D_NUMP, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
    !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
    !$OMP END TARGET DATA
#endif
    CALL GSTATS(473,1)

    IF (LSYNC_TRANS) THEN
#ifdef ACCGPU
      CALL GSTATS(478,0)
      !$ACC WAIT(1)
      CALL GSTATS(478,1)
#endif
      CALL GSTATS(444,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(444,1)
    ENDIF
#ifdef ACCGPU
    CALL GSTATS(478,0)
    !$ACC WAIT(1)
    CALL GSTATS(478,1)
#endif
    CALL GSTATS(424,1)

#ifdef OMPGPU
    !$OMP END TARGET DATA
#endif
#ifdef ACCGPU
    !$ACC END DATA
#endif

    IF (LHOOK) CALL DR_HOOK('LE_DGEMM',1,ZHOOK_HANDLE)
    !     ------------------------------------------------------------------
    END ASSOCIATE
  END SUBROUTINE LEINV

  SUBROUTINE LEINV_GEMM_ANTISYM(ALLOCATOR,ZINP,ZINP0,ZOUTA,ZOUTA0,KF_LEG)
    USE TPM_GEN,                     ONLY: NCUR_RESOL
    USE TPM_DIM,                     ONLY: R
    USE TPM_GEOMETRY,                ONLY: G
    USE TPM_FIELDS_GPU,              ONLY: FG
    USE TPM_DISTR,                   ONLY: D
#ifndef TRANS_SINGLE
    USE TPM_FLT,                     ONLY: S
    USE BUTTERFLY_GPU_DP_MOD,        ONLY: APPLY_BUTTERFLY_INVERSE
#endif
    USE HICBLAS_MOD,                 ONLY: HIP_DGEMM_BATCHED, &
      &                                    HIP_DGEMM_GROUPED, HIP_SGEMM_GROUPED, &
      &                                    HIP_DGEMM_GROUPED_ASYNC, HIP_SGEMM_GROUPED_ASYNC
    USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_INTPTR_T, C_LOC
    USE TPM_STATS,                   ONLY: GSTATS => GSTATS_NVTX
#ifdef ACCGPU
    USE OPENACC_LIB, ONLY: ACC_GET_HIP_STREAM
#endif
#ifdef TRANS_SINGLE
#ifdef ACCGPU
#define HIP_GEMM_PACKED_RUN HIP_SGEMM_GROUPED_ASYNC
#else
#define HIP_GEMM_PACKED_RUN HIP_SGEMM_GROUPED
#endif
#else
#ifdef ACCGPU
#define HIP_GEMM_PACKED_RUN HIP_DGEMM_GROUPED_ASYNC
#else
#define HIP_GEMM_PACKED_RUN HIP_DGEMM_GROUPED
#endif
#endif

    IMPLICIT NONE

    TYPE(BUFFERED_ALLOCATOR), INTENT(IN) :: ALLOCATOR
    REAL(KIND=JPRBT), POINTER, INTENT(INOUT) :: ZINP(:), ZOUTA(:)
    REAL(KIND=JPRD),  POINTER, INTENT(INOUT) :: ZINP0(:), ZOUTA0(:)
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_LEG

    INTEGER(KIND=JPIM) :: KM, KMLOC, IMLOC0(1)
    INTEGER(KIND=JPIM) :: KS(D%NUMP), NS(D%NUMP)
    INTEGER(KIND=JPIB) :: AOFFSETS(D%NUMP), BOFFSETS(D%NUMP), COFFSETS(D%NUMP)
    INTEGER(KIND=JPIM) :: IOUT_STRIDES0
    INTEGER(KIND=JPIB) :: IOUT_SIZE
    INTEGER(KIND=JPIM) :: IIN_STRIDES0
    INTEGER(KIND=JPIB) :: IIN_SIZE
    INTEGER(KIND=JPIM) :: IOUT0_STRIDES0, IOUT0_SIZE
    INTEGER(KIND=JPIM) :: IIN0_STRIDES0, IIN0_SIZE
    INTEGER(C_INTPTR_T) :: HIP_STREAM

    ASSOCIATE(D_NUMP=>D%NUMP, R_NSMAX=>R%NSMAX, G_NDGLU=>G%NDGLU, &
        D_MYMS=>D%MYMS, D_OFFSETS_GEMM1=>D%OFFSETS_GEMM1, &
        D_OFFSETS_GEMM2=>D%OFFSETS_GEMM2, ZAA=>FG%ZAA, ZAA0=>FG%ZAA0)

#ifdef ACCGPU
    HIP_STREAM = INT(ACC_GET_HIP_STREAM(1_C_INT), C_INTPTR_T)
#endif
#ifdef OMPGPU
    HIP_STREAM = 0_C_INTPTR_T
#endif

    CALL LEINV_STRIDES(KF_LEG,IOUT_STRIDES0,IOUT_SIZE,IIN_STRIDES0,IIN_SIZE,&
                       IOUT0_STRIDES0,IOUT0_SIZE,IIN0_STRIDES0,IIN0_SIZE)

#ifdef OMPGPU
    !$OMP TARGET DATA &
    !$OMP&              MAP(PRESENT,ALLOC:D,D_MYMS,D_NUMP) &
    !$OMP&              MAP(PRESENT,ALLOC:ZINP,ZOUTA,ZINP0,ZOUTA0) &
    !$OMP&              MAP(PRESENT,ALLOC:ZAA) &
    !$OMP&              MAP(PRESENT,ALLOC:R,R_NSMAX,D_OFFSETS_GEMM2)
#endif
#ifdef ACCGPU
    !$ACC DATA PRESENT(D,D_MYMS,D_NUMP) &
    !$ACC&     PRESENT(ZINP,ZOUTA,ZINP0,ZOUTA0) &
    !$ACC&     PRESENT(ZAA) &
    !$ACC&     PRESENT(R,R_NSMAX,D_OFFSETS_GEMM2)
#endif

    CALL GSTATS(471,0)

    IMLOC0 = FINDLOC(D_MYMS,0)
    IF (IMLOC0(1) > 0) THEN
#ifdef OMPGPU
      !$OMP TARGET DATA USE_DEVICE_ADDR(ZAA0,ZINP0,ZOUTA0)
#endif
#ifdef ACCGPU
      !$ACC HOST_DATA USE_DEVICE(ZAA0,ZINP0,ZOUTA0)
#endif
      CALL HIP_DGEMM_BATCHED( &
        & 'N', 'T', &
        & KF_LEG, G_NDGLU(0), (R_NSMAX+2)/2, &
        & 1.0_JPRD, &
        & C_LOC(ZINP0), IIN0_STRIDES0, 0, &
        & C_LOC(ZAA0), SIZE(ZAA0,1), 0, &
        & 0.0_JPRD, &
        & C_LOC(ZOUTA0), IOUT0_STRIDES0, 0, &
        & 1, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
      !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
      !$OMP END TARGET DATA
#endif
    ENDIF

#ifndef TRANS_SINGLE
    CALL APPLY_BUTTERFLY_INVERSE(S%PLAN_A,ZINP,ZOUTA,IIN_STRIDES0,&
      & D_OFFSETS_GEMM2(D_NUMP+1),D_OFFSETS_GEMM1(D_NUMP+1),1,1310,ALLOCATOR)
#endif

    DO KMLOC=1,D_NUMP
      KM = D_MYMS(KMLOC)
      KS(KMLOC) = (R_NSMAX-KM+2)/2
      NS(KMLOC) = G_NDGLU(KM)
      AOFFSETS(KMLOC) = IIN_STRIDES0*D_OFFSETS_GEMM2(KMLOC)
      BOFFSETS(KMLOC) = D%OFFSETS_GEMM_MATRIX(KMLOC)
      COFFSETS(KMLOC) = IOUT_STRIDES0*D_OFFSETS_GEMM1(KMLOC)
#ifndef TRANS_SINGLE
      IF (S%PLAN_A%ENABLED .AND. S%PLAN_A%USE_HYBRID(KMLOC)) THEN
        NS(KMLOC)=0
        KS(KMLOC)=0
      ENDIF
#endif
    ENDDO
    IF(IMLOC0(1) > 0) THEN
      NS(IMLOC0(1)) = 0
      KS(IMLOC0(1)) = 0
    ENDIF
#ifdef OMPGPU
    !$OMP TARGET DATA USE_DEVICE_ADDR(ZAA,ZINP,ZOUTA)
#endif
#ifdef ACCGPU
    !$ACC HOST_DATA USE_DEVICE(ZAA,ZINP,ZOUTA)
#endif
    CALL HIP_GEMM_PACKED_RUN( &
      & NCUR_RESOL, 11, &
      & 'N', 'T', &
      & 2*KF_LEG, NS(:), KS(:), &
      & 1.0_JPRBT, &
      & C_LOC(ZINP), IIN_STRIDES0, AOFFSETS, &
      & C_LOC(ZAA), D%LEGENDRE_MATRIX_STRIDES, BOFFSETS, &
      & 0.0_JPRBT, &
      & C_LOC(ZOUTA), IOUT_STRIDES0, COFFSETS, &
      & D_NUMP, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
    !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
    !$OMP END TARGET DATA
#endif

    CALL GSTATS(471,1)

#ifdef OMPGPU
    !$OMP END TARGET DATA
#endif
#ifdef ACCGPU
    !$ACC END DATA
#endif

    END ASSOCIATE
  END SUBROUTINE LEINV_GEMM_ANTISYM

  SUBROUTINE LEINV_GEMM_SYM(ALLOCATOR,ZINP,ZINP0,ZOUTS,ZOUTS0,KF_LEG)
    USE TPM_GEN,                     ONLY: LSYNC_TRANS, NCUR_RESOL
    USE TPM_DIM,                     ONLY: R
    USE TPM_GEOMETRY,                ONLY: G
    USE TPM_FIELDS_GPU,              ONLY: FG
    USE TPM_DISTR,                   ONLY: D
#ifndef TRANS_SINGLE
    USE TPM_FLT,                     ONLY: S
    USE BUTTERFLY_GPU_DP_MOD,        ONLY: APPLY_BUTTERFLY_INVERSE
#endif
    USE HICBLAS_MOD,                 ONLY: HIP_DGEMM_BATCHED, &
      &                                    HIP_DGEMM_GROUPED, HIP_SGEMM_GROUPED, &
      &                                    HIP_DGEMM_GROUPED_ASYNC, HIP_SGEMM_GROUPED_ASYNC
    USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_INTPTR_T, C_LOC
    USE MPL_MODULE,                  ONLY: MPL_BARRIER,MPL_ALL_MS_COMM
    USE TPM_STATS,                   ONLY: GSTATS => GSTATS_NVTX
#ifdef ACCGPU
    USE OPENACC_LIB, ONLY: ACC_GET_HIP_STREAM
#endif

    IMPLICIT NONE

    TYPE(BUFFERED_ALLOCATOR), INTENT(IN) :: ALLOCATOR
    REAL(KIND=JPRBT), POINTER, INTENT(INOUT) :: ZINP(:), ZOUTS(:)
    REAL(KIND=JPRD),  POINTER, INTENT(INOUT) :: ZINP0(:), ZOUTS0(:)
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_LEG

    INTEGER(KIND=JPIM) :: KM, KMLOC, IMLOC0(1)
    INTEGER(KIND=JPIM) :: KS(D%NUMP), NS(D%NUMP)
    INTEGER(KIND=JPIB) :: AOFFSETS(D%NUMP), BOFFSETS(D%NUMP), COFFSETS(D%NUMP)
    INTEGER(KIND=JPIM) :: IOUT_STRIDES0
    INTEGER(KIND=JPIB) :: IOUT_SIZE
    INTEGER(KIND=JPIM) :: IIN_STRIDES0
    INTEGER(KIND=JPIB) :: IIN_SIZE
    INTEGER(KIND=JPIM) :: IOUT0_STRIDES0, IOUT0_SIZE
    INTEGER(KIND=JPIM) :: IIN0_STRIDES0, IIN0_SIZE
    INTEGER(C_INTPTR_T) :: HIP_STREAM

    ASSOCIATE(D_NUMP=>D%NUMP, R_NSMAX=>R%NSMAX, G_NDGLU=>G%NDGLU, &
        D_MYMS=>D%MYMS, D_OFFSETS_GEMM1=>D%OFFSETS_GEMM1, &
        D_OFFSETS_GEMM2=>D%OFFSETS_GEMM2, ZAS=>FG%ZAS, ZAS0=>FG%ZAS0)

#ifdef ACCGPU
    HIP_STREAM = INT(ACC_GET_HIP_STREAM(2_C_INT), C_INTPTR_T)
#endif
#ifdef OMPGPU
    HIP_STREAM = 0_C_INTPTR_T
#endif

    CALL LEINV_STRIDES(KF_LEG,IOUT_STRIDES0,IOUT_SIZE,IIN_STRIDES0,IIN_SIZE,&
                       IOUT0_STRIDES0,IOUT0_SIZE,IIN0_STRIDES0,IIN0_SIZE)

#ifdef OMPGPU
    !$OMP TARGET DATA &
    !$OMP&              MAP(PRESENT,ALLOC:D,D_MYMS,D_NUMP) &
    !$OMP&              MAP(PRESENT,ALLOC:ZINP,ZOUTS,ZINP0,ZOUTS0) &
    !$OMP&              MAP(PRESENT,ALLOC:ZAS) &
    !$OMP&              MAP(PRESENT,ALLOC:R,R_NSMAX,D_OFFSETS_GEMM2)
#endif
#ifdef ACCGPU
    !$ACC DATA PRESENT(D,D_MYMS,D_NUMP) &
    !$ACC&     PRESENT(ZINP,ZOUTS,ZINP0,ZOUTS0) &
    !$ACC&     PRESENT(ZAS) &
    !$ACC&     PRESENT(R,R_NSMAX,D_OFFSETS_GEMM2)
#endif

    CALL GSTATS(473,0)

    IMLOC0 = FINDLOC(D_MYMS,0)
    IF (IMLOC0(1) > 0) THEN
#ifdef OMPGPU
      !$OMP TARGET DATA USE_DEVICE_ADDR(ZAS0,ZINP0,ZOUTS0)
#endif
#ifdef ACCGPU
      !$ACC HOST_DATA USE_DEVICE(ZAS0,ZINP0,ZOUTS0)
#endif
      CALL HIP_DGEMM_BATCHED( &
        & 'N', 'T', &
        & KF_LEG, G_NDGLU(0), (R_NSMAX+3)/2, &
        & 1.0_JPRD, &
        & C_LOC(ZINP0), IIN0_STRIDES0, 0, &
        & C_LOC(ZAS0), SIZE(ZAS0,1), 0, &
        & 0.0_JPRD, &
        & C_LOC(ZOUTS0), IOUT0_STRIDES0, 0, &
        & 1, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
      !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
      !$OMP END TARGET DATA
#endif
    ENDIF

#ifndef TRANS_SINGLE
    CALL APPLY_BUTTERFLY_INVERSE(S%PLAN_S,ZINP,ZOUTS,IIN_STRIDES0,&
      & D_OFFSETS_GEMM2(D_NUMP+1),D_OFFSETS_GEMM1(D_NUMP+1),2,1410,ALLOCATOR)
#endif

    DO KMLOC=1,D_NUMP
      KM = D_MYMS(KMLOC)
      KS(KMLOC) = (R_NSMAX-KM+3)/2
      NS(KMLOC) = G_NDGLU(KM)
      AOFFSETS(KMLOC) = IIN_STRIDES0*D_OFFSETS_GEMM2(KMLOC)
      BOFFSETS(KMLOC) = D%OFFSETS_GEMM_MATRIX(KMLOC)
      COFFSETS(KMLOC) = IOUT_STRIDES0*D_OFFSETS_GEMM1(KMLOC)
#ifndef TRANS_SINGLE
      IF (S%PLAN_S%ENABLED .AND. S%PLAN_S%USE_HYBRID(KMLOC)) THEN
        NS(KMLOC)=0
        KS(KMLOC)=0
      ENDIF
#endif
    ENDDO
    IF(IMLOC0(1) > 0) THEN
      NS(IMLOC0(1)) = 0
      KS(IMLOC0(1)) = 0
    ENDIF
#ifdef OMPGPU
    !$OMP TARGET DATA USE_DEVICE_ADDR(ZAS,ZINP,ZOUTS)
#endif
#ifdef ACCGPU
    !$ACC HOST_DATA USE_DEVICE(ZAS,ZINP,ZOUTS)
#endif
    CALL HIP_GEMM_PACKED_RUN( &
      & NCUR_RESOL, 12, &
      & 'N', 'T', &
      & 2*KF_LEG, NS(:), KS(:), &
      & 1.0_JPRBT, &
      & C_LOC(ZINP), IIN_STRIDES0, AOFFSETS, &
      & C_LOC(ZAS), D%LEGENDRE_MATRIX_STRIDES, BOFFSETS, &
      & 0.0_JPRBT, &
      & C_LOC(ZOUTS), IOUT_STRIDES0, COFFSETS, &
      & D_NUMP, HIP_STREAM, C_LOC(ALLOCATOR%PTR))
#ifdef ACCGPU
    !$ACC END HOST_DATA
#endif
#ifdef OMPGPU
    !$OMP END TARGET DATA
#endif
    CALL GSTATS(473,1)

#ifdef ACCGPU
    CALL GSTATS(478,0)
    !$ACC WAIT(1,2)
    CALL GSTATS(478,1)
#endif
    IF (LSYNC_TRANS) THEN
      CALL GSTATS(444,0)
      CALL MPL_BARRIER(MPL_ALL_MS_COMM,CDSTRING='')
      CALL GSTATS(444,1)
    ENDIF
#ifdef OMPGPU
    !$OMP END TARGET DATA
#endif
#ifdef ACCGPU
    !$ACC END DATA
#endif

    END ASSOCIATE
  END SUBROUTINE LEINV_GEMM_SYM
END MODULE LEINV_MOD
