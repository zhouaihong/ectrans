! (C) Copyright 2001- ECMWF.
! (C) Copyright 2001- Meteo-France.
! (C) Copyright 2022- NVIDIA.
! 
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation
! nor does it submit to any jurisdiction.
!

MODULE DIR_TRANS_CTL_MOD
CONTAINS
  SUBROUTINE DIR_TRANS_CTL(KF_UV_G,KF_SCALARS_G,KF_GP,KF_FS,KF_UV,KF_SCALARS,&
    & PSPVOR,PSPDIV,PSPSCALAR,KVSETUV,KVSETSC,PGP,&
    & PSPSC3A,PSPSC3B,PSPSC2,KVSETSC3A,KVSETSC3B,KVSETSC2,PGPUV,PGP3A,PGP3B,PGP2,LDUPDATE_DEVICE)

    !**** *DIR_TRANS_CTL* - Control routine for direct spectral transform.

    !     Purpose.
    !     --------
    !        Control routine for the direct spectral transform

    !**   Interface.
    !     ----------
    !     CALL DIR_TRANS_CTL(...)

    !     Explicit arguments :
    !     --------------------
    !     KF_UV_G      - global number of spectral u-v fields
    !     KF_SCALARS_G - global number of scalar spectral fields
    !     KF_GP        - total number of output gridpoint fields
    !     KF_FS        - total number of fields in fourier space
    !     KF_UV        - local number of spectral u-v fields
    !     KF_SCALARS   - local number of scalar spectral fields
    !     PSPVOR(:,:)  - spectral vorticity
    !     PSPDIV(:,:)  - spectral divergence
    !     PSPSCALAR(:,:) - spectral scalarvalued fields
    !     KVSETUV(:)  - indicating which 'b-set' in spectral space owns a
    !                   vor/div field. Equivalant to NBSETLEV in the IFS.
    !                   The length of KVSETUV should be the GLOBAL number
    !                   of u/v fields which is the dimension of u and v releated
    !                   fields in grid-point space.
    !     KVESETSC(:) - indicating which 'b-set' in spectral space owns a
    !                   scalar field. As for KVSETUV this argument is required
    !                   if the total number of processors is greater than
    !                   the number of processors used for distribution in
    !                   spectral wave space.
    !     PGP(:,:,:)  - gridpoint fields

    !                  The ordering of the output fields is as follows (all
    !                  parts are optional depending on the input switches):
    !
    !       u             : KF_UV_G fields
    !       v             : KF_UV_G fields
    !       scalar fields : KF_SCALARS_G fields

    !     Method.
    !     -------

    !     Externals.  SHUFFLE     - reshuffle fields for load balancing
    !     ----------  FIELD_SPLIT - split fields in NPROMATR packets
    !                 LTDIR_CTL   - control of Legendre transform
    !                 FTDIR_CTL   - control of Fourier transform

    !     Author.
    !     -------
    !        Mats Hamrud *ECMWF*

    !     Modifications.
    !     --------------
    !        Original : 01-01-03

    !     ------------------------------------------------------------------

    USE PARKIND_ECTRANS,        ONLY: JPRBT, JPRD, JPRB, JPIM
    USE TPM_GEN,                ONLY: NPROMATR, NOUT, NPRINTLEV
    USE TPM_DISTR,              ONLY: MYPROC
    USE TPM_TRANS,              ONLY: GROWING_ALLOCATION
    USE BUFFERED_ALLOCATOR_MOD, ONLY: BUFFERED_ALLOCATOR, MAKE_BUFFERED_ALLOCATOR, &
      &                               INSTANTIATE_ALLOCATOR
    USE FTDIR_MOD,              ONLY: FTDIR_HANDLE, PREPARE_FTDIR, FTDIR
    USE LTDIR_MOD,              ONLY: LTDIR_HANDLE, PREPARE_LTDIR, LTDIR
    USE TRGTOL_MOD,             ONLY: TRGTOL_HANDLE, PREPARE_TRGTOL, TRGTOL
    USE TRLTOM_MOD,             ONLY: TRLTOM_HANDLE, PREPARE_TRLTOM, TRLTOM
    USE TRLTOM_PACK_UNPACK,     ONLY: TRLTOM_PACK_HANDLE, TRLTOM_UNPACK_HANDLE, &
      &                               PREPARE_TRLTOM_PACK, PREPARE_TRLTOM_UNPACK, TRLTOM_PACK, &
      &                               TRLTOM_UNPACK
    USE ABORT_TRANS_MOD,        ONLY: ABORT_TRANS

    IMPLICIT NONE

    ! Declaration of arguments

    INTEGER(KIND=JPIM), INTENT(IN) :: KF_UV_G
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_SCALARS_G
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_GP
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_FS
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_UV
    INTEGER(KIND=JPIM), INTENT(IN) :: KF_SCALARS
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(OUT) :: PSPVOR(:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(OUT) :: PSPDIV(:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(OUT) :: PSPSCALAR(:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(OUT) :: PSPSC3A(:,:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(OUT) :: PSPSC3B(:,:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(OUT) :: PSPSC2(:,:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN)  :: KVSETUV(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN)  :: KVSETSC(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN)  :: KVSETSC3A(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN)  :: KVSETSC3B(:)
    INTEGER(KIND=JPIM) ,OPTIONAL, INTENT(IN)  :: KVSETSC2(:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(IN)  :: PGP(:,:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(IN)  :: PGPUV(:,:,:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(IN)  :: PGP3A(:,:,:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(IN)  :: PGP3B(:,:,:,:)
    REAL(KIND=JPRB)    ,OPTIONAL, INTENT(IN)  :: PGP2(:,:,:)
    LOGICAL             ,OPTIONAL, INTENT(IN)  :: LDUPDATE_DEVICE

    ! Local variables
    REAL(KIND=JPRBT), POINTER :: FOUBUF_IN(:), FOUBUF(:)
    REAL(KIND=JPRBT), POINTER :: PREEL_REAL(:), PREEL_COMPLEX(:)

    REAL(KIND=JPRBT), POINTER :: ZINPS(:), ZINPA(:)
    REAL(KIND=JPRD), POINTER :: ZINPS0(:), ZINPA0(:)
    CHARACTER(LEN=16) :: DEBUG_VALUE
    INTEGER :: DEBUG_STATUS
    INTEGER(KIND=8) :: DEBUG_T0, DEBUG_TRATE
    LOGICAL :: LDEBUG

    TYPE(BUFFERED_ALLOCATOR) :: ALLOCATOR
    TYPE(TRGTOL_HANDLE) :: HTRGTOL
    TYPE(FTDIR_HANDLE) :: HFTDIR
    TYPE(TRLTOM_PACK_HANDLE) :: HTRLTOM_PACK
    TYPE(TRLTOM_HANDLE) :: HTRLTOM
    TYPE(TRLTOM_UNPACK_HANDLE) :: HTRLTOM_UNPACK
    TYPE(LTDIR_HANDLE) :: HLTDIR
    ! 260420 wrqt begin
    ! 260420 wrqt comment仅在首个调用时输出，避免重复刷屏
    LOGICAL,SAVE :: LREPORTED = .FALSE.
    ! 260420 wrqt end
    INTEGER,SAVE :: DEBUG_CALL = 0

    IF (NPROMATR > 0) THEN
      CALL ABORT_TRANS("NPROMATR > 0 not supported for GPU")
    ENDIF

    CALL GET_ENVIRONMENT_VARIABLE('ECTRANS_GPU_WARMUP_DEBUG',DEBUG_VALUE,STATUS=DEBUG_STATUS)
    LDEBUG = DEBUG_STATUS == 0 .AND. LEN_TRIM(DEBUG_VALUE) > 0 .AND. TRIM(DEBUG_VALUE) /= '0'
    IF (LDEBUG) THEN
      DEBUG_CALL = DEBUG_CALL + 1
      CALL SYSTEM_CLOCK(DEBUG_T0,DEBUG_TRATE)
    ENDIF

    ! Prepare everything
    ALLOCATOR = MAKE_BUFFERED_ALLOCATOR()
    HTRGTOL = PREPARE_TRGTOL(ALLOCATOR,KF_GP,KF_FS)
    IF (KF_FS > 0) THEN
      HFTDIR = PREPARE_FTDIR(ALLOCATOR,KF_FS)
      HTRLTOM_PACK = PREPARE_TRLTOM_PACK(ALLOCATOR, KF_FS)
      HTRLTOM = PREPARE_TRLTOM(ALLOCATOR, KF_FS)
      HTRLTOM_UNPACK = PREPARE_TRLTOM_UNPACK(ALLOCATOR, KF_FS)
      HLTDIR = PREPARE_LTDIR(ALLOCATOR, KF_FS, KF_UV)
    ENDIF

    CALL INSTANTIATE_ALLOCATOR(ALLOCATOR, GROWING_ALLOCATION)
    CALL REPORT_WARMUP_STAGE('prepare')

    ! 260420 wrqt begin
    ! 260420 wrqt comment输出GPU直接变换控制层收到的关键尺寸，便于核对外层传参与控制层是否一致
    IF (NPRINTLEV > 0 .AND. MYPROC == 1 .AND. .NOT. LREPORTED) THEN
      WRITE(NOUT,'(A,6(I0,1X))') 'TRACE GPU DIR_TRANS_CTL: KF_UV_G KF_SCALARS_G KF_GP KF_FS KF_UV KF_SCALARS = ', &
        & KF_UV_G, KF_SCALARS_G, KF_GP, KF_FS, KF_UV, KF_SCALARS
      LREPORTED = .TRUE.
    ENDIF
    ! 260420 wrqt end

    ! from the PGP arrays to PREEL_REAL
    CALL GSTATS(158,0)
    CALL TRGTOL(ALLOCATOR,HTRGTOL,PREEL_REAL,KF_FS,KF_GP,KF_UV_G,KF_SCALARS_G,&
     & KVSETUV=KVSETUV,KVSETSC=KVSETSC,&
     & KVSETSC3A=KVSETSC3A,KVSETSC3B=KVSETSC3B,KVSETSC2=KVSETSC2,&
     & PGP=PGP,PGPUV=PGPUV,PGP3A=PGP3A,PGP3B=PGP3B,PGP2=PGP2,LDUPDATE_DEVICE=LDUPDATE_DEVICE)
    CALL REPORT_WARMUP_STAGE('g2l')
    CALL GSTATS(158,1)

    IF (KF_FS > 0) THEN

      ! fourier transform from PREEL_REAL to PREEL_COMPLEX (in-place!)
      CALL GSTATS(106,0)
      CALL FTDIR(ALLOCATOR,HFTDIR,PREEL_REAL,PREEL_COMPLEX,KF_FS)
      CALL REPORT_WARMUP_STAGE('ftdir')
      CALL GSTATS(106,1)

      CALL GSTATS(153,0)

      CALL TRLTOM_PACK(ALLOCATOR,HTRLTOM_PACK,PREEL_COMPLEX,FOUBUF_IN,KF_FS)
      CALL TRLTOM(ALLOCATOR,HTRLTOM,FOUBUF_IN,FOUBUF,KF_FS)
      CALL TRLTOM_UNPACK(ALLOCATOR,HTRLTOM_UNPACK,FOUBUF,ZINPS,ZINPA,ZINPS0,ZINPA0,KF_FS,KF_UV)
      CALL REPORT_WARMUP_STAGE('l2m')
      CALL GSTATS(153,1)

      CALL GSTATS(103,0)
      CALL LTDIR(ALLOCATOR,HLTDIR,ZINPS,ZINPA,ZINPS0,ZINPA0,KF_FS,KF_UV,KF_SCALARS, &
            & PSPVOR,PSPDIV,PSPSCALAR,&
            & PSPSC3A,PSPSC3B,PSPSC2)
      CALL REPORT_WARMUP_STAGE('ltdir')
      CALL GSTATS(103,1)

    ENDIF

  CONTAINS
    SUBROUTINE REPORT_WARMUP_STAGE(CDSTAGE)
      CHARACTER(LEN=*), INTENT(IN) :: CDSTAGE
      INTEGER(KIND=8) :: DEBUG_T1

      IF (.NOT. LDEBUG) RETURN
#ifdef ACCGPU
      !$ACC WAIT
#endif
#ifdef OMPGPU
      !$OMP TASKWAIT
#endif
      CALL SYSTEM_CLOCK(DEBUG_T1)
      WRITE(NOUT,'(A,A,A,I0,A,I0,A,F10.3)') &
        & 'EC_WARMUP_DEBUG event=transform_stage direction=dir stage=',TRIM(CDSTAGE), &
        & ' proc=',MYPROC,' call=',DEBUG_CALL,' ms=', &
        & 1000.0_JPRD*REAL(DEBUG_T1-DEBUG_T0,KIND=JPRD)/REAL(DEBUG_TRATE,KIND=JPRD)
      DEBUG_T0 = DEBUG_T1
    END SUBROUTINE REPORT_WARMUP_STAGE

  END SUBROUTINE DIR_TRANS_CTL
END MODULE DIR_TRANS_CTL_MOD
