/* =============================================================================
 *  AEGIS-LINK :: shared/messages.h
 * -----------------------------------------------------------------------------
 *  Wire-level data contract between the Julia physics engine, the C++ tracking
 *  core and the Python orchestrator.
 *
 *  Design constraints
 *  ------------------
 *   1.  C ABI compatible (no C++-only features) so that Julia (`Base.unsafe_load`
 *       on a `Ptr{TrackPacket}`) and Python (`ctypes.Structure` / `numpy.dtype`)
 *       can map the bytes 1:1 without any serialisation layer.
 *   2.  Explicit padding + `alignas(64)` => the whole struct occupies exactly
 *       128 bytes (2 cache lines on x86_64). No compiler-inserted padding,
 *       no endianness ambiguity (we mandate little-endian, native on x86_64
 *       and ARM64 used in WSL2 / desktop hosts).
 *   3.  All floating point fields are IEEE-754 binary64 (double precision) to
 *       preserve the sub-mm metrology heritage; integer fields are fixed-width.
 *   4.  Timestamp is an unsigned 64-bit integer expressing nanoseconds since
 *       the Unix epoch on `CLOCK_TAI` (see README §Clock Synchronisation).
 *
 *  Memory layout (offsets are bytes, computed and asserted at compile time):
 *
 *      offset  size   field
 *      ------  ----   ---------------------------------------------------------
 *        0      4     packet_id            (monotonic per-publisher counter)
 *        4      4     producer_id          (1=sim, 2=tracker, 3=orchestrator)
 *        8      8     timestamp_ns         (CLOCK_TAI, ns since Unix epoch)
 *       16     48     state[6]             (x,y,z, vx,vy,vz)            [m, m/s]
 *       64     48     cov_diag[6]          (sigma^2 of the 6 state comps)
 *      112      2     schema_version       (bump on breaking changes)
 *      114      2     flags                (bit 0 = MANEUVER, bit 1 = LOST_TRK)
 *      116     12     _padding             (to reach 128B / 2 cache lines)
 *      ------  ----
 *      total  128
 *
 * ============================================================================= */
#ifndef AEGIS_LINK_MESSAGES_H
#define AEGIS_LINK_MESSAGES_H

#include <stdint.h>

#if defined(__cplusplus)
  #include <cstddef>
  #define AEGIS_ALIGNAS(x) alignas(x)
  #define AEGIS_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
extern "C" {
#else
  #include <stddef.h>
  #include <assert.h>
  #if __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
    #define AEGIS_ALIGNAS(x) _Alignas(x)
    #define AEGIS_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
  #else
    #define AEGIS_ALIGNAS(x)
    #define AEGIS_STATIC_ASSERT(cond, msg)
  #endif
#endif

/* Producer identifiers (carried in every packet for routing/debug). */
enum AegisProducer {
    AEGIS_PRODUCER_SIMULATOR    = 1u,
    AEGIS_PRODUCER_TRACKER      = 2u,
    AEGIS_PRODUCER_ORCHESTRATOR = 3u
};

/* Bit flags. */
enum AegisFlags {
    AEGIS_FLAG_NONE      = 0x0000u,
    AEGIS_FLAG_MANEUVER  = 0x0001u,  /* anomaly score above gate              */
    AEGIS_FLAG_LOST_TRK  = 0x0002u,  /* tracker has lost lock                 */
    AEGIS_FLAG_TEST      = 0x8000u   /* synthetic / test packet (do not act)  */
};

#define AEGIS_STATE_DIM   6u   /* (x,y,z, vx,vy,vz)                           */
#define AEGIS_SCHEMA_V    1u

#pragma pack(push, 1)
typedef struct AEGIS_ALIGNAS(64) TrackPacket {
    uint32_t packet_id;             /*  0 .. 4   */
    uint32_t producer_id;           /*  4 .. 8   */
    uint64_t timestamp_ns;          /*  8 .. 16  CLOCK_TAI ns since epoch     */

    double   state[AEGIS_STATE_DIM];     /* 16 .. 64  position[3] | velocity[3]  */
    double   cov_diag[AEGIS_STATE_DIM];  /* 64 .. 112 diagonal of P (variances)  */

    uint16_t schema_version;        /* 112 .. 114 */
    uint16_t flags;                 /* 114 .. 116 */
    uint8_t  _padding[12];          /* 116 .. 128 (zeroed by producers)        */
} TrackPacket;
#pragma pack(pop)

/* --- Compile-time invariants --------------------------------------------- */
AEGIS_STATIC_ASSERT(sizeof(TrackPacket) == 128,
                    "TrackPacket must be exactly 128 bytes (2 cache lines)");

#if defined(__cplusplus)
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, packet_id)      ==   0, "off pkt_id");
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, producer_id)    ==   4, "off prod_id");
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, timestamp_ns)   ==   8, "off ts");
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, state)          ==  16, "off state");
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, cov_diag)       ==  64, "off cov");
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, schema_version) == 112, "off sv");
AEGIS_STATIC_ASSERT(offsetof(TrackPacket, flags)          == 114, "off flags");
#endif

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* AEGIS_LINK_MESSAGES_H */
