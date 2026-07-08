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
 *      114      2     flags                (see AegisFlags enum)
 *      116     12     _padding             (to reach 128B / 2 cache lines)
 *      ------  ----
 *      total  128
 *
 *  -----------------------------------------------------------------------------
 *  Fire-control upgrade (schema_version == 1, byte-compatible):
 *
 *    A second 128-byte message type, `EngagementPacket`, is published by the
 *    Python `engagement_engine` process on `tcp://*:5557`. It mirrors
 *    `TrackPacket`'s layout (same offsets for id/timestamp/state/cov/flags)
 *    so that subscribers can sniff either stream with the same struct
 *    unpack format. The semantic mapping is:
 *
 *      state[0..3]   = interceptor position  [m]
 *      state[3..6]   = interceptor velocity  [m/s]
 *      cov_diag[0]   = time-to-go             [s]
 *      cov_diag[1]   = predicted miss-distance at intercept [m]
 *      cov_diag[2]   = current LOS range to estimated target [m]
 *      cov_diag[3]   = closing speed          [m/s]
 *      cov_diag[4]   = remaining fuel fraction in [0,1]
 *      cov_diag[5]   = commanded lateral acceleration magnitude [m/s^2]
 *
 *    The `flags` field carries the fire-control state-machine bits
 *    (LOCKED / ENGAGED / KILL / MISS) defined in `AegisFlags` below.
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
    AEGIS_PRODUCER_ORCHESTRATOR = 3u,
    AEGIS_PRODUCER_INTERCEPTOR  = 4u,  /* fire-control / PN guidance engine    */
    AEGIS_PRODUCER_IR_SENSOR    = 5u   /* passive IRST sensor (tcp://*:5558)   */
};

/* Bit flags. */
enum AegisFlags {
    AEGIS_FLAG_NONE      = 0x0000u,
    AEGIS_FLAG_MANEUVER  = 0x0001u,  /* anomaly score above gate              */
    AEGIS_FLAG_LOST_TRK  = 0x0002u,  /* tracker has lost lock                 */
    AEGIS_FLAG_LOCKED    = 0x0004u,  /* fire-control lock acquired on track   */
    AEGIS_FLAG_ENGAGED   = 0x0008u,  /* interceptor launched, PN guidance live*/
    AEGIS_FLAG_KILL      = 0x0010u,  /* miss-distance < lethal radius         */
    AEGIS_FLAG_MISS      = 0x0020u,  /* engagement ended without a kill       */
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

/* -------------------------------------------------------------------------
 *  EngagementPacket — fire-control / interceptor telemetry.
 *
 *  Identical 128-byte layout as TrackPacket (same offsets and types) so that
 *  every existing subscriber can decode it with the same struct format.
 *  The semantic meaning of `state[]` and `cov_diag[]` differs (see the
 *  header comment at the top of this file).
 *
 *  Producer:  AEGIS_PRODUCER_INTERCEPTOR (4)
 *  Endpoint:  tcp://*:5557 (PUB)
 * ------------------------------------------------------------------------- */
typedef TrackPacket EngagementPacket;

AEGIS_STATIC_ASSERT(sizeof(EngagementPacket) == 128,
                    "EngagementPacket must be exactly 128 bytes");

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* AEGIS_LINK_MESSAGES_H */
