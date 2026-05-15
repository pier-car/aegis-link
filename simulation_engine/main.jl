################################################################################
#  AEGIS-LINK :: simulation_engine/main.jl
# ------------------------------------------------------------------------------
#  High-fidelity physical engine for a single airborne target (drone) immersed
#  in a turbulent atmosphere. The dynamics are formulated as a system of
#  Stochastic Differential Equations (SDEs):
#
#      dr  = v dt
#      dv  = a_cmd(t) dt + W dt                (W = wind-induced acceleration)
#      dW  = -theta * W dt + sigma_w dB(t)     (Ornstein-Uhlenbeck, "colored noise")
#
#  The Wiener increment dB drives a *colored* (band-limited) acceleration W with
#  correlation time tau = 1/theta — physically realistic for atmospheric gusts,
#  unlike pure white noise which would be unphysical (infinite power).
#
#  State vector layout (9D internally, 6D published):
#      x = [ r_x r_y r_z | v_x v_y v_z | W_x W_y W_z ]
#
#  Performance:
#      * `StaticArrays.SVector` => stack-allocated, branch-free, SIMD-friendly.
#      * In-place SDE problem with `Tsit5`-style adaptive solver from
#        DifferentialEquations.jl (`SOSRI` for additive-noise SDEs).
#      * Binary publication via ZeroMQ PUB socket — packet bytes mirror
#        `shared/messages.h::TrackPacket` exactly (128 bytes, no serialisation).
#
#  Run:   julia --project=. simulation_engine/main.jl
################################################################################

using Pkg
# Ensure deps are present (cheap if already instantiated). `Random` is a
# standard library, so we skip it here.
for pkg in ("ZMQ", "StaticArrays", "DifferentialEquations")
    if Base.find_package(pkg) === nothing
        Pkg.add(pkg)
    end
end

using ZMQ
using StaticArrays
using DifferentialEquations
using StochasticDiffEq: SOSRI
using Random
using Printf
using Dates

# ---------------------------------------------------------------------------
#  TrackPacket binary layout (mirrors shared/messages.h, 128 bytes total)
# ---------------------------------------------------------------------------
const SCHEMA_VERSION   = UInt16(1)
const PRODUCER_SIM     = UInt32(1)
const FLAG_NONE        = UInt16(0)
const PACKET_NBYTES    = 128

"""
    pack_track_packet!(buf, pkt_id, ts_ns, state6, covdiag6; flags=0)

Write a 128-byte TrackPacket into the preallocated `buf::Vector{UInt8}`.
The struct layout is:

    [0:4)    UInt32 packet_id
    [4:8)    UInt32 producer_id
    [8:16)   UInt64 timestamp_ns
    [16:64)  6 × Float64 state
    [64:112) 6 × Float64 cov_diag
    [112:114) UInt16 schema_version
    [114:116) UInt16 flags
    [116:128) 12 bytes padding (zeroed)
"""
function pack_track_packet!(buf::Vector{UInt8}, pkt_id::UInt32, ts_ns::UInt64,
                            state6::SVector{6,Float64}, covdiag6::SVector{6,Float64};
                            flags::UInt16 = FLAG_NONE)
    @assert length(buf) == PACKET_NBYTES
    fill!(buf, 0x00)
    GC.@preserve buf begin
        p = pointer(buf)
        unsafe_store!(Ptr{UInt32}(p + 0),  pkt_id)
        unsafe_store!(Ptr{UInt32}(p + 4),  PRODUCER_SIM)
        unsafe_store!(Ptr{UInt64}(p + 8),  ts_ns)
        for i in 1:6
            unsafe_store!(Ptr{Float64}(p + 16 + (i-1)*8), state6[i])
            unsafe_store!(Ptr{Float64}(p + 64 + (i-1)*8), covdiag6[i])
        end
        unsafe_store!(Ptr{UInt16}(p + 112), SCHEMA_VERSION)
        unsafe_store!(Ptr{UInt16}(p + 114), flags)
    end
    return buf
end

# ---------------------------------------------------------------------------
#  Physics: SDE drift & diffusion (in-place, allocation-free)
# ---------------------------------------------------------------------------
# Parameters bag (immutable, stack-friendly).
struct SimParams
    theta::Float64        # OU mean-reversion rate         [1/s]   (=> tau = 1/theta)
    sigma_w::Float64      # OU diffusion (gust intensity) [m/s^2 / sqrt(s)]
    g::Float64            # gravity                       [m/s^2]
    a_cmd::SVector{3,Float64}  # commanded acceleration   [m/s^2]
end

"Drift f(x,p,t): deterministic part of the SDE."
function drift!(du, u, p::SimParams, t)
    # u = [r(3), v(3), W(3)]
    @inbounds begin
        du[1] = u[4];  du[2] = u[5];  du[3] = u[6]                       # dr/dt = v
        du[4] = p.a_cmd[1] + u[7]                                         # dv/dt = a_cmd + W
        du[5] = p.a_cmd[2] + u[8]
        du[6] = p.a_cmd[3] + u[9] - p.g                                   # +gravity on z
        du[7] = -p.theta * u[7]                                           # OU mean-reversion
        du[8] = -p.theta * u[8]
        du[9] = -p.theta * u[9]
    end
    return nothing
end

"Diffusion g(x,p,t): noise enters ONLY the wind components (W_x,W_y,W_z)."
function diffusion!(du, u, p::SimParams, t)
    @inbounds begin
        du[1] = 0.0; du[2] = 0.0; du[3] = 0.0
        du[4] = 0.0; du[5] = 0.0; du[6] = 0.0
        du[7] = p.sigma_w
        du[8] = p.sigma_w
        du[9] = p.sigma_w
    end
    return nothing
end

# ---------------------------------------------------------------------------
#  Main simulation loop
# ---------------------------------------------------------------------------
function run_simulation(; bind_addr::String = "tcp://*:5555",
                          dt_publish::Float64  = 0.01,   # 100 Hz telemetry
                          duration_s::Float64  = 600.0,
                          rng_seed::Integer    = 0xA3615)

    Random.seed!(rng_seed)

    # --- ZeroMQ publisher (binary frames) ---------------------------------
    ctx  = Context()
    sock = Socket(ctx, PUB)
    # Drop oldest if subscriber is slow — we are real-time, never block.
    ZMQ.set_sndhwm(sock, 16)
    ZMQ.bind(sock, bind_addr)
    @info "AEGIS-LINK simulator publishing on $bind_addr"

    # --- Initial conditions: drone @ 1 km altitude, 50 m/s eastbound ------
    u0 = [ 0.0, 0.0, 1000.0,    # r_x r_y r_z
          50.0, 0.0,    0.0,    # v_x v_y v_z
           0.0, 0.0,    0.0 ]   # W_x W_y W_z

    params = SimParams(
        0.5,                          # theta  -> tau = 2 s (typical gust scale)
        2.5,                          # sigma_w in m/s^2/sqrt(s)
        9.80665,
        SVector(0.0, 0.0, 0.0)        # nominal: ballistic, no thrust
    )

    tspan   = (0.0, duration_s)
    prob    = SDEProblem(drift!, diffusion!, u0, tspan, params)
    integ   = init(prob, SOSRI(); dt = 1e-3, adaptive = true,
                   save_everystep = false, save_start = false)

    buf       = Vector{UInt8}(undef, PACKET_NBYTES)
    pkt_id    = UInt32(0)
    t_next    = dt_publish

    # Process-wide measurement-noise model (variance broadcast on the wire so
    # downstream consumers know the *truth-side* uncertainty of this packet).
    cov_truth = SVector(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)

    # Wall-clock pacing: we want to *emit* one packet every dt_publish seconds
    # of real time (soft real-time, 100 Hz default). Without this the loop
    # would run at maximum CPU speed and finish 10 minutes of simulated flight
    # in < 1 second of wall time, before any subscriber can connect.
    t_wall0 = time()

    try
        while integ.t < duration_s
            step!(integ, dt_publish, true)            # advance exactly dt_publish
            u = integ.u
            state6 = SVector(u[1], u[2], u[3], u[4], u[5], u[6])

            # CLOCK_TAI ≈ CLOCK_REALTIME + 37 s (leap seconds offset, see README).
            ts_ns = UInt64(round(time() * 1e9)) + UInt64(37) * UInt64(1_000_000_000)

            pkt_id += UInt32(1)
            pack_track_packet!(buf, pkt_id, ts_ns, state6, cov_truth)
            ZMQ.send(sock, buf)

            if pkt_id % 200 == 0
                @printf("[sim] t=%7.2fs  pos=(%8.2f %8.2f %8.2f)  |W|=%.3f\n",
                        integ.t, u[1], u[2], u[3], hypot(u[7], u[8], u[9]))
            end

            # Soft real-time pacing.
            t_target = t_wall0 + integ.t
            t_lag    = t_target - time()
            if t_lag > 0
                sleep(t_lag)
            end
        end
    catch e
        @warn "Simulation interrupted" exception=(e, catch_backtrace())
    finally
        close(sock); close(ctx)
        @info "Simulator stopped after $(pkt_id) packets."
    end
end

# Entry point
if abspath(PROGRAM_FILE) == @__FILE__
    run_simulation()
end
