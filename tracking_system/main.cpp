// =============================================================================
//  AEGIS-LINK :: tracking_system/main.cpp
// -----------------------------------------------------------------------------
//  Real-time Extended Kalman Filter (EKF) with a Constant-Acceleration (CA)
//  motion model in 3D. Internal state is 9-dimensional:
//
//        x = [ p_x p_y p_z   v_x v_y v_z   a_x a_y a_z ]^T
//
//  The continuous-time model is:
//
//        d/dt p = v
//        d/dt v = a
//        d/dt a = w(t)              w ~ N(0, q_a)   (white jerk)
//
//  Exact discretisation (Van Loan / closed-form for CA):
//
//        F(dt) = | I   dt I   0.5 dt^2 I |
//                | 0    I        dt   I  |
//                | 0    0         I      |
//
//        Q(dt) = q_a * | dt^5/20 I   dt^4/8 I   dt^3/6 I |
//                      | dt^4/8  I   dt^3/3 I   dt^2/2 I |
//                      | dt^3/6  I   dt^2/2 I    dt   I  |
//
//  We compute F and Q analytically (NOT via Taylor expansion) — this is the
//  "symbolic / accurate" discretisation the spec asks for, and is numerically
//  stable for any dt > 0.
//
//  Observation: the simulator publishes p AND v (6D). We treat them as a
//  noisy measurement (sensor model H = [I_6 | 0_{6x3}]). The filter then
//  publishes its own TrackPacket on tcp://*:5556 with the estimated 6D state
//  and the diagonal of the 6x6 marginal covariance P[0:6,0:6].
//
//  Build:    see CMakeLists.txt   (requires Eigen3, cppzmq, libzmq)
// =============================================================================

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <thread>

#include <Eigen/Dense>
#include <zmq.hpp>

#include "../shared/messages.h"

namespace {

constexpr int    NX            = 9;     // state dimension (CA, 3D)
constexpr int    NZ            = 6;     // measurement dimension (pos + vel)
constexpr double Q_JERK_PSD    = 4.0;   // (m/s^3)^2 / Hz — process-noise PSD on jerk
constexpr double R_POS_STD     = 0.05;  // m   — sub-pixel-class metrology heritage
constexpr double R_VEL_STD     = 0.20;  // m/s
constexpr double GATE_CHI2_99  = 16.81; // chi^2_{6, 0.99} — measurement gate

using Mat   = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using MatNX = Eigen::Matrix<double, NX, NX>;
using MatNZ = Eigen::Matrix<double, NZ, NZ>;
using MatHX = Eigen::Matrix<double, NZ, NX>;
using VecNX = Eigen::Matrix<double, NX, 1>;
using VecNZ = Eigen::Matrix<double, NZ, 1>;

std::atomic<bool> g_running{true};
void on_sigint(int) { g_running.store(false, std::memory_order_relaxed); }

/// Closed-form CA transition matrix F(dt).
MatNX make_F(double dt)
{
    MatNX F = MatNX::Identity();
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    F.block<3,3>(0,3) = dt * I3;
    F.block<3,3>(0,6) = 0.5 * dt * dt * I3;
    F.block<3,3>(3,6) = dt * I3;
    return F;
}

/// Closed-form CA process-noise covariance Q(dt) for white-jerk model.
MatNX make_Q(double dt, double q)
{
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    const double dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt3*dt, dt5 = dt4*dt;
    MatNX Q = MatNX::Zero();
    Q.block<3,3>(0,0) = (dt5 / 20.0) * I3;
    Q.block<3,3>(0,3) = (dt4 /  8.0) * I3;
    Q.block<3,3>(0,6) = (dt3 /  6.0) * I3;
    Q.block<3,3>(3,0) = Q.block<3,3>(0,3);
    Q.block<3,3>(3,3) = (dt3 /  3.0) * I3;
    Q.block<3,3>(3,6) = (dt2 /  2.0) * I3;
    Q.block<3,3>(6,0) = Q.block<3,3>(0,6);
    Q.block<3,3>(6,3) = Q.block<3,3>(3,6);
    Q.block<3,3>(6,6) =  dt          * I3;
    return q * Q;
}

class EKF_CA {
public:
    EKF_CA()
    {
        x_.setZero();
        P_ = MatNX::Identity() * 1e3;        // wide initial covariance
        H_.setZero();
        H_.block<6,6>(0,0) = Eigen::Matrix<double,6,6>::Identity();
        R_.setZero();
        R_.block<3,3>(0,0) = (R_POS_STD*R_POS_STD) * Eigen::Matrix3d::Identity();
        R_.block<3,3>(3,3) = (R_VEL_STD*R_VEL_STD) * Eigen::Matrix3d::Identity();
    }

    void predict(double dt)
    {
        const MatNX F = make_F(dt);
        const MatNX Q = make_Q(dt, Q_JERK_PSD);
        x_ = F * x_;
        P_ = F * P_ * F.transpose() + Q;
        // enforce symmetry (cancel rounding asymmetries)
        P_ = 0.5 * (P_ + P_.transpose());
    }

    /// Returns Mahalanobis^2 of the innovation (used for gating + diagnostics).
    double update(const VecNZ& z)
    {
        const VecNZ y  = z - H_ * x_;                    // innovation
        const MatNZ S  = H_ * P_ * H_.transpose() + R_;   // innovation covariance
        // Solve K = P H^T S^{-1} via LDLT (fast, SPD-stable).
        const auto   Sllt = S.llt();
        const Eigen::Matrix<double, NX, NZ> K = Sllt.solve(H_ * P_).transpose();

        const double d2 = y.transpose() * Sllt.solve(y);  // Mahalanobis^2

        if (d2 > GATE_CHI2_99) {
            // Gate-reject: keep prediction, do not corrupt the filter.
            return d2;
        }
        x_ += K * y;
        // Joseph form for numerical stability of P.
        const MatNX I = MatNX::Identity();
        const MatNX IKH = I - K * H_;
        P_ = IKH * P_ * IKH.transpose() + K * R_ * K.transpose();
        P_ = 0.5 * (P_ + P_.transpose());
        return d2;
    }

    const VecNX& x() const noexcept { return x_; }
    const MatNX& P() const noexcept { return P_; }
    bool         initialised() const noexcept { return initialised_; }
    void         set_initialised() noexcept   { initialised_ = true; }

    /// Bootstrap from first measurement (pos + vel).
    void seed(const VecNZ& z)
    {
        x_.setZero();
        x_.head<6>() = z;
        P_ = MatNX::Identity() * 1.0;
        P_.block<3,3>(0,0) = (R_POS_STD*R_POS_STD) * Eigen::Matrix3d::Identity();
        P_.block<3,3>(3,3) = (R_VEL_STD*R_VEL_STD) * Eigen::Matrix3d::Identity();
        P_.block<3,3>(6,6) = 25.0                  * Eigen::Matrix3d::Identity();
        initialised_ = true;
    }

private:
    VecNX x_;
    MatNX P_;
    MatHX H_;
    MatNZ R_;
    bool  initialised_ = false;
};

/// CLOCK_TAI in nanoseconds since the Unix epoch (best effort on Linux).
uint64_t now_tai_ns()
{
    timespec ts{};
#ifdef CLOCK_TAI
    if (::clock_gettime(CLOCK_TAI, &ts) == 0) {
        return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + ts.tv_nsec;
    }
#endif
    ::clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + ts.tv_nsec
           + 37ull * 1'000'000'000ull;   // leap-second offset fallback
}

void publish_estimate(zmq::socket_t& pub, const EKF_CA& ekf,
                      uint32_t pkt_id, uint16_t flags)
{
    TrackPacket out{};
    out.packet_id      = pkt_id;
    out.producer_id    = AEGIS_PRODUCER_TRACKER;
    out.timestamp_ns   = now_tai_ns();
    out.schema_version = AEGIS_SCHEMA_V;
    out.flags          = flags;

    for (int i = 0; i < 6; ++i) {
        out.state[i]    = ekf.x()(i);
        out.cov_diag[i] = ekf.P()(i, i);
    }
    zmq::message_t msg(sizeof(TrackPacket));
    std::memcpy(msg.data(), &out, sizeof(TrackPacket));
    (void) pub.send(msg, zmq::send_flags::dontwait);
}

} // namespace

int main(int argc, char** argv)
{
    std::signal(SIGINT,  on_sigint);
    std::signal(SIGTERM, on_sigint);

    const std::string sub_addr = (argc > 1) ? argv[1] : "tcp://127.0.0.1:5555";
    const std::string pub_addr = (argc > 2) ? argv[2] : "tcp://*:5556";

    zmq::context_t ctx(1);
    zmq::socket_t  sub(ctx, zmq::socket_type::sub);
    sub.set(zmq::sockopt::subscribe, "");
    sub.set(zmq::sockopt::rcvhwm,    16);
    sub.set(zmq::sockopt::conflate,  0);    // keep order, but small HWM => fresh
    sub.connect(sub_addr);

    zmq::socket_t  pub(ctx, zmq::socket_type::pub);
    pub.set(zmq::sockopt::sndhwm, 16);
    pub.bind(pub_addr);

    std::cerr << "[trk] subscribed to " << sub_addr
              << "  publishing on "     << pub_addr << '\n';

    // Synthetic measurement noise (the link is lossless; we simulate the sensor).
    std::mt19937_64 rng{0xC0FFEEull};
    std::normal_distribution<double> n_pos(0.0, R_POS_STD);
    std::normal_distribution<double> n_vel(0.0, R_VEL_STD);

    EKF_CA   ekf;
    uint64_t last_ts_ns = 0;
    uint32_t out_seq    = 0;

    while (g_running.load(std::memory_order_relaxed)) {
        zmq::message_t msg;
        const auto rc = sub.recv(msg, zmq::recv_flags::none);
        if (!rc) continue;
        if (msg.size() != sizeof(TrackPacket)) {
            std::cerr << "[trk] WARN: bad packet size " << msg.size() << '\n';
            continue;
        }

        TrackPacket in{};
        std::memcpy(&in, msg.data(), sizeof(TrackPacket));
        if (in.schema_version != AEGIS_SCHEMA_V) {
            std::cerr << "[trk] WARN: schema mismatch v" << in.schema_version << '\n';
            continue;
        }

        // Build noisy measurement (3D pos + 3D vel) from the truth packet.
        VecNZ z;
        for (int i = 0; i < 3; ++i) z(i)     = in.state[i]     + n_pos(rng);
        for (int i = 0; i < 3; ++i) z(3 + i) = in.state[3 + i] + n_vel(rng);

        if (!ekf.initialised()) {
            ekf.seed(z);
            last_ts_ns = in.timestamp_ns;
            publish_estimate(pub, ekf, ++out_seq, AEGIS_FLAG_NONE);
            continue;
        }

        // dt from producer-side timestamps (TAI, monotonic across the link).
        double dt = (in.timestamp_ns >= last_ts_ns)
                  ? double(in.timestamp_ns - last_ts_ns) * 1e-9
                  : 0.0;
        last_ts_ns = in.timestamp_ns;
        if (dt <= 0.0 || dt > 1.0) dt = 0.01;   // clamp pathological gaps

        ekf.predict(dt);
        const double d2 = ekf.update(z);

        const uint16_t flags = (d2 > GATE_CHI2_99) ? AEGIS_FLAG_MANEUVER
                                                   : AEGIS_FLAG_NONE;
        publish_estimate(pub, ekf, ++out_seq, flags);

        if ((out_seq % 200u) == 0u) {
            std::cerr << "[trk] seq=" << out_seq
                      << "  d^2="     << d2
                      << "  pos=("    << ekf.x()(0) << ", "
                                      << ekf.x()(1) << ", "
                                      << ekf.x()(2) << ")\n";
        }
    }
    std::cerr << "[trk] shutting down (seq=" << out_seq << ")\n";
    return 0;
}
