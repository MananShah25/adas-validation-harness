#include "adas_core/metrics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

namespace {

TEST(TimeToCollision, BasicClosing) {
    EXPECT_DOUBLE_EQ(adas_core::time_to_collision(20.0, 10.0, 50.0), 5.0);
}

TEST(TimeToCollision, LeadFasterReturnsInfinity) {
    EXPECT_TRUE(std::isinf(adas_core::time_to_collision(20.0, 25.0, 50.0)));
}

TEST(TimeToCollision, EqualSpeedsReturnsInfinity) {
    EXPECT_TRUE(std::isinf(adas_core::time_to_collision(20.0, 20.0, 50.0)));
}

TEST(TimeToCollision, ZeroGapReturnsZero) {
    EXPECT_DOUBLE_EQ(adas_core::time_to_collision(20.0, 0.0, 0.0), 0.0);
}

TEST(TimeToCollision, NegativeGapReturnsZero) {
    EXPECT_DOUBLE_EQ(adas_core::time_to_collision(20.0, 0.0, -5.0), 0.0);
}

TEST(TimeToCollision, ShrinksAsClosingSpeedGrows) {
    const double slow = adas_core::time_to_collision(15.0, 10.0, 50.0);
    const double fast = adas_core::time_to_collision(30.0, 10.0, 50.0);
    EXPECT_LT(fast, slow);
}

TEST(LaneDeviation, PositiveOffset) {
    EXPECT_DOUBLE_EQ(adas_core::lane_deviation(0.25), 0.25);
}

TEST(LaneDeviation, NegativeOffsetIsAbsolute) {
    EXPECT_DOUBLE_EQ(adas_core::lane_deviation(-0.25), 0.25);
}

TEST(LaneDeviation, ZeroIsZero) {
    EXPECT_DOUBLE_EQ(adas_core::lane_deviation(0.0), 0.0);
}

TEST(FollowingDistanceError, FartherThanTargetIsPositive) {
    EXPECT_DOUBLE_EQ(adas_core::following_distance_error(30.0, 20.0), 10.0);
}

TEST(FollowingDistanceError, CloserThanTargetIsNegative) {
    EXPECT_DOUBLE_EQ(adas_core::following_distance_error(15.0, 20.0), -5.0);
}

TEST(BrakeResponseLatency, Basic) {
    EXPECT_NEAR(adas_core::brake_response_latency(10.0, 10.4), 0.4, 1e-12);
}

TEST(BrakeResponseLatency, ZeroLatencyAllowed) {
    EXPECT_DOUBLE_EQ(adas_core::brake_response_latency(10.0, 10.0), 0.0);
}

TEST(BrakeResponseLatency, NegativeThrows) {
    // Braking before the event is a caller ordering bug. The Python reference
    // raises ValueError; std::invalid_argument is what pybind11 maps to it.
    EXPECT_THROW(adas_core::brake_response_latency(10.0, 9.5), std::invalid_argument);
}

}  // namespace
