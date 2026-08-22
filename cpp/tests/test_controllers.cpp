#include "adas_core/controllers.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

// ---- ACC ------------------------------------------------------------

TEST(AccController, ZeroErrorYieldsZeroAccel) {
    // gap exactly at desired_gap and speeds matched -> no correction needed
    const double ego_speed = 20.0;
    const double desired_gap = 2.0 + 1.5 * ego_speed;
    EXPECT_NEAR(adas_core::acc_controller(ego_speed, ego_speed, desired_gap), 0.0, 1e-12);
}

TEST(AccController, GapTooSmallDecelerates) {
    EXPECT_LT(adas_core::acc_controller(20.0, 20.0, 5.0), 0.0);
}

TEST(AccController, GapTooLargeAccelerates) {
    EXPECT_GT(adas_core::acc_controller(20.0, 20.0, 200.0), 0.0);
}

TEST(AccController, ClipsToMaxAccel) {
    EXPECT_DOUBLE_EQ(adas_core::acc_controller(10.0, 10.0, 1000.0), 2.0);
}

TEST(AccController, ClipsToMinAccel) {
    EXPECT_DOUBLE_EQ(adas_core::acc_controller(20.0, 0.0, 0.0), -4.0);
}

TEST(AccController, RespectsCustomLimits) {
    const double accel = adas_core::acc_controller(10.0, 10.0, 1000.0, 2.0, 1.5, -1.0, 0.5);
    EXPECT_DOUBLE_EQ(accel, 0.5);
}

// ---- LKA ------------------------------------------------------------

TEST(LkaController, ZeroErrorYieldsZeroSteering) {
    EXPECT_DOUBLE_EQ(adas_core::lka_controller(0.0, 0.0), 0.0);
}

TEST(LkaController, PositiveOffsetSteersNegative) {
    EXPECT_LT(adas_core::lka_controller(0.5, 0.0), 0.0);
}

TEST(LkaController, NegativeOffsetSteersPositive) {
    EXPECT_GT(adas_core::lka_controller(-0.5, 0.0), 0.0);
}

TEST(LkaController, ClipsToMaxSteer) {
    EXPECT_DOUBLE_EQ(adas_core::lka_controller(10.0, 10.0, 0.4), -0.4);
}

TEST(LkaController, ClipsToMinSteer) {
    EXPECT_DOUBLE_EQ(adas_core::lka_controller(-10.0, -10.0, 0.4), 0.4);
}

// ---- AEB ------------------------------------------------------------

TEST(AebController, AboveWarningIsNormal) {
    EXPECT_EQ(adas_core::aeb_controller(5.0), adas_core::AebState::Normal);
}

TEST(AebController, BetweenThresholdsIsWarning) {
    EXPECT_EQ(adas_core::aeb_controller(1.9, 2.0, 1.8), adas_core::AebState::Warning);
}

TEST(AebController, BelowBrakeIsFullBrake) {
    EXPECT_EQ(adas_core::aeb_controller(1.0, 2.0, 1.8), adas_core::AebState::FullBrake);
}

TEST(AebController, WarningBoundaryIsExclusive) {
    // ttc < warning_ttc is the trigger; exactly-equal stays Normal.
    EXPECT_EQ(adas_core::aeb_controller(2.0, 2.0, 1.8), adas_core::AebState::Normal);
    EXPECT_EQ(adas_core::aeb_controller(1.999, 2.0, 1.8), adas_core::AebState::Warning);
}

TEST(AebController, BrakeBoundaryIsExclusive) {
    EXPECT_EQ(adas_core::aeb_controller(1.8, 2.0, 1.8), adas_core::AebState::Warning);
    EXPECT_EQ(adas_core::aeb_controller(1.799, 2.0, 1.8), adas_core::AebState::FullBrake);
}

TEST(AebController, InfiniteTtcIsNormal) {
    EXPECT_EQ(adas_core::aeb_controller(std::numeric_limits<double>::infinity()),
              adas_core::AebState::Normal);
}

TEST(AebState, StringsMatchPythonReference) {
    EXPECT_STREQ(adas_core::to_string(adas_core::AebState::Normal), "normal");
    EXPECT_STREQ(adas_core::to_string(adas_core::AebState::Warning), "warning");
    EXPECT_STREQ(adas_core::to_string(adas_core::AebState::FullBrake), "full_brake");
}

}  // namespace
