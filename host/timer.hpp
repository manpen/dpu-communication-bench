#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>

struct Timer {
  std::chrono::time_point<std::chrono::steady_clock> start;
  uint64_t scaler;
  std::string name;
  bool reported{false};

  Timer(std::string name, size_t scaler = 1)
      : start(std::chrono::steady_clock::now()), scaler(scaler), name(name) {}

  ~Timer() {
    if (reported) {
      return;
    }

    report();
  }

  void report() {
    auto elapsed = seconds_since_start() / scaler;
    const char *prefix = "";

    if (elapsed < 1) {
      elapsed *= 1e3;
      prefix = "m";
    }

    if (elapsed < 1) {
      elapsed *= 1e3;
      prefix = "u";
    }

    if (elapsed < 1) {
      elapsed *= 1e3;
      prefix = "n";
    }

    std::cout << "Timer[" << name << "]: " << elapsed << prefix
              << "s (scale=" << scaler << ")\n";
    reported = true;
  }

  void hide() { reported = true; }

  double seconds_since_start() {
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> diff = end - start;
    return diff.count();
  }
};

struct BandwidthTimer {
  Timer timer;
  size_t bytes;

  BandwidthTimer(std::string name, size_t bytes) : timer(name), bytes(bytes) {
    timer.hide();
  }

  ~BandwidthTimer() {
    double elapsed = timer.seconds_since_start();
    double bandwidth = bytes / elapsed;

    const char *prefix = "";

    if (bandwidth > 1e3) {
      prefix = "Ki";
      bandwidth /= 1024;
    }

    if (bandwidth > 1e3) {
      prefix = "Mi";
      bandwidth /= 1024;
    }

    if (bandwidth > 1e3) {
      prefix = "Gi";
      bandwidth /= 1024;
    }

    std::cout << "Bandwidth[" << timer.name << "]: " << bandwidth << prefix
              << "B/s\n";
  }
};