#pragma once

/**
 * @file crystalLattice_stretched.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author David Fang (david.fang@ikmail.com)
 * @brief
 *
 */

#include "shammath/DiscontinuousIterator.hpp"

namespace shammath {
    class LatticeError : public std::exception {
        public:
        explicit LatticeError(const char *message) : msg_(message) {}

        explicit LatticeError(const std::string &message) : msg_(message) {}

        ~LatticeError() noexcept override = default;

        [[nodiscard]] const char *what() const noexcept override { return msg_.c_str(); }

        protected:
        std::string msg_;
    };
} // namespace shammath
