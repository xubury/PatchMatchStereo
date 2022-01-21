#ifndef COST_COMPUTER_HPP
#define COST_COMPUTER_HPP

#include <cstdint>

class CostComputer {
   public:
    CostComputer() = default;
    virtual ~CostComputer() = default;
    virtual float Compute(int32_t x, int32_t y, float d) const = 0;

};

#endif  // !COST_COMPUTER_HPP
