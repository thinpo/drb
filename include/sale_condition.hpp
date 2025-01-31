#pragma once

#include <array>
#include <string>
#include <unordered_set>
#include <functional>

namespace drb::market {

// Sale Condition is a 3-character array where each position can have specific values
using SaleCondition = std::array<char, 3>;

namespace sale_condition {

// UTP (Tape C) Sale Conditions
static const std::unordered_set<char> UTP_VALUES = {
    '@',  // Regular Sale
    'A',  // Acquisition
    'B',  // Bunched Trade
    'C',  // Cash Trade
    'D',  // Distribution
    'E',  // Automatic Execution
    'F',  // Intermarket Sweep
    'G',  // Bunched Sold Trade
    'H',  // Price Variation Trade
    'I',  // Odd Lot Trade
    'K',  // Rule 155 Trade
    'L',  // Sold Last
    'M',  // Market Center Close Price
    'N',  // Next Day Trade
    'O',  // Opening Prints
    'P',  // Prior Reference Price
    'Q',  // Market Center Open Price
    'R',  // Seller
    'S',  // Split Trade
    'T',  // Form T
    'U',  // Extended Hours
    'V',  // Contingent Trade
    'W',  // Average Price Trade
    'X',  // Cross Trade
    'Y',  // Yellow Flag
    'Z',  // Sold Out of Sequence
    ' '   // Empty/Space
};

// CTA (Tape A/B) Sale Conditions
static const std::unordered_set<char> CTA_VALUES = {
    '@',  // Regular Sale
    'A',  // Acquisition
    'B',  // Bunched Trade
    'C',  // Cash Trade
    'D',  // Distribution
    'E',  // Automatic Execution
    'F',  // Intermarket Sweep Order
    'G',  // Bunched Sold Trade
    'H',  // Price Variation Trade
    'I',  // Odd Lot Trade
    'K',  // Rule 127/155 Trade
    'L',  // Sold Last
    'M',  // Market Center Official Close
    'N',  // Next Day Trade
    'O',  // Market Center Opening Trade
    'P',  // Prior Reference Price
    'Q',  // Market Center Official Open
    'R',  // Seller
    'S',  // Split Trade
    'T',  // Form T
    'U',  // Extended Hours Trade
    'V',  // Contingent Trade
    'W',  // Average Price Trade
    'X',  // Cross Trade
    'Y',  // Yellow Flag Regular Trade
    'Z',  // Sold (Out of Sequence)
    ' '   // Empty/Space
};

// Validate if a sale condition character is valid for UTP
inline bool is_valid_utp(char c) {
    return UTP_VALUES.find(c) != UTP_VALUES.end();
}

// Validate if a sale condition character is valid for CTA
inline bool is_valid_cta(char c) {
    return CTA_VALUES.find(c) != CTA_VALUES.end();
}

// Validate entire sale condition array for UTP
inline bool is_valid_utp_condition(const SaleCondition& condition) {
    return is_valid_utp(condition[0]) && 
           is_valid_utp(condition[1]) && 
           is_valid_utp(condition[2]);
}

// Validate entire sale condition array for CTA
inline bool is_valid_cta_condition(const SaleCondition& condition) {
    return is_valid_cta(condition[0]) && 
           is_valid_cta(condition[1]) && 
           is_valid_cta(condition[2]);
}

// Convert string to sale condition
inline SaleCondition from_string(const std::string& str) {
    SaleCondition condition;
    for (size_t i = 0; i < 3; ++i) {
        condition[i] = (i < str.length()) ? str[i] : ' ';
    }
    return condition;
}

// Convert sale condition to string
inline std::string to_string(const SaleCondition& condition) {
    return std::string(condition.begin(), condition.end());
}

} // namespace sale_condition
} // namespace drb::market

// Hash function for SaleCondition
namespace std {
    template<>
    struct hash<drb::market::SaleCondition> {
        size_t operator()(const drb::market::SaleCondition& condition) const {
            // Combine the three characters into a single hash value
            return (static_cast<size_t>(condition[0]) << 16) |
                   (static_cast<size_t>(condition[1]) << 8) |
                   static_cast<size_t>(condition[2]);
        }
    };

    // Equality operator for SaleCondition
    template<>
    struct equal_to<drb::market::SaleCondition> {
        bool operator()(const drb::market::SaleCondition& lhs, 
                       const drb::market::SaleCondition& rhs) const {
            return lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] == rhs[2];
        }
    };
} 