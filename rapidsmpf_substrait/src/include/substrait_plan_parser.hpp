/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "substrait/plan.pb.h"

namespace rapidsmpf_substrait {

/**
 * @brief Parses Substrait plans from binary protobuf or JSON format.
 *
 * This class handles deserialization of Substrait plans and provides
 * access to extension functions and URIs.
 */
class SubstraitPlanParser {
  public:
    /**
     * @brief Parse a Substrait plan from binary protobuf format.
     *
     * @param data The binary protobuf data.
     * @return The parsed plan.
     * @throws std::runtime_error if parsing fails.
     */
    static substrait::Plan ParseFromBinary(std::string const& data);

    /**
     * @brief Parse a Substrait plan from JSON format.
     *
     * @param json The JSON string.
     * @return The parsed plan.
     * @throws std::runtime_error if parsing fails.
     */
    static substrait::Plan ParseFromJson(std::string const& json);

    /**
     * @brief Parse a Substrait plan from a file.
     *
     * The file format is detected automatically based on content.
     *
     * @param file_path Path to the plan file.
     * @return The parsed plan.
     * @throws std::runtime_error if parsing fails.
     */
    static substrait::Plan ParseFromFile(std::string const& file_path);

    /**
     * @brief Construct a parser for a specific plan.
     *
     * @param plan The Substrait plan to work with.
     */
    explicit SubstraitPlanParser(substrait::Plan plan);

    /**
     * @brief Get the underlying plan.
     */
    [[nodiscard]] substrait::Plan const& GetPlan() const { return plan_; }

    /**
     * @brief Look up a function name by its anchor ID.
     *
     * @param function_anchor The function anchor ID.
     * @return The function name.
     * @throws std::runtime_error if the function is not found.
     */
    [[nodiscard]] std::string FindFunction(uint32_t function_anchor) const;

    /**
     * @brief Get the function map (anchor -> name).
     */
    [[nodiscard]] std::unordered_map<uint32_t, std::string> const& GetFunctionMap() const {
        return functions_map_;
    }

    /**
     * @brief Get the root relation from the plan.
     *
     * @return Reference to the root relation.
     * @throws std::runtime_error if there are no relations.
     */
    [[nodiscard]] substrait::RelRoot const& GetRootRelation() const;

  private:
    substrait::Plan plan_;
    std::unordered_map<uint32_t, std::string> functions_map_;

    void BuildFunctionMap();
};

}  // namespace rapidsmpf_substrait

