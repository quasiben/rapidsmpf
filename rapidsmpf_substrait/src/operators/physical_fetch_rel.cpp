/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/operators/physical_fetch_rel.hpp"

#include <sstream>

// Undefine DEBUG to avoid conflict
#ifdef DEBUG
#undef DEBUG
#endif

#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>

#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf_substrait {

PhysicalFetchRel::PhysicalFetchRel(
    std::unique_ptr<PhysicalOperator> child,
    int64_t offset,
    int64_t count
) : PhysicalOperator(
        PhysicalOperatorType::FETCH,
        child->OutputTypes(),
        count > 0 ? static_cast<cudf::size_type>(count) : child->EstimatedCardinality()
    ),
    offset_(offset),
    count_(count) {
    children_.push_back(std::move(child));
}

rapidsmpf::streaming::Node PhysicalFetchRel::BuildNode(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    auto offset = offset_;
    auto count = count_;

    return [](
        std::shared_ptr<rapidsmpf::streaming::Context> ctx,
        std::shared_ptr<rapidsmpf::streaming::Channel> input,
        std::shared_ptr<rapidsmpf::streaming::Channel> output,
        int64_t offset,
        int64_t count
    ) -> rapidsmpf::streaming::Node {
        rapidsmpf::streaming::ShutdownAtExit shutdown_guard(output);
        std::uint64_t seq = 0;

        int64_t rows_skipped = 0;
        int64_t rows_returned = 0;

        while (true) {
            auto msg = co_await input->receive();
            if (msg.empty()) {
                break;
            }
            co_await ctx->executor()->schedule();

            // Check if we've already returned enough rows
            if (count >= 0 && rows_returned >= count) {
                // Consume remaining messages but don't process
                continue;
            }

            auto chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                msg.release<rapidsmpf::streaming::TableChunk>()
            );

            if (!chunk->is_available()) {
                auto [reservation, overbooking] = ctx->br()->reserve(
                    rapidsmpf::MemoryType::DEVICE,
                    chunk->make_available_cost(),
                    true
                );
                *chunk = chunk->make_available(reservation);
            }

            auto table_view = chunk->table_view();
            auto stream = chunk->stream();
            auto mr = ctx->br()->device_mr();
            int64_t chunk_rows = table_view.num_rows();

            // Handle offset
            int64_t start_idx = 0;
            if (rows_skipped < offset) {
                int64_t to_skip = offset - rows_skipped;
                if (to_skip >= chunk_rows) {
                    // Skip entire chunk
                    rows_skipped += chunk_rows;
                    continue;
                }
                start_idx = to_skip;
                rows_skipped = offset;
            }

            // Handle count
            int64_t end_idx = chunk_rows;
            if (count >= 0) {
                int64_t remaining = count - rows_returned;
                if (remaining <= 0) {
                    continue;
                }
                int64_t available = chunk_rows - start_idx;
                if (available > remaining) {
                    end_idx = start_idx + remaining;
                }
            }

            // Slice the table if needed
            std::unique_ptr<cudf::table> output_table;
            if (start_idx == 0 && end_idx == chunk_rows) {
                output_table = std::make_unique<cudf::table>(table_view, stream, mr);
            } else {
                std::vector<cudf::size_type> indices = {
                    static_cast<cudf::size_type>(start_idx),
                    static_cast<cudf::size_type>(end_idx)
                };
                auto sliced = cudf::slice(table_view, indices);
                if (!sliced.empty()) {
                    output_table = std::make_unique<cudf::table>(sliced[0], stream, mr);
                }
            }

            if (output_table && output_table->num_rows() > 0) {
                rows_returned += output_table->num_rows();
                
                auto new_chunk = std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(output_table),
                    stream
                );
                co_await output->send(rapidsmpf::streaming::to_message(seq++, std::move(new_chunk)));
            }
        }
        co_await output->drain(ctx->executor());
    }(ctx, ch_in, ch_out, offset, count);
}

std::string PhysicalFetchRel::ToString() const {
    std::ostringstream oss;
    oss << "FETCH(offset=" << offset_ << ", count=" << count_ << ")";
    return oss.str();
}

}  // namespace rapidsmpf_substrait
