// One factory per kernel, so the catalog in kernel.cpp stays a flat table of
// names while each kernel's implementation lives with its own kind.
#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "fastfeaturetransform/kernel.hpp"

namespace fastfeaturetransform {

// kernels/basic.cpp
std::unique_ptr<Kernel> make_select(const nlohmann::json& params);
std::unique_ptr<Kernel> make_cast(const nlohmann::json& params);
std::unique_ptr<Kernel> make_label_encode(const nlohmann::json& params);

// kernels/group_agg.cpp
std::unique_ptr<Kernel> make_group_agg(const nlohmann::json& params);

// kernels/count.cpp
std::unique_ptr<Kernel> make_count_encode(const nlohmann::json& params);

// kernels/hash.cpp
std::unique_ptr<Kernel> make_hash_encode(const nlohmann::json& params);

// kernels/basen.cpp
std::unique_ptr<Kernel> make_basen_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_binary_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_gray_encode(const nlohmann::json& params);

// kernels/ordinal.cpp
std::unique_ptr<Kernel> make_ordinal_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_one_hot_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_rank_hot_encode(const nlohmann::json& params);

// kernels/contrast.cpp
std::unique_ptr<Kernel> make_backward_difference_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_helmert_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_sum_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_polynomial_encode(const nlohmann::json& params);

// kernels/target.cpp
std::unique_ptr<Kernel> make_target_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_m_estimate_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_woe_encode(const nlohmann::json& params);

// kernels/shrinkage.cpp
std::unique_ptr<Kernel> make_james_stein_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_leave_one_out_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_cat_boost_encode(const nlohmann::json& params);

// kernels/quantile.cpp
std::unique_ptr<Kernel> make_quantile_encode(const nlohmann::json& params);
std::unique_ptr<Kernel> make_summary_encode(const nlohmann::json& params);

// kernels/scale.cpp
std::unique_ptr<Kernel> make_standard_scale(const nlohmann::json& params);
std::unique_ptr<Kernel> make_min_max_scale(const nlohmann::json& params);
std::unique_ptr<Kernel> make_max_abs_scale(const nlohmann::json& params);
std::unique_ptr<Kernel> make_binarize(const nlohmann::json& params);
std::unique_ptr<Kernel> make_normalize(const nlohmann::json& params);

// kernels/binning.cpp
std::unique_ptr<Kernel> make_robust_scale(const nlohmann::json& params);
std::unique_ptr<Kernel> make_quantile_transform(const nlohmann::json& params);
std::unique_ptr<Kernel> make_k_bins_discretize(const nlohmann::json& params);

// kernels/expand.cpp
std::unique_ptr<Kernel> make_polynomial_features(const nlohmann::json& params);
std::unique_ptr<Kernel> make_spline_transform(const nlohmann::json& params);
std::unique_ptr<Kernel> make_power_transform(const nlohmann::json& params);

}  // namespace fastfeaturetransform
