// The kernel catalog: one flat table, hand-written, in one place. Each kernel
// itself lives with its own kind under kernels/, but nothing self-registers —
// a kernel exists for the engine exactly when it is named here.

#include "fastfeaturetransform/kernel.hpp"

#include <ostream>

#include "fastfeaturetransform/kernels/kernels.hpp"

namespace fastfeaturetransform {

void Kernel::write_state(std::ostream& output) const { output << state().dump(); }

std::unique_ptr<Kernel> make_kernel(const std::string& id, const nlohmann::json& params) {
  try {
    if (id == "select") return make_select(params);
    if (id == "cast") return make_cast(params);
    if (id == "label_encode") return make_label_encode(params);
    if (id == "group_agg") return make_group_agg(params);
    if (id == "count_encode") return make_count_encode(params);
    if (id == "ordinal_encode") return make_ordinal_encode(params);
    if (id == "one_hot_encode") return make_one_hot_encode(params);
    if (id == "rank_hot_encode") return make_rank_hot_encode(params);
    if (id == "hash_encode") return make_hash_encode(params);
    if (id == "target_encode") return make_target_encode(params);
    if (id == "m_estimate_encode") return make_m_estimate_encode(params);
    if (id == "woe_encode") return make_woe_encode(params);
    if (id == "james_stein_encode") return make_james_stein_encode(params);
    if (id == "leave_one_out_encode") return make_leave_one_out_encode(params);
    if (id == "cat_boost_encode") return make_cat_boost_encode(params);
    if (id == "quantile_encode") return make_quantile_encode(params);
    if (id == "summary_encode") return make_summary_encode(params);
    if (id == "basen_encode") return make_basen_encode(params);
    if (id == "binary_encode") return make_binary_encode(params);
    if (id == "gray_encode") return make_gray_encode(params);
    if (id == "backward_difference_encode") return make_backward_difference_encode(params);
    if (id == "helmert_encode") return make_helmert_encode(params);
    if (id == "sum_encode") return make_sum_encode(params);
    if (id == "polynomial_encode") return make_polynomial_encode(params);
    if (id == "standard_scale") return make_standard_scale(params);
    if (id == "min_max_scale") return make_min_max_scale(params);
    if (id == "max_abs_scale") return make_max_abs_scale(params);
    if (id == "binarize") return make_binarize(params);
    if (id == "normalize") return make_normalize(params);
    if (id == "robust_scale") return make_robust_scale(params);
    if (id == "quantile_transform") return make_quantile_transform(params);
    if (id == "k_bins_discretize") return make_k_bins_discretize(params);
    if (id == "polynomial_features") return make_polynomial_features(params);
    if (id == "spline_transform") return make_spline_transform(params);
    if (id == "power_transform") return make_power_transform(params);
  } catch (const nlohmann::json::exception& e) {
    throw Error(id + ": invalid params: " + e.what());
  }
  throw Error("unknown kernel: " + id +
              " (this runtime knows: select, cast, label_encode, group_agg, count_encode, ordinal_encode, "
              "one_hot_encode, rank_hot_encode, hash_encode, target_encode, m_estimate_encode, woe_encode, "
              "james_stein_encode, leave_one_out_encode, cat_boost_encode, quantile_encode, summary_encode, "
              "basen_encode, binary_encode, gray_encode, backward_difference_encode, helmert_encode, "
              "sum_encode, polynomial_encode, standard_scale, min_max_scale, max_abs_scale, binarize, "
              "normalize, robust_scale, quantile_transform, k_bins_discretize, polynomial_features, "
              "spline_transform, power_transform)");
}

std::vector<std::string> list_kernels() {
  return {"select",
          "cast",
          "label_encode",
          "group_agg",
          "count_encode",
          "ordinal_encode",
          "one_hot_encode",
          "rank_hot_encode",
          "hash_encode",
          "target_encode",
          "m_estimate_encode",
          "woe_encode",
          "james_stein_encode",
          "leave_one_out_encode",
          "cat_boost_encode",
          "quantile_encode",
          "summary_encode",
          "basen_encode",
          "binary_encode",
          "gray_encode",
          "backward_difference_encode",
          "helmert_encode",
          "sum_encode",
          "polynomial_encode",
          "standard_scale",
          "min_max_scale",
          "max_abs_scale",
          "binarize",
          "normalize",
          "robust_scale",
          "quantile_transform",
          "k_bins_discretize",
          "polynomial_features",
          "spline_transform",
          "power_transform"};
}

}  // namespace fastfeaturetransform
