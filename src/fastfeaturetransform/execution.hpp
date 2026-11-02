// Shared bounded task dispatcher. With zero or one thread, every task runs
// on the caller, in order.
#pragma once

#include <cstddef>
#include <functional>

namespace fastfeaturetransform {

// With more threads, workers (the caller among them) race a task counter. A
// failing task stops the claims above it, while lower tasks still run; once
// every worker joins, the lowest failing task's exception reaches the caller,
// as a serial run would raise it.
void parallel_for(unsigned n_threads, std::size_t n_tasks, const std::function<void(unsigned, std::size_t)>& fn);

}  // namespace fastfeaturetransform
