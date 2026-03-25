#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>

namespace op3 {

/**
 * A bounded blocking queue that keeps only the newest items when full.
 *
 * This queue is used for real-time style pipelines where stale work is less useful
 * than the most recent message.
 */
template <typename T>
class BlockingQueue {
 public:
  /**
   * Creates a queue with a fixed maximum number of retained items.
   *
   * @param capacity Maximum number of items to retain. When full, the oldest item is dropped.
   */
  explicit BlockingQueue(std::size_t capacity) : capacity_(capacity) {}

  /**
   * Pushes a new item into the queue.
   *
   * If the queue is already full, the oldest item is discarded so the newest one wins.
   */
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;
      }

      if (capacity_ > 0 && queue_.size() >= capacity_) {
        queue_.pop_front();
      }

      queue_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  /**
   * Pops the next item, blocking until data arrives or the queue is closed.
   *
   * @param item Output slot for the popped item.
   * @return True when an item was produced, false when the queue is closed and empty.
   */
  bool pop(T& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });

    if (queue_.empty()) {
      return false;
    }

    item = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  /**
   * Closes the queue and wakes all waiting consumers.
   */
  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    cv_.notify_all();
  }

  /**
   * Returns the current number of retained items.
   */
  std::size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::size_t capacity_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<T> queue_;
  bool closed_ = false;
};

}  // namespace op3
