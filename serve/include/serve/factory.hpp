#pragma once
#include <algorithm>
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ssdf::serve {
template <class IProduct, typename IdentifierType, typename Creator>
class Factory {
 public:
  bool doRegister(const IdentifierType& id, Creator creator) {
    if (associations_.find(id) != associations_.end()) {
      return false;
    }
    associations_.emplace(id, creators_.size());
    creators_.emplace_back(creator);
    return true;
  }

  bool doRegister(std::initializer_list<IdentifierType> ids, Creator creator) {
    if (std::any_of(ids.begin(), ids.end(),
                    [&associations_ = associations_](const IdentifierType& id) {
                      return associations_.find(id) != associations_.end();
                    })) {
      return false;
    }
    size_t index = creators_.size();
    std::for_each(ids.begin(), ids.end(),
                  [&associations_ = associations_, &index](const IdentifierType& id) {
                    associations_.emplace(id, index);
                  });
    creators_.emplace_back(creator);
    return true;
  }

  std::unique_ptr<IProduct> createObject(const IdentifierType& id) { return nullptr; }

 private:
  std::vector<Creator> creators_;
  std::unordered_map<IdentifierType, int> associations_;
};
}  // namespace ssdf::serve
