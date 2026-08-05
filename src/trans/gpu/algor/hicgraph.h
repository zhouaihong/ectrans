// (C) Copyright 2000- ECMWF.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
// In applying this licence, ECMWF does not waive the privileges and immunities
// granted to it by virtue of its status as an intergovernmental organisation
// nor does it submit to any jurisdiction.

#ifndef __HICGRAPH_H__
#define __HICGRAPH_H__

#include <memory>
#include <type_traits>

struct hic_graph_deleter {
  void operator()(std::remove_pointer_t<hipGraph_t> *graph) const noexcept {
    if (graph != nullptr)
      HIC_CHECK(hipGraphDestroy(graph));
  }
};

using hic_graph_owner =
    std::unique_ptr<std::remove_pointer_t<hipGraph_t>, hic_graph_deleter>;

#endif
