// -*- mode: c++; indent-tabs-mode: nil; tab-width:2  -*-
#ifndef _ug_mm_tsa_h
#define _ug_mm_tsa_h

// (c) 2007-2009 Ulrich Germann. All rights reserved.

#include <iostream>
#include <stdexcept>
#include <sstream>

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/dynamic_bitset.hpp>

#include "tpt_tightindex.h"
#include "tpt_tokenindex.h"
#include "tpt_pickler.h"
#include "ug_tsa_base.h"

namespace sapt
{
  namespace bio=boost::iostreams;

  template<typename TOKEN>
  class mmTSA : public TSA<TOKEN>
  {
  public:
    typedef typename TSA<TOKEN>::tree_iterator tree_iterator;
    friend class TSA_tree_iterator<TOKEN>;
  private:
    bio::mapped_file_source file;

  public: // temporarily for debugging

    filepos_type const* index; // random access to top-level sufa ranges

  private:

    virtual char const* index_jump(char const* a, char const* z, float ratio) const;

    virtual char const* index_jump_precise(char const* startRange, char const* stopRange, size_t idx) const;

    char const* getLowerBound(id_type t) const;
    char const* getUpperBound(id_type t) const;

  public:
    mmTSA();
    mmTSA(std::string fname, Ttrack<TOKEN> const* c);
    void open(std::string fname, typename boost::shared_ptr<Ttrack<TOKEN> const> c);

    count_type
    sntCnt(char const* p, char const * const q) const;

    count_type
    rawCnt(char const* p, char const * const q) const;

    void
    getCounts(char const* p, char const * const q,
              count_type& sids, count_type& raw) const;

    char const*
    readSid(char const* p, char const* q, id_type& sid) const;

    char const*
    readOffset(char const* p, char const* q, offset_type& offset) const;

    void sanityCheck() const;

  };

  // ======================================================================

  /**
   * jump to the point 1/ratio in an index
   */
  template<typename TOKEN>
  char const*
  mmTSA<TOKEN>::
  index_jump(char const* a, char const* z, float ratio) const
  {
    assert(ratio >= 0 && ratio < 1);
    int jump = (ratio*(z-a));
    char const* m = a+jump-(jump%(sizeof(tpt::id_type)+sizeof(tpt::offset_type)));  // ensure we are landing on exact location
    assert(m >= a && m < z);
    return m;
  }

  /** @return an index position idx between
   *  /startRange/ and /endRange/.
   */
  template<typename TOKEN>
  char const*
  mmTSA<TOKEN>::
  index_jump_precise(char const* startRange,
                     char const* stopRange,
                     size_t idx) const
  {
    char const* m = startRange + idx * (sizeof(id_type) + sizeof(offset_type));
    assert(m < stopRange);
    return m;
  }

  // ======================================================================

  template<typename TOKEN>
  mmTSA<TOKEN>::
  mmTSA()
  {
    this->startArray   = NULL;
    this->endArray     = NULL;
  };

  // ======================================================================

  template<typename TOKEN>
  mmTSA<TOKEN>::
  mmTSA(std::string fname, Ttrack<TOKEN> const* c)
  {
    open(fname,c);
  }

  // ======================================================================

  template<typename TOKEN>
  void
  mmTSA<TOKEN>::
  open(std::string fname, typename boost::shared_ptr<Ttrack<TOKEN> const> c)
  {
    if (access(fname.c_str(),F_OK))
      {
        std::ostringstream msg;
        msg << "mmTSA<>::open: File '" << fname << "' does not exist.";
        throw std::runtime_error(msg.str().c_str());
      }
    assert(c);
    this->corpus = c;
    file.open(fname);
    Moses::prime(file);
    char const* p = file.data();
    filepos_type idxOffset;
    uint64_t versionMagic;
    p = tpt::numread(p,versionMagic);
    if (versionMagic != tpt::INDEX_V2_MAGIC)
      {
        std::ostringstream msg;
        msg << "mmTSA<>::open: File '" << fname << "' does not contain a recent v2 index (magic is wrong). Please re-build with mtt-build.";
        throw std::runtime_error(msg.str().c_str());
      }
    p = tpt::numread(p,idxOffset);
    p = tpt::numread(p,this->indexSize);

    // cerr << fname << ": " << idxOffset << " " << this->indexSize << std::endl;

    this->startArray = p;
    this->index      = reinterpret_cast<filepos_type const*>(file.data()+idxOffset);
    this->endArray   = reinterpret_cast<char const*>(index);
    this->corpusSize = c->size();
    this->numTokens  = c->numTokens();
  }

  // ======================================================================

  template<typename TOKEN>
  char const*
  mmTSA<TOKEN>::
  getLowerBound(id_type id) const
  {
    if (id >= this->indexSize)
      return NULL;
    return this->startArray + this->index[id];
  }

  // ======================================================================

  template<typename TOKEN>
  char const*
  mmTSA<TOKEN>::
  getUpperBound(id_type id) const
  {
    if (id >= this->indexSize)
      return NULL;
    // if (index[id] == index[id+1])
    // return NULL;
    else
      return this->startArray + this->index[id+1];
  }

  // ======================================================================

  template<typename TOKEN>
  char const*
  mmTSA<TOKEN>::
  readSid(char const* p, char const* q, id_type& sid) const
  {
    return tpt::numread(p,sid);
  }

  // ======================================================================

/*
  template<typename TOKEN>
  char const*
  mmTSA<TOKEN>::
  readSid(char const* p, char const* q, ::uint64_t& sid) const
  {
    // TODO: XXX: is this used anywhere? Why? (typedef id_type is not good???)
    return tpt::numread(p,sid);
  }
*/

  // ======================================================================

  template<typename TOKEN>
  inline
  char const*
  mmTSA<TOKEN>::
  readOffset(char const* p, char const* q, offset_type& offset) const
  {
    return tpt::numread(p,offset);
  }

  // ======================================================================
/*
  template<typename TOKEN>
  inline
  char const*
  mmTSA<TOKEN>::
  readOffset(char const* p, char const* q, ::uint64_t& offset) const
  {
    // TODO: WHY??????
    return tpt::tightread(p,q,offset);
  }
*/
  // ======================================================================

  template<typename TOKEN>
  count_type
  mmTSA<TOKEN>::
  rawCnt(char const* p, char const* const q) const
  {
    size_t ret = (q - p) / (sizeof(id_type) + sizeof(offset_type));
    return (count_type) ret;
  }

  // ======================================================================

  template<typename TOKEN>
  void
  mmTSA<TOKEN>::
  getCounts(char const* p, char const* const q,
	    count_type& sids, count_type& raw) const
  {
    raw = 0;
    id_type sid; offset_type off;
    boost::dynamic_bitset<uint64_t> check(this->corpus->size());
    while (p < q)
      {
	p = tpt::numread(p,sid);
	p = tpt::numread(p,off);
	check.set(sid);
	raw++;
      }
    sids = check.count();
  }

  // ======================================================================

} // end of namespace ugdiss

// #include "ug_mm_tsa_extra.h"
#endif