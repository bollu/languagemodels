#include <iostream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <map>
#include <list>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

using namespace std;
using Dim = int;
using DimSize = int;

int last_word_index = 0;
set<string> vocab;
map<string, int> word2ix;
map<int, string> ix2word;

// end of sentence is denoted by -1
int num_sentences = 0;
static const long MAX_SENTENCES = 1e5;
static const long MAX_SENTENCE_LEN = 1e2;
int ss[MAX_SENTENCES][MAX_SENTENCE_LEN];
vector<vector<int> > sentences;


struct Shape {
    static const int MAXDIM = 10;
    int ndim = -42;
    DimSize vals[Shape::MAXDIM] = {-42};;

    int nelem() {
        int n = 1;
        for(int i = 0; i < ndim; ++i) n *= vals[i];
        return n;
    }

    int operator [](int dim) const {
        assert(dim >= 0);
        assert(dim < ndim);
        return vals[dim];
    }

    bool operator == (const Shape &other) const {
        if (ndim != other.ndim) return false;
        for(int i = 0; i < ndim; ++i) {
            if(vals[i] != other.vals[i]) return false;
        }
        return true;
    }

    bool operator != (const Shape &other) const {
        return !(*this == other);
    }

    // lex comparison
    bool operator < (const Shape &other) const {
        if (ndim < other.ndim) return true;
        if (ndim > other.ndim) return false;

        assert(ndim == other.ndim);

        for(int i = 0; i < ndim; ++i) {
            if (vals[i] < other.vals[i]) return true;
            if (vals[i] > other.vals[i]) return false;
            assert(vals[i] == other.vals[i]);
        }

        assert(*this == other);
        return false;

    }

    static Shape zerod() {
        Shape sh;
        sh.ndim = 0;
        return sh;
    }

    static Shape oned(int n) {
        Shape sh;
        sh.ndim = 1;
        sh.vals[0] = n;
        return sh;
    }

    static Shape twod(int n, int m) {
        Shape sh;
        sh.ndim = 2;
        sh.vals[0] = n;
        sh.vals[1] = m;
        return sh;
    }

    string to_str() const {
        string s =  "sh[";
        for(int i = 0; i < ndim; ++i) {
            s += to_string(vals[i]) + (i < ndim - 1 ? " " : "");
        }
        s += "]";
        return s;
    }

    // remove an outermost (leftmost) dimension
    Shape removeOutermost() const {
        Shape sh;
        assert(ndim > 0);
        sh.ndim = ndim - 1;
        for(int i = 0; i < ndim - 1; ++i) {
            sh.vals[i] = vals[i+1];
        }
        return sh;
    }

    // append an outermost dimension
    Shape addOutermost(int size) const {
        Shape sh = *this;
        sh.ndim++;
        for(int i = 1; i < ndim+1; ++i) {
            sh.vals[i] = vals[i-1];
        }
        sh.vals[0] = size;
        return sh;
    }
};


enum class ExprType {
    Add,
    Mul,
    Undef,
    Arr,
    Slice,
    Contract,
    Delta,
    ConstantInt,
    ConstantFloat,
    Tanh,
};

struct Index {
    string name;
    Index(string name) : name(name) {};

    bool operator == (const Index &other) const { return name == other.name; }
    bool operator != (const Index &other) const { return name != other.name; }
    bool operator < (const Index &other) const { return name < other.name; }

    string to_str() const { return name; };
};

struct Slice;

struct Expr {
    // gives type of expression
    ExprType ty = ExprType::Undef;
    Expr(ExprType ty): ty(ty) {};

    Expr *contract(Index ix);

    // print as string
    virtual string to_str() const = 0;

    // return free indeces
    virtual set<Index> free() const = 0;

    // substitute old indeces for new indeces.
    virtual Expr *subst(Index old, Index new_) const = 0;

    // creates dirac deltas by taking gradients
    virtual Expr *grad(string name, vector<Index> ixs) = 0;

    string detailed_to_str() const {
        string s = "";
        s += "[[";
        for(Index i : free()) s += i.name + ",";
        s += "]] ";
        s += to_str();
        return s;
    }

    // simplify the expression to normal form
    Expr *normalize();

};


struct ConstantInt : public Expr {
    int i;
    ConstantInt(int i) : Expr(ExprType::ConstantInt), i(i) {};

    string to_str() const { return std::to_string(i); }
    set<Index> free() const { return {}; }
    Expr *grad(string name, vector<Index> ixs) { 
        return new ConstantInt(0);
    }

    Expr *subst(Index old, Index new_) const {
        return new ConstantInt(i);
    }

};

struct ConstantFloat : public Expr {
    float f;
    ConstantFloat(float f) : Expr(ExprType::ConstantFloat), f(f) {};

    string to_str() const { return std::to_string(f); }
    set<Index> free() const { return {}; }
    Expr *grad(string name, vector<Index> ixs) { 
        return new ConstantInt(0);
    }

    Expr *subst(Index old, Index new_) const {
        return new ConstantFloat(f);
    }

};


struct Delta : public Expr {
    Index old;
    Index new_;

    Delta(Index old, Index new_) : 
        Expr(ExprType::Delta), 
    old(old), new_(new_) { };

    string to_str() const {
        string s = "(δ ";
        s +=  old.to_str() + "->" + new_.to_str();
        s += ")";
        return s;
    }

    virtual set<Index> free() const {
        set<Index> s;
        s.insert(old);
        s.insert(new_);
        return s;
    }

    Expr *grad(string arr, vector<Index> ixs) {
        return new ConstantInt(0);
    };

    Expr *subst(Index sold, Index snew) const {
        // (i->k) delta_ij : delta_kj
        // (i->k) delta_ji : delta_jk
        // (i->k) delta_ii : delta_kk
        // (i->k) delta_jl : delta_jl
        return new Delta(old == sold ? snew : old,
                    new_ == sold ? snew : new_);
    }


};


struct Arr : public Expr {
    string name;
    Shape sh;
    Arr(string name, Shape sh) : 
        Expr(ExprType::Arr), name(name), sh(sh) {};
    Arr(string name) :
        Expr(ExprType::Arr), name(name), sh(Shape::zerod()) {}
    Arr(string name, int ix1) : 
        Expr(ExprType::Arr), name(name), sh(Shape::oned(ix1)) {}
    Arr(string name, int ix1, int ix2) : 
        Expr(ExprType::Arr), name(name), sh(Shape::twod(ix1, ix2)) {}

    string to_str() const { return name + " " + sh.to_str(); };

    set<Index> free() const {
        return set<Index>();
    }

    Expr *grad(string arr, vector<Index> ixs) {
        // if we ever get here, then we should be 0-dimensional
        assert(ixs.size() == 0);
        if (arr == name) return new ConstantInt(1);
        return new ConstantInt(0);
    };

    Expr *subst(Index old, Index new_) const {
        // if we ever get here, then we should be 0-dimensional
        return new Arr(name);
    }

    Shape shape() const { return sh; }

    // helpers to index
    Expr *ix(vector<Index> ix);
    Expr *ix(Index ix1);
    Expr *ix(Index ix, Index ix2);


};


struct Add : public Expr {
    vector<Expr*>inner;

    Add(vector<Expr *> inner) : 
        Expr(ExprType::Add), inner(inner) {
            assert(inner.size() >= 1);
        };

    Add (Expr *l, Expr *r) : Expr(ExprType::Add) {
        inner.push_back(l);
        inner.push_back(r);

    }

    string to_str() const {
        string s = "(+ ";

        for(int i = 0; i < (int)inner.size(); ++i) {
            s += inner[i]->to_str() + 
                (i < (int)inner.size() - 1 ? " " : "");
        }
        s += ")";
        return s;
    }

    set<Index> free() const {
        set<Index> f;

        for(Expr *i: inner) {
            set<Index> ifree = i->free();
            f.insert(ifree.begin(), ifree.end());
        }
        return f;
    }

    Expr *grad(string name, vector<Index> ixs) {
        vector<Expr *>dinner;
        for(Expr *e : inner) {
            dinner.push_back(e->grad(name, ixs));
        }
        return new Add(dinner);
    }

    Expr *subst(Index old, Index new_) const {
        vector<Expr *> sinner;
        for(Expr *e : inner) {
            sinner.push_back(e->subst(old, new_));
        }
        return new Add(sinner);
    }
};


struct Mul : public Expr {
    vector<Expr *>inner;
   
    Mul (Expr *l, Expr *r) : Expr(ExprType::Mul) {
        inner.push_back(l);
        inner.push_back(r);
    }

    Mul(vector<Expr *> inner) : Expr(ExprType::Mul), inner(inner) {
        assert(inner.size() >= 1);
    };

    string to_str() const {
        string s =  "(* ";
        for(int i = 0; i < (int)inner.size(); ++i) {
            s += inner[i]->to_str();
            s += (i < (int)inner.size() - 1 ? " " : "");
        }
        return s;
    }

    set<Index> free() const {
        set<Index> f;

        for(Expr *i: inner) {
            set<Index> ifree = i->free();
            f.insert(ifree.begin(), ifree.end());
        }
        return f;
    }

    Expr *grad(string name, vector<Index> ixs) {

        // d(fgh) = df gh + f dg h + fg dh
        vector<Expr *> dsum;

        for(int i = 0; i < (int)inner.size(); ++i) {
            vector<Expr *> dinner_i = inner;
            dinner_i[i] = dinner_i[i]->grad(name, ixs);

            dsum.push_back(new Mul(dinner_i));
        }

        return new Add(dsum);
    }

    Expr *subst(Index old, Index new_) const {
        vector<Expr *> sinner;
        for(Expr *e : inner) {
            sinner.push_back(e->subst(old, new_));
        }
        return new Mul(sinner);
    }
};

Expr *exprNegate(Expr *e) {
    return new Mul(new ConstantInt(-1), e);
};

Expr *exprsub(Expr *l, Expr *r) {
    return new Add(l, exprNegate(r));
}

struct Slice : public Expr {
    Arr *arr;
    vector<Index> ixs;

    Slice(Arr *arr, vector<Index> ixs): 
        Expr(ExprType::Slice), arr(arr), ixs(ixs) {
            assert(arr && "must only slice arrays");
            // ensure that we are fully indexing the array.
            if(arr->shape().ndim != (int)ixs.size()) {
                cout << "array " << arr->to_str() << "| shape: " 
                    << arr->shape().to_str() << " | nixs: " << ixs.size()
                    << "\n";
            };
            assert(arr->shape().ndim == (int)ixs.size());

        }; 

    string to_str() const  {
        string s =  "(! " + arr->to_str() + " ";

        for(int i = 0; i < (int)ixs.size(); ++i) {
            s += ixs[i].to_str() + (i < (int)ixs.size() - 1 ? " " : "");
        }

        s += ")";
        return s;
    }

    set<Index> free() const {
        set<Index> s = arr->free();
        for (Index ix :ixs) { s.insert(ix); };
        return s;
    }

    Expr *grad(string name, vector<Index> gixs) {
        // this IS the slice of an array, but not the array we are
        // looking for. Return zero
        if(name != arr->name) { return new ConstantInt(0); }

        // we ARE the slce of the array we were looking for.
        assert(name == arr->name);

        assert(gixs.size() >= 1);
        assert(gixs.size() == ixs.size());

        // create deltas, one for each index of the array.
        Expr *cur = new Delta(ixs[0], gixs[0]);
        for(int i = 1; i < (int)ixs.size(); ++i) {
            cur = new Mul(cur, new Delta(ixs[i], gixs[i]));
        }
        return cur;
    }

    // substitute old indeces for new indeces.
    virtual Expr *subst(Index old, Index new_) const {
        vector<Index> ixsnew;
        for(int i = 0; i < (int)ixs.size(); ++i) {
            ixsnew.push_back(ixs[i] == old ? new_ : ixs[i]);
        }
        return new Slice(arr, ixsnew);
    }

};


Expr *Arr::ix(vector<Index> ixs) {
    return new Slice(this, ixs);
}

Expr *Arr::ix(Index ix1) {
    return new Slice(this, {ix1});
}

Expr *Arr::ix(Index ix1, Index ix2) {
    return new Slice(this, {ix1, ix2});
}


struct Contract : public Expr {
    Index ix;
    Expr *inner;
    Contract (Index ix, Expr *inner) : Expr(ExprType::Contract), ix(ix),
        inner(inner) {};

    string to_str() const {
        return "(>< " + ix.to_str() + " " + inner->to_str() +  ")";
    }

    set<Index> free() const {
        set<Index> infree = inner->free();
        auto it = infree.find(ix);
        if (it != infree.end()) infree.erase(it);
        return infree;
    }

    Expr *grad(string name, vector<Index> ixs) {
        return new Contract(ix, inner->grad(name, ixs));
    }

    Expr *subst(Index old, Index new_) const {
        return new Contract(ix == old ? new_ : ix, inner);
    }

};


Expr *Expr::contract(Index ix) {
    return new Contract(ix, this);
}

struct Tanh : public Expr {
    Expr *inner;
    Tanh(Expr *inner) : Expr(ExprType::Tanh), inner(inner) {};

    string to_str() const {
        return "(tanh " + inner->to_str() + ")";
    }

    set<Index> free() const {
        return inner->free();
    }


    Expr *subst(Index old, Index new_) const { 
        return new Tanh(inner->subst(old, new_));
    }

    Expr *grad(string name, vector<Index> ixs) {
        Expr *dtan = exprsub(new ConstantInt(1), new Mul(new Tanh(inner), new
                    Tanh(inner)));
        Expr *dinner = inner->grad(name, ixs);
        return new Mul(dtan, dinner);
    }

};

struct ExprVisitor {
    Expr *visitExpr(Expr *e) {
        if (Add *a = dynamic_cast<Add *>(e)) {
            return visitAdd(a);
        }
        else if (Mul *m = dynamic_cast<Mul *>(e)) {
            return visitMul(m);
        }
        else if (Contract *c = dynamic_cast<Contract *>(e)) {
            return visitContract(c);
        } 
        else {
            return e;
        }

        // assert(false && "unknown expr type!");

    };

    virtual Expr *visitMul(Mul *m) { 
        vector<Expr *> inner;
        for(Expr *e : m->inner) {
            inner.push_back(visitExpr(e));
        }
        return new Mul(inner);
    }

    virtual Expr *visitAdd(Add *a) { 
        vector<Expr *> inner;
        for(Expr *e : a->inner) {
            inner.push_back(visitExpr(e));
        }
        return new Add(inner);
    };
    virtual Expr *visitContract(Contract *c) { 
        Expr *inner = visitExpr(c->inner);
        return new Contract(c->ix, inner);
    }
};

enum class StmtType {
    Assign,
    Incr,
};
struct Stmt {
    StmtType type;
    Arr *lhs;
    Expr *rhs;

    Stmt(StmtType type, Arr *lhs, Expr *rhs) : type(type),
        lhs(lhs), rhs(rhs) {};

    string to_str() {

        string eqname = "";
        switch(type) {
            case StmtType::Assign: eqname = ":="; break;
            case StmtType::Incr: eqname = "+="; break;
        }
        return "(" + lhs->to_str() + " " + eqname + " " +
            rhs->detailed_to_str() + ")";
    }
};

struct Program {
    list<Stmt> stmts;

    void assign(Arr *arr, Expr *rhs) {
        for(Stmt &s : stmts) {
            assert(s.lhs != arr && "Reusing arrays not allowed (SSA)");
            assert(s.lhs->name != arr->name && "Different array objects with the same name are not allowed (SSA)");
        }

        stmts.push_back(Stmt(StmtType::Assign, arr, rhs));
    }

    void incr(Arr *arr, Expr *rhs) {
        assert(arr != nullptr && "creating an array");

        stmts.push_back(Stmt(StmtType::Incr, arr, rhs));
    }

    string to_str() {
        string str = "";
        for (Stmt s: stmts) {
            str += s.to_str() + "\n";
        }
        return str;
    }

    Expr *&operator [](Arr *arr) {
        for (Stmt &s : stmts) {
            if (s.lhs == arr) return s.rhs;
        }
        assert(false && "no such array present");
    }
};


// (>< (+ a b c)) = (+ (>< a) (>< b) (>< c))
struct PushContractionInwardsVisitor : ExprVisitor {
    virtual Expr* visitContract(Contract *c) {
        Add *a = dynamic_cast<Add *>(c->inner);
        if (!a) return c;

        vector<Expr *> inner;
        for(Expr *e : a->inner) {
            inner.push_back(new Contract(c->ix, visitExpr(e)));
        }
        return new Add(inner);
    }
};


bool is_const_zero(const Expr *e) {
    const ConstantInt *i = dynamic_cast<const ConstantInt*>(e);
    if (!i) return false;
    return i->i == 0;
}

bool is_const_one(const Expr *e) {
    const ConstantInt *i = dynamic_cast<const ConstantInt*>(e);
    if (!i) return false;
    return i->i == 1;
}


bool is_all_const_zero(const vector<Expr *> &elist) {
    for(const Expr *e : elist) {
        if (!is_const_zero(e)) return false;
    }
    return true;
}


bool is_any_const_zero(const vector<Expr *> &elist) {
    for(const Expr *e : elist) {
        if (is_const_zero(e)) return true;
    }
    return false;
}

struct ConstantFoldVisitor : ExprVisitor {
    virtual Expr *visitMul(Mul *m) {

        // constant fold multiplication with 0, remove multiplication
        // with 1.
        if (is_any_const_zero(m->inner)) { return new ConstantInt(0); }

        vector<Expr *> inner;
        for(Expr *e : m->inner) {
            if(is_const_one(e)) continue;
            inner.push_back(visitExpr(e));
        }

        if (inner.size() == 1) { return inner[0]; }

        return new Mul(inner);
    }

    virtual Expr *visitAdd(Add *a) {
        vector<Expr *> inner;
        for(Expr *e : a->inner) {
            if (is_const_zero(e)) continue;
            inner.push_back(visitExpr(e));
        }

        if (inner.size() == 1) return inner[0];
        return new Add(inner);
    }

    virtual Expr *visitContract(Contract *c) {
        if (is_const_zero(c->inner)) return new ConstantInt(0);
        return ExprVisitor::visitContract(c);
    }
};

// return a Delta that is within es, which replaces the
// the index ix. return -1 otherwise.
int findDeltaForIndex(Index ix, vector<Expr *> es) {
    for(int i = 0; i < (int)es.size(); ++i) {
        Delta *d = dynamic_cast<Delta *>(es[i]);
        if(d && d->old == ix) return i;
    }
    return -1;
}

// eliminate (>< i (* t1 t2 (δ i->j) t3)) with 
//     (* t1[i->j] t2[i->j] t3[i->j])
struct EliminateContractionVisitor : public ExprVisitor {
    virtual Expr *visitContract(Contract *c) {
        Mul *m = dynamic_cast<Mul *>(c->inner);
        if (!m) { 
            return new Contract(c->ix, ExprVisitor::visitExpr(c->inner));
        };

        vector<Expr *> subst;
        const int deltaix = findDeltaForIndex(c->ix, m->inner);
        if (deltaix == -1) { 
            return new Contract(c->ix, ExprVisitor::visitExpr(c->inner));
        }

        const Delta *d = dynamic_cast<Delta *>(m->inner[deltaix]);
        assert(d && "output from findDeltaForIndex should be Delta");

        // success
        vector<Expr *> substInner;
        for(Expr *e : m->inner) {
            if (e == d) continue;
            substInner.push_back(ExprVisitor::visitExpr(e->subst(c->ix,
                            d->new_)));
        }
        return new Mul(substInner);
    };
};


Expr *pushContractionsInwards(Expr *e) {
    if (Add *a = dynamic_cast<Add *>(e)) {
        vector<Expr *> inner;
        for(Expr *e : a->inner) inner.push_back(pushContractionsInwards(e));
        return new Add(inner);
    }
    if (Mul *m = dynamic_cast<Mul *>(e)) {
        vector<Expr *> inner;
        for(Expr *e : m->inner) inner.push_back(pushContractionsInwards(e));
        return new Mul(inner);
    }
    // (>< (+ a b c)) = (+ (>< a) (>< b) (>< c))
    else if (Contract *c = dynamic_cast<Contract *>(e)) {
        Add *a = dynamic_cast<Add *>(c->inner);
        vector<Expr *> inner;
        for(Expr *e : a->inner) {
            inner.push_back(new Contract(c->ix, pushContractionsInwards(e)));
        }
        return new Add(inner);

    }
    return e;
}

// convert (+ (+ a b) (+ c d)) to (+ a b c d) and similary
// for multiplication
struct FlattenVisitor : public ExprVisitor {
    virtual Expr *visitMul(Mul *m) {
        vector<Expr *> inner;
        for(Expr *e : m->inner) {
            Mul *me = dynamic_cast<Mul *>(e);
            if (!me) { inner.push_back(e); }
            else {
                // copy everything from the inner multiplication
                inner.insert(inner.end(), me->inner.begin(), me->inner.end());
            }
        }
        return new Mul(inner);
    }

    virtual Expr *visitAdd(Add *a) {
        vector<Expr *> inner;
        for(Expr *e : a->inner) {
            Add *ae = dynamic_cast<Add *>(e);
            if (!ae) { inner.push_back(e); }
            else {
                // copy everything from the inner multiplication
                inner.insert(inner.end(), ae->inner.begin(), ae->inner.end());
            }
        }
        return new Add(inner);
    }
};


// move dirac deltas leftwards
/*
Expr *reassocDelta(Expr *e) {
    if (Add *a = dynamic_cast<Add *>(e)) {
        return new Add(reassocDelta(a->l), reassocDelta(a->r));
    } else if (Mul *m = dynamic_cast<Mul *>(e)) {
        if (Delta *dr = dynamic_cast<Delta *>(m->r)) {
            return new Mul(dr, reassocDelta(m->l));
        } else {
            return new Mul(reassocDelta(m->l), reassocDelta(m->r));
        }
    } else if (Contract *c = dynamic_cast<Contract *>(e)) {
        return new Contract(c->ix, reassocDelta(c->inner));
    }
    return e;
}

// eliminate (>< i (* (δ i->j) t1)) with t1[i/j]
Expr *eliminateContractions(Expr *e)  {
    if (Add *a = dynamic_cast<Add *>(e)) {
        return new Add(eliminateContractions(a->l), eliminateContractions(a->r));
    }
    else if (Mul *m = dynamic_cast<Mul *>(e)) {
        return new Mul(eliminateContractions(m->l), eliminateContractions(m->r));
    } else if (Contract *c = dynamic_cast<Contract *>(e)) {
        if (!c) return e;
        Mul *m = dynamic_cast<Mul *>(c->inner);
        if (!m) return e;
        Delta *d = dynamic_cast<Delta *>(m->l);
        if (!d) return e;

        if (c->ix == d->old) {
            return m->r->subst(d->old, d->new_);
        }
    }

    return e;
};
*/


/*
Expr *constantFold(Expr *e) {
    if (Mul *m = dynamic_cast<Mul*>(e)) {
        if (is_const_zero(m->l)) return new ConstantInt(0);
        if (is_const_zero(m->r)) return new ConstantInt(0);
        if (is_const_one(m->r)) return m->l;
        if (is_const_one(m->l)) return m->r;
    }

    if (Add *a = dynamic_cast<Add*>(e)) {
        if (is_const_zero(a->l)) return a->r;
        if (is_const_zero(a->r)) return a->l;

        return new Add(constantFold(a->l), constantFold(a->r));
    }

    if (Contract *c = dynamic_cast<Contract*>(e)) {
        // contracting over zero is just 0
        if (is_const_zero(c->inner)) { return new ConstantInt(0); }
        return new Contract(c->ix, constantFold(c->inner));
    }

    return e;
}
*/

Expr *simplify(Expr *e, bool debug) {
    PushContractionInwardsVisitor pushv;
    ConstantFoldVisitor cfv;
    EliminateContractionVisitor ecv;
    FlattenVisitor fv;
    for(int i = 0; i < 5; ++i) {
        if (debug) {
            cout << "--\n";
            cout << i << "|"  << e->to_str() << "\n";
        }
        e = pushv.visitExpr(e);
        if (debug) {
            cout << i << "|PUSH|" << e->to_str() << "\n";
        }
        e = cfv.visitExpr(e);
        if (debug) {
            cout << i << "|FOLD|" << e->to_str() << "\n";
        }
        e = ecv.visitExpr(e);
        if (debug) {
            cout << i << "|ELIM|" << e->to_str() << "\n";
        }
        e = fv.visitExpr(e);
        if (debug) {
            cout << i << "|FLAT|" << e->to_str() << "\n";
        }
    }
    return e;
}


Expr *Expr::normalize() {
    return simplify(this, false);
};

void add_word_to_vocab(string w) {
    if (vocab.find(w) != vocab.end()) { return; }
    word2ix[w] = last_word_index;
    ix2word[last_word_index] = w;
    last_word_index++;
}

bool is_char_word_break(char c) {
    return c == ' ' || c== '\t' || c == '\n' || c == '.';
}

// disable LSAN. Only keep ASAN. I leak far too much memory ;) 
extern "C" int __lsan_is_turned_off() { return 1; }

// consume whitespace and get the next word, or empty string if run out.
string parse_word(FILE *f) {
    char c;
    do {
        c = fgetc(f);
    } while(c != EOF && is_char_word_break(c));

    if (c == EOF) { return ""; }
    ungetc(c, f);

    string w;
    while(1) {
        c = fgetc(f); 
        if (is_char_word_break(c)) break;
        w += c;
    } 

    return w;
}

bool consume_till_sentence_end_or_word(FILE *f) {
    while(1) {
        char c = fgetc(f);
        if (c == '.') { return true; }
        if (c == ' ' || c == '\t' || c == '\n')  { continue; }
        ungetc(c, f);
        return false;

    }
}


vector<int> parse_sentence(FILE *f) {
    vector<int> sentence;
    while(1) {
        // returns true if '.' was consumed
        if (consume_till_sentence_end_or_word(f)) { return sentence; }
        string w = parse_word(f);
        if (w == "") { return sentence; }
        auto it = word2ix.find(w);
        assert(it != word2ix.end());
        sentence.push_back(it->second);
    }
}


void test_expr_dot() {
    int NDIMS = 3;
    Arr *a = new Arr("a", NDIMS);
    Arr *b = new Arr("b", NDIMS);
    Index i("i");
    Expr *mul = new Mul(a->ix(i), b->ix(i));
    Expr *dot = new Contract(i, mul);

    cout << "***dot***\n";
    cout << dot->detailed_to_str() << "\n";
    Index k("k");
    Expr *grad = dot->grad("a", {k});
    cout << "## dot->grad[a k]: ##\n\t" << grad->detailed_to_str() << "\n";
    simplify(grad, true);
}

// contract over all free indeces
Expr* contractOverFree(Expr *e) {
    set<Index> free = e->free();
    for(Index ix : free) {
        e = new Contract(ix, e);
    }
    return e;
}

void test_expr_matvec() {
    const int NOUT = 3;
    const int NIN = 3;
    Arr *a = new Arr("a", NOUT, NIN);
    Arr *b = new Arr("b", NIN);
    Index i("i"), j("j");
    Expr *mul = new Mul(a->ix(i, j), b->ix(j));
    Expr *matvec = new Contract(j, mul);

    cout << "***mul***\n";
    cout << matvec->detailed_to_str() << "\n";


    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *grada = contractOverFree(matvec)->grad("a", 
            {Index("k"), Index("l")});
    cout << "## mul->grad[a k l]: ##\n\t" << grada->to_str() << "\n";
    simplify(grada, true);

    Expr *gradb = contractOverFree(matvec)->grad("b", {Index("k")});
    cout << "## mul->grad[b k]: ##\n\t" << gradb->to_str() << "\n";
    simplify(gradb, true);
}

void test_expr_tanh() {
    const int NDIM = 3;
    Arr *a = new Arr("a", NDIM, NDIM);
    Index i("i"), j("j");
    Expr *aixed = a->ix(i, j);
    Expr *matvec = new Tanh(aixed);

    cout << "***tanh on matrix***\n";
    cout << matvec->detailed_to_str() << "\n";


    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *grada = contractOverFree(matvec)->grad("a", 
            {Index("k"), Index("l")});
    cout << "## mul->grad[a k l]: ##\n\t" << grada->to_str() << "\n";
    simplify(grada, true);

    // Expr *gradb = contractOverFree(matvec)->grad("b", {Index("k")});
    // cout << "## mul->grad[b k]: ##\n\t" << gradb->to_str() << "\n";
    // simplify(gradb);

}

void test_program_dot() {
    Program p;

    static const int ARRSIZE = 10;
    Arr *a = new Arr("a", ARRSIZE);
    Arr *b = new Arr("b", ARRSIZE);
    Arr *grada = new Arr("grada", ARRSIZE);
    Arr *gradb = new Arr("gradb", ARRSIZE);
    Arr *dot = new Arr("dot");
    Index i("i"), k("k");
    p.assign(dot, new Contract(i, new Mul(a->ix(i), b->ix(i))));
    p.assign(grada, new Mul(new ConstantFloat(1e-2),
                p[dot]->grad("a", {k})->normalize()));
    p.assign(gradb, new Mul(new ConstantFloat(1e-2),
                p[dot]->grad("b", {k})->normalize()));

    p.incr(a, grada);
    p.incr(b, gradb);

    cout << "***program dot**\n";
    cout << p.to_str();
}


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: <input-path>\n";
        return 1;
    }
    FILE *f = fopen(argv[1], "r");

    // construct vocab
    while(1) {
        string w = parse_word(f);
        cout << "found word: " << w << "\n";
        if (feof(f)) { break; }
        add_word_to_vocab(w);
    }
    rewind(f);
    while(1) {
        vector<int> sentence = parse_sentence(f);
        sentences.push_back(sentence);
        if (feof(f)) { break; }
    }
    fclose(f);

    for(auto it: word2ix) {
        cout << "|" << it.first << "|->" << it.second << "\n";
    }

    cout << "SENTENCES:\n";
    for(auto s: sentences) {
        for (auto w : s) {
            cout << w << " ";
        }
        cout << "\n";
    }

    // use_expr();
    test_expr_dot();
    test_expr_matvec();
    test_expr_tanh();
    test_program_dot();
    // test_expr_rnn_batched();
}
