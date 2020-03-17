//
// Created by bollu on 18/03/20.
//

#ifndef LANGUAGEMODELS_LANG_H
#define LANGUAGEMODELS_LANG_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
using Dim = int;
using DimSize = int;


struct Shape {
    static const int MAXDIM = 10;
    int ndim = -42;
    DimSize vals[Shape::MAXDIM] = {-42};

    int nelem() {
        int n = 1;
        for (int i = 0; i < ndim; ++i) n *= vals[i];
        return n;
    }

    int operator[](int dim) const {
        assert(dim >= 0);
        assert(dim < ndim);
        return vals[dim];
    }

    bool operator==(const Shape &other) const {
        if (ndim != other.ndim) return false;
        for (int i = 0; i < ndim; ++i) {
            if (vals[i] != other.vals[i]) return false;
        }
        return true;
    }

    bool operator!=(const Shape &other) const { return !(*this == other); }

    // lex comparison
    bool operator<(const Shape &other) const {
        if (ndim < other.ndim) return true;
        if (ndim > other.ndim) return false;

        assert(ndim == other.ndim);

        for (int i = 0; i < ndim; ++i) {
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
        string s = "<";
        for (int i = 0; i < ndim; ++i) {
            s += to_string(vals[i]) + (i < ndim - 1 ? " " : "");
        }
        s += ">";
        return s;
    }

    Shape removeOutermostN(int N) const {
        Shape sh;
        assert(N <= ndim);
        sh.ndim = ndim - N;
        for (int i = 0; i < ndim - N; ++i) {
            sh.vals[i] = vals[i + N];
        }
        return sh;
    }

    // remove an outermost (leftmost) dimension
    Shape removeOutermost() const { return removeOutermostN(1); }

    // append an outermost dimension
    Shape addOutermost(int size) const {
        Shape sh = *this;
        sh.ndim++;
        for (int i = 1; i < ndim + 1; ++i) {
            sh.vals[i] = vals[i - 1];
        }
        sh.vals[0] = size;
        return sh;
    }

    // append another shape inside this shape.
    // So, 0..ndim will be this shape,
    // ndim+1...ndim+ndim' will be next
    Shape appendInside(Shape next) {
        Shape sh;
        sh.ndim = ndim + next.ndim;
        assert(sh.ndim < MAXDIM);

        int ix = 0;
        for (int i = 0; i < ndim; ++i) {
            sh.vals[ix++] = vals[i];
        }

        for (int i = 0; i < next.ndim; ++i) {
            sh.vals[ix++] = next.vals[i];
        }

        return sh;
    }
};

enum class ExprType {
    Add,
    Sub,
    Mul,
    Arr,
    Index,
    Contract,
    Delta,
    ConstantInt,
    ConstantFloat,
    Tanh,
};

struct Index;
struct Arr;
struct Program;

enum class NumType { Float, Int };

// TODO: we need some way for an Arr to decay into an Index gracefully?
typedef struct Arr {
    string name;
    Shape sh;
    NumType ty = NumType::Float;

    Arr() : name("UNINITIALIZED(-42)"){};
    Arr(string name, Shape sh, NumType ty = NumType::Float)
            : name(name), sh(sh), ty(ty){};
    Arr(string name) : name(name), sh(Shape::zerod()) {}
    Arr(string name, int ix1, NumType ty = NumType::Float)
            : name(name), sh(Shape::oned(ix1)), ty(ty) {}
    Arr(string name, int ix1, int ix2, NumType ty = NumType::Float)
            : name(name), sh(Shape::twod(ix1, ix2)), ty(ty) {}

    string to_str() const { return name + (sh.ndim > 0 ? sh.to_str() : ""); };

    Shape shape() const { return sh; }

    // helpers to index
    Index *ix(vector<Arr> ix);
    Index *ix();
    Index *ix(Arr ix1);
    Index *ix(Arr ix1, Arr ix2);

    bool operator==(const Arr &other) const {
        return name == other.name && sh == other.sh;
    }

    bool operator!=(const Arr &other) const { return !(*this == other); }

    bool operator<(const Arr &other) const {
        return name < other.name || (name == other.name && sh < other.sh);
    }

} Arr;

struct Expr {
    // gives type of expression
    ExprType ty;
    Expr(ExprType ty) : ty(ty){};

    // print as string
    virtual string to_str() const = 0;

    // return free indeces
    virtual set<Arr> free() const = 0;

    // substitute old indeces for new indeces.
    virtual Expr *subst(const Arr old, const Arr new_) const = 0;

    // creates dirac deltas by taking gradients
    virtual Expr *grad_(string name, vector<Arr> ixs) = 0;

    Expr *grad(string name, vector<Arr> ixs);

    string to_str_with_shape() const;

    // simplify the expression to normal form
    Expr *normalize();

    virtual Shape shape() const = 0;
};

map<Arr, set<pair<int, Index *>>> verifyIndexing(Expr *e);

struct ConstantInt : public Expr {
    int i;
    ConstantInt(int i) : Expr(ExprType::ConstantInt), i(i){};

    string to_str() const { return std::to_string(i); }
    set<Arr> free() const { return {}; }
    Expr *grad_(string name, vector<Arr> ixs) { return new ConstantInt(0); }

    Expr *subst(const Arr old, const Arr new_) const {
        return new ConstantInt(i);
    }

    Shape shape() const { return Shape::zerod(); }
};

string Expr::to_str_with_shape() const {
    string s = "";
    s += "(size<";
    for (Arr i : free()) s += i.name + ",";
    s += "> ";
    s += to_str();

    s += ")";
    return s;
}

Expr *Expr::grad(string name, vector<Arr> ixs) {
    set<string> freeNames;
    for (const Arr f : this->free()) {
        freeNames.insert(f.name);
    }
    for (const Arr ix : ixs) {
        if (freeNames.count(ix.name)) {
            cerr << "\nreused name for gradient index: " << ix.name << "\n";
            assert(false && "gradient indexing names must be fresh");
        }
    }

    return grad_(name, ixs);
};

struct ConstantFloat : public Expr {
    float f;
    ConstantFloat(float f) : Expr(ExprType::ConstantFloat), f(f){};

    string to_str() const { return std::to_string(f); }
    set<Arr> free() const { return {}; }
    Expr *grad_(string name, vector<Arr> ixs) { return new ConstantInt(0); }

    Expr *subst(const Arr old, const Arr new_) const {
        return new ConstantFloat(f);
    }

    Shape shape() const { return Shape::zerod(); }
};

struct Delta : public Expr {
    const Arr old;
    const Arr new_;

    Delta(const Arr old, const Arr new_)
            : Expr(ExprType::Delta), old(old), new_(new_) {
        assert(old.sh == Shape::zerod());
        assert(new_.sh == Shape::zerod());
    };

    string to_str() const {
        string s = "(δ ";
        s += old.to_str() + "->" + new_.to_str();
        s += ")";
        return s;
    }

    virtual set<Arr> free() const { return {old, new_}; }

    Expr *grad_(string arr, vector<Arr> ixs) { return new ConstantInt(0); };

    Expr *subst(const Arr sold, const Arr snew) const {
        // (i->k) delta_ij : delta_kj
        // (i->k) delta_ji : delta_jk
        // (i->k) delta_ii : delta_kk
        // (i->k) delta_jl : delta_jl
        return new Delta(old == sold ? snew : old, new_ == sold ? snew : new_);
    }

    Shape shape() const { return Shape::zerod(); }
};

struct Add : public Expr {
    vector<Expr *> inner;

    Add(vector<Expr *> inner) : Expr(ExprType::Add), inner(inner) {
        assert(inner.size() >= 1);

        for (Expr *e : inner) {
            assert(e->shape() == Shape::zerod());
        }
    };

    Add(Expr *l, Expr *r) : Expr(ExprType::Add) {
        inner.push_back(l);
        inner.push_back(r);
    }

    string to_str() const {
        string s = "(+ ";

        for (int i = 0; i < (int)inner.size(); ++i) {
            s += inner[i]->to_str() + (i < (int)inner.size() - 1 ? " " : "");
        }
        s += ")";
        return s;
    }

    set<Arr> free() const {
        set<Arr> f;

        for (Expr *i : inner) {
            set<Arr> ifree = i->free();
            f.insert(ifree.begin(), ifree.end());
        }
        return f;
    }

    Expr *grad_(string name, vector<Arr> ixs) {
        vector<Expr *> dinner;
        for (Expr *e : inner) {
            dinner.push_back(e->grad(name, ixs));
        }
        return new Add(dinner);
    }

    Expr *subst(const Arr old, const Arr new_) const {
        vector<Expr *> sinner;
        for (Expr *e : inner) {
            sinner.push_back(e->subst(old, new_));
        }
        return new Add(sinner);
    }

    Shape shape() const { return Shape::zerod(); }
};

struct Sub : public Expr {
    Expr *l, *r;

    Sub(Expr *l, Expr *r) : Expr(ExprType::Sub), l(l), r(r) {
        assert(l->shape() == Shape::zerod());
        assert(r->shape() == Shape::zerod());
    }

    string to_str() const {
        return "(- " + l->to_str() + " " + r->to_str() + ")";
    }

    set<Arr> free() const {
        set<Arr> f;
        set<Arr> lf = l->free();
        f.insert(lf.begin(), lf.end());

        set<Arr> rf = r->free();
        f.insert(rf.begin(), rf.end());
        return f;
    }

    Expr *grad_(string name, vector<Arr> ixs) {
        return new Sub(l->grad(name, ixs), r->grad(name, ixs));
    }

    Expr *subst(const Arr old, const Arr new_) const {
        return new Sub(l->subst(old, new_), r->subst(old, new_));
    }

    Shape shape() const { return Shape::zerod(); }
};

struct Mul : public Expr {
    vector<Expr *> inner;

    Mul(Expr *l, Expr *r) : Expr(ExprType::Mul) {
        inner.push_back(l);
        inner.push_back(r);

        assert(l->shape() == Shape::zerod());
        assert(r->shape() == Shape::zerod());
    }

    Mul(vector<Expr *> inner) : Expr(ExprType::Mul), inner(inner) {
        assert(inner.size() >= 1);
    };

    string to_str() const {
        string s = "(* ";
        for (int i = 0; i < (int)inner.size(); ++i) {
            s += inner[i]->to_str();
            s += (i < (int)inner.size() - 1 ? " " : "");
        }
        s += ")";
        return s;
    }

    set<Arr> free() const {
        set<Arr> f;

        for (Expr *i : inner) {
            set<Arr> ifree = i->free();
            f.insert(ifree.begin(), ifree.end());
        }
        return f;
    }

    Expr *grad_(string name, vector<Arr> ixs) {
        // d(fgh) = df gh + f dg h + fg dh
        vector<Expr *> dsum;

        for (int i = 0; i < (int)inner.size(); ++i) {
            vector<Expr *> dinner_i = this->inner;
            dinner_i[i] = dinner_i[i]->grad(name, ixs);
            dsum.push_back(new Mul(dinner_i));
        }

        return new Add(dsum);
    }

    Expr *subst(Arr old, Arr new_) const {
        vector<Expr *> sinner;
        for (Expr *e : inner) {
            sinner.push_back(e->subst(old, new_));
        }
        return new Mul(sinner);
    }

    Shape shape() const { return Shape::zerod(); }
};

struct Index : public Expr {
    Arr arr;
    vector<Arr> ixs;

    Index(Arr arr, vector<Arr> ixs)
            : Expr(ExprType::Index), arr(arr), ixs(ixs) {
        for (const Arr ix : ixs) {
            assert(ix.sh.ndim == 0 && "slices must be zero-dimensional");
        }

        // ensure that we are fully indexing the array.
        if ((int)ixs.size() > arr.sh.ndim) {
            cerr << "\n";
            cerr << "array " << arr.to_str() << "| shape: " << arr.sh.to_str()
                 << " | nixs: " << ixs.size();
            cerr << "\nixs: ";
            for (Arr a : ixs) cerr << a.name << " ";
            cerr << "\n" << flush;
        };
        assert((int)ixs.size() <= arr.sh.ndim);
    };

    string to_str() const {
        string s = arr.to_str() + "[";

        for (int i = 0; i < (int)ixs.size(); ++i) {
            s += ixs[i].to_str() + (i < (int)ixs.size() - 1 ? " " : "");
        }

        s += "]";
        return s;
    }

    set<Arr> free() const {
        set<Arr> s;
        s.insert(ixs.begin(), ixs.end());
        return s;
    }

    Expr *grad_(string name, vector<Arr> gixs) {
        // can only take gradients if the slice is fully saturated
        assert((int)ixs.size() == arr.sh.ndim);

        // this IS the Index of an array, but not the array we are
        // looking for. Return zero
        if (name != arr.name) {
            return new ConstantInt(0);
        }

        // we ARE the slce of the array we were looking for.
        assert(name == arr.name);

        assert(gixs.size() == ixs.size());

        // we have a scalar.
        if (gixs.size() == 0) {
            if (arr.name == name) {
                return new ConstantInt(1);
            } else {
                return new ConstantInt(0);
            }
        }

        assert(gixs.size() >= 1);

        // create deltas, one for each index of the array.
        Expr *cur = new Delta(ixs[0], gixs[0]);
        for (int i = 1; i < (int)ixs.size(); ++i) {
            cur = new Mul(cur, new Delta(ixs[i], gixs[i]));
        }
        return cur;
    }

    // substitute old indeces for new indeces.
    virtual Expr *subst(Arr old, Arr new_) const {
        vector<Arr> ixsnew;
        for (int i = 0; i < (int)ixs.size(); ++i) {
            ixsnew.push_back(ixs[i] == old ? new_ : ixs[i]);
        }
        return new Index(arr, ixsnew);
    }

    Shape shape() const { return arr.sh.removeOutermostN(ixs.size()); }
};

Index *Arr::ix(vector<Arr> ixs) { return new Index(*this, ixs); }

Index *Arr::ix(Arr ix1) { return new Index(*this, {ix1}); }

Index *Arr::ix() { return new Index(*this, {}); }

Index *Arr::ix(Arr ix1, Arr ix2) { return new Index(*this, {ix1, ix2}); }

struct Contract : public Expr {
    Arr ix;
    Expr *inner;
    Contract(Arr ix, Expr *inner)
            : Expr(ExprType::Contract), ix(ix), inner(inner) {
        assert(inner->shape() == Shape::zerod());
    };

    string to_str() const {
        return "(>< " + ix.to_str() + " " + inner->to_str() + ")";
    }

    set<Arr> free() const {
        set<Arr> infree = inner->free();
        auto it = infree.find(ix);
        if (it != infree.end()) infree.erase(it);
        return infree;
    }

    Expr *grad_(string name, vector<Arr> ixs) {
        return new Contract(ix, inner->grad(name, ixs));
    }

    Expr *subst(Arr old, Arr new_) const {
        // TODO: think about this carefully!
        const bool shadowed = ix == old;
        const Arr newix = shadowed ? new_ : ix;
        return new Contract(newix, inner->subst(old, new_));
    }

    Shape shape() const { return Shape::zerod(); }

    static Expr *contract(vector<Arr> ixs, Expr *e) {
        for (Arr ix : ixs) {
            e = new Contract(ix, e);
        }
        return e;
    }
};

struct Tanh : public Expr {
    Expr *inner;
    Tanh(Expr *inner) : Expr(ExprType::Tanh), inner(inner) {
        assert(inner->shape() == Shape::zerod());
    };

    string to_str() const { return "(tanh " + inner->to_str() + ")"; }

    set<Arr> free() const { return inner->free(); }

    Expr *subst(Arr old, Arr new_) const {
        return new Tanh(inner->subst(old, new_));
    }

    Expr *grad_(string name, vector<Arr> ixs) {
        Expr *dtan = new Sub(new ConstantInt(1),
                             new Mul(new Tanh(inner), new Tanh(inner)));
        Expr *dinner = inner->grad(name, ixs);
        return new Mul(dtan, dinner);
    }

    Shape shape() const { return Shape::zerod(); }
};

struct ExprVisitor {
    Expr *visitExpr(Expr *e) {
        if (Add *a = dynamic_cast<Add *>(e)) {
            return visitAdd(a);
        }
        if (Sub *s = dynamic_cast<Sub *>(e)) {
            return visitSub(s);
        } else if (Mul *m = dynamic_cast<Mul *>(e)) {
            return visitMul(m);
        } else if (Contract *c = dynamic_cast<Contract *>(e)) {
            return visitContract(c);
        } else if (Index *i = dynamic_cast<Index *>(e)) {
            return visitIndex(i);
        } else if (Tanh *tanh = dynamic_cast<Tanh *>(e)) {
            return visitExpr(tanh->inner);
        } else {
            return e;
        }
    };

    virtual Expr *visitMul(Mul *m) {
        vector<Expr *> inner;
        for (Expr *e : m->inner) {
            inner.push_back(visitExpr(e));
        }
        return new Mul(inner);
    }

    virtual Expr *visitAdd(Add *a) {
        vector<Expr *> inner;
        for (Expr *e : a->inner) {
            inner.push_back(visitExpr(e));
        }
        return new Add(inner);
    };

    virtual Expr *visitSub(Sub *s) {
        return new Sub(visitExpr(s->l), visitExpr(s->r));
    };

    virtual Expr *visitContract(Contract *c) {
        Expr *inner = visitExpr(c->inner);
        return new Contract(c->ix, inner);
    }

    virtual Expr *visitIndex(Index *i) { return new Index(i->arr, i->ixs); }
};

enum class AssignType {
    Copy,
    Reference,
    Incr,
};

struct Assign;
struct Stmt {
    virtual string to_str(int depth = 0) const = 0;

    // return the RHS of the expression if the statement has it.
    // return nullptr otherwise;
    virtual void findAssign(Arr arr, Assign **arrptr) = 0;
    virtual vector<Assign> assigns() = 0;
};

template <typename T>
set<T> subtract(const set<T> &a, const set<T> &b) {
    set<T> d;
    for (T t : a) {
        if (!b.count(t)) {
            d.insert(t);
        }
    }
    return d;
}
// (a - b) U (b - a)
template <typename T>
set<T> symmetricdifference(const set<T> &a, const set<T> &b) {
    set<T> d;
    set<T> dab = subtract(a, b);
    d.insert(dab.begin(), dab.end());

    set<T> dba = subtract(b, a);
    d.insert(dba.begin(), dba.end());
    return d;
}

// statement to zero an array
struct ZeroStmt : public Stmt {
    Arr lhs;

    ZeroStmt(Arr lhs) : lhs(lhs){};

    string to_str(int depth) const {
        cout << string(depth, ' ');

        string s = "(";
        s += lhs.to_str();
        s += " := ";
        s += "ZERO)";
        return s;
    }

    virtual void findAssign(Arr arr, Assign **arrptr) {
        (void)arr;
        *arrptr = nullptr;
    }
    virtual vector<Assign> assigns() { return {}; }
};

struct Assign : public Stmt {
    AssignType type;
    // TODO: make this non-pointer.
    Index *lhs;
    // Arr *lhs;
    // vector<const Arr *> indeces;
    Expr *rhs;

    Assign(AssignType type, Index *lhs, Expr *rhs)
            : type(type), lhs(lhs), rhs(rhs) {
        // TODO: add an exhaustiveness check that the set of indeces
        // is equal to the set of free indeces.

        for (Arr ix : lhs->ixs) {
            assert(ix.sh.ndim == 0 &&
                   "can only index with zero dimensional arrays");
        }
        map<Arr, set<pair<int, Index *>>> index2indeces = verifyIndexing(rhs);

        for (int i = 0; i < (int)lhs->ixs.size(); ++i) {
            Arr ix = lhs->ixs[i];
            // the index is not used on the RHS, so we have something
            // like A[i] = 10
            if (!index2indeces.count(ix)) continue;

            set<pair<int, Index *>> equivclass = index2indeces[ix];
            assert(equivclass.size() > 0);

            int refsize;
            Index *refix;
            std::tie(refsize, refix) = *equivclass.begin();

            if (refsize == lhs->arr.sh[i]) continue;

            cerr << "inconsistent use of size in LHS and RHS along index |"
                 << ix.name << "|\n";
            cerr << "  - LHS size: " << lhs->arr.sh[i]
                 << " | array: " << lhs->arr.to_str() << "\n";
            cerr << "  - RHS size: " << refsize
                 << " | index: " << refix->to_str_with_shape() << "\n";
            cerr << "  - RHS: " << rhs->to_str_with_shape() << "\n";
            assert(false && "inconsistent use of size in LHS and RHS");
        }

        if (lhs->shape() != rhs->shape()) {
            cerr << "mismatched array sizes of LHS and rhs:";
            cerr << "\n\tlhs: "
                 << "|shape: " << lhs->shape().to_str()
                 << "|expr:" << lhs->to_str();
            cerr << "\n\trhs: "
                 << "|shape: " << rhs->shape().to_str()
                 << "|expr: " << rhs->to_str() << "\n";
        }

        assert(lhs->shape() == rhs->shape());

        const set<Arr> free = rhs->free();
        if (type == AssignType::Reference) {
            if (lhs->ixs.size() >= free.size()) {
                cerr << "Taking a slice of zero or negative dimension\n";
                cerr << "\tlhs: " << lhs->to_str_with_shape() << "\n";
                cerr << "\tindeces: ";
                for (Arr a : lhs->ixs) cerr << a.name << " ";
                cerr << "\n\texpression: ";
                cerr << rhs->to_str_with_shape() << "\n";
            }

            assert(lhs->ixs.size() < rhs->free().size());
        } else {
            if (free.size() != lhs->ixs.size()) {
                cerr << "mismatch between indeces and number of free "
                        "variables\n";
                cerr << "\tlhs: " << lhs->to_str_with_shape() << "\n";
                cerr << "\tindeces: ";
                for (Arr a : lhs->ixs) cerr << a.name << " ";
                cerr << "\n\texpression: ";
                cerr << rhs->to_str_with_shape() << "\n";
            }

            assert(free.size() == lhs->ixs.size() &&
                   "need those many free variables as free indices");

            set<Arr> symdiff = symmetricdifference(
                    set<Arr>(lhs->ixs.begin(), lhs->ixs.end()), free);
            if (symdiff.size() != 0) {
                cerr << "\tlhs: " << lhs->to_str_with_shape() << "\n";
                cerr << "\tindeces: ";
                for (Arr a : lhs->ixs) cerr << a.name << " ";
                cerr << "\n\texpression: ";
                cerr << rhs->to_str_with_shape() << "\n";

                cerr << "\nkeys not present in LHS: ";
                for (Arr a : free) {
                    if (symdiff.count(a)) cerr << a.name << " ";
                }

                cerr << "\nkeys not present in RHS: ";
                for (Arr a : lhs->ixs) {
                    if (symdiff.count(a)) cerr << a.name << " ";
                }
                cerr << "\n";
            }
            assert(symdiff.size() == 0 &&
                   "index set and free variable set are not equal");
        }
    };

    bool operator==(const Assign &other) const {
        return this->lhs == other.lhs && this->rhs == other.rhs;
    }

    bool operator<(const Assign &other) const {
        return this->lhs < other.lhs ||
               (this->lhs == other.lhs && this->rhs < other.rhs);
    }

    string to_str(int depth) const {
        cout << string(depth, ' ');

        string s = "(";
        s += lhs->to_str();
        s += " ";

        switch (type) {
            case AssignType::Copy:
                s += ":=";
                break;
            case AssignType::Reference:
                s += "&=";
                break;
            case AssignType::Incr:
                s += "+=";
                break;
        }

        s += " " + rhs->to_str_with_shape() + ")";
        return s;
    }

    virtual void findAssign(Arr arr, Assign **arrptr) {
        assert(arrptr);
        if (arr == lhs->arr) {
            *arrptr = this;
        } else {
            *arrptr = nullptr;
        }
    }

    std::vector<Assign> assigns() { return {*this}; }
};

struct Block : public Stmt {
    list<Stmt *> stmts;

    string to_str(int depth) const {
        string str = "";
        for (Stmt *s : stmts) {
            str += string(depth, ' ') + s->to_str() + "\n";
        }
        return str;
    }

    virtual void findAssign(Arr arr, Assign **arrptr) {
        assert(arrptr);
        *arrptr = nullptr;
        for (auto it = stmts.rbegin(); it != stmts.rend(); ++it) {
            (*it)->findAssign(arr, arrptr);
            if (*arrptr) {
                return;
            }
        }
    }

    vector<Assign> assigns() {
        vector<Assign> as;
        for (Stmt *s : stmts) {
            vector<Assign> sas = s->assigns();
            as.insert(as.end(), sas.begin(), sas.end());
        }
        return as;
    }
};

struct Forall : public Stmt {
    vector<Arr> ixs;
    Block inner;

    Forall(vector<Arr> ixs) : ixs(ixs){};

    string to_str(int depth) const {
        string s = "";
        s += "forall ";
        for (Arr ix : ixs) {
            s += ix.name + " ";
        }
        s += "{\n";
        s += inner.to_str(depth + 1);
        s += "\n" + string(depth, ' ') + "}";
        return s;
    }

    virtual void findAssign(Arr arr, Assign **assignptr) {
        return inner.findAssign(arr, assignptr);
    }

    vector<Assign> assigns() { return inner.assigns(); }
};

struct Program {
    Block stmts;

    string to_str() { return stmts.to_str(0); }

    Assign operator[](Arr arr) {
        Assign *a = nullptr;
        stmts.findAssign(arr, &a);
        assert(a);
        return *a;
    }

    bool is_array_assigned(Arr arr) {
        Assign *a = nullptr;
        stmts.findAssign(arr, &a);
        return a != nullptr;
    }

    vector<Assign> assigns() { return stmts.assigns(); }
};

// welcome LLVM my old friend
struct IRBuilder {
    Program &p;
    Block *insertPoint;

    IRBuilder(Program &p) : p(p), insertPoint(&p.stmts){};

    void setInsertPoint(Program &p) { insertPoint = &p.stmts; }

    void setInsertPoint(Forall &f) { insertPoint = &f.inner; }

    void setInsertPoint(Forall *f) { insertPoint = &f->inner; }

    void copy(Index *ix, Expr *rhs) {
        assert(ix != nullptr);
        insertPoint->stmts.push_back(new Assign(AssignType::Copy, ix, rhs));
    }

    void incr(Index *ix, Expr *rhs) {
        assert(ix != nullptr && "creating an array");
        insertPoint->stmts.push_back(new Assign(AssignType::Incr, ix, rhs));
    }

    void reference(Index *ix, Expr *rhs) {
        assert(ix != nullptr && "creating an array");
        insertPoint->stmts.push_back(
                new Assign(AssignType::Reference, ix, rhs));
    }

    void zero(Arr arr) { insertPoint->stmts.push_back(new ZeroStmt(arr)); }

    Forall *insertFor(vector<Arr> ix) {
        Forall *f = new Forall(ix);
        insertPoint->stmts.push_back(f);
        return f;
    }
};

// (>< (+ a b c)) = (+ (>< a) (>< b) (>< c))
struct PushContractionInwardsVisitor : ExprVisitor {
    virtual Expr *visitContract(Contract *c) {
        Add *a = dynamic_cast<Add *>(c->inner);
        if (!a) return c;

        vector<Expr *> inner;
        for (Expr *e : a->inner) {
            inner.push_back(new Contract(c->ix, visitExpr(e)));
        }
        return new Add(inner);
    }
};

bool is_const_zero(const Expr *e) {
    const ConstantInt *i = dynamic_cast<const ConstantInt *>(e);
    if (!i) return false;
    return i->i == 0;
}

bool is_const_one(const Expr *e) {
    const ConstantInt *i = dynamic_cast<const ConstantInt *>(e);
    if (!i) return false;
    return i->i == 1;
}

bool is_all_const_zero(const vector<Expr *> &elist) {
    for (const Expr *e : elist) {
        if (!is_const_zero(e)) return false;
    }
    return true;
}

bool is_any_const_zero(const vector<Expr *> &elist) {
    for (const Expr *e : elist) {
        if (is_const_zero(e)) return true;
    }
    return false;
}

struct ConstantFoldVisitor : ExprVisitor {
    virtual Expr *visitMul(Mul *m) {
        // constant fold multiplication with 0, remove multiplication
        // with 1.
        if (is_any_const_zero(m->inner)) {
            return new ConstantInt(0);
        }

        vector<Expr *> inner;
        for (Expr *e : m->inner) {
            if (is_const_one(e)) continue;
            inner.push_back(visitExpr(e));
        }

        if (inner.size() == 1) {
            return inner[0];
        }
        if (inner.size() == 0) {
            return new ConstantInt(1);
        }

        m = new Mul(inner);

        // Used to expose dirac deltas.
        // convert (* x[i] (- 0 y)) into (* x[i] y -1)
        // this is useful so that we can have
        // (>< i (* x[i] (- 0 delta_i_j)))
        // => (>< i (* x[i] delta_i_j -1))
        // => (* x[j] -1)
        inner.clear();
        for (Expr *e : m->inner) {
            // cout << "inspecting: " << m->to_str() << " | " << e->to_str() <<
            // "\n";
            if (auto *s = dynamic_cast<Sub *>(e)) {
                if (is_const_zero(s->l)) {
                    inner.push_back(new ConstantInt(-1));
                    inner.push_back(s->r);
                    continue;
                }
            }

            inner.push_back(e);
        }

        if (inner.size() == 1) {
            return inner[0];
        }
        if (inner.size() == 0) {
            return new ConstantInt(1);
        }

        m = new Mul(inner);
        return m;
    }

    virtual Expr *visitAdd(Add *a) {
        vector<Expr *> inner;
        for (Expr *e : a->inner) {
            if (is_const_zero(e)) continue;
            inner.push_back(visitExpr(e));
        }

        if (inner.size() == 1) return inner[0];
        if (inner.size() == 0) {
            return new ConstantInt(0);
        }

        return new Add(inner);
    }

    virtual Expr *visitSub(Sub *b) {
        if (is_const_zero(b->r)) {
            return visitExpr(b->l);
        };

        // convert (- 0 x) into (* (-1) x)
        // Used to convert
        //(>< i (- 0 (* (δ i->j) v[i]))
        //=> (>< i (* -1 (δ i j) v[i]))
        //=> (* -1 v[j])
        if (is_const_zero(b->l)) {
            if (Mul *m = dynamic_cast<Mul *>(b->r)) {
                vector<Expr *> inner = m->inner;
                inner.push_back(new ConstantInt(-1));
                return visitExpr(new Mul(inner));
            }
        }

        return ExprVisitor::visitSub(b);
    }

    virtual Expr *visitContract(Contract *c) {
        if (is_const_zero(c->inner)) return new ConstantInt(0);
        return ExprVisitor::visitContract(c);
    }
};

// return a Delta that is within es, which replaces the
// the index ix. return -1 otherwise.
int findDeltaForIndex(Arr ix, vector<Expr *> es) {
    for (int i = 0; i < (int)es.size(); ++i) {
        Delta *d = dynamic_cast<Delta *>(es[i]);
        if (d && d->old == ix) return i;
    }
    return -1;
}

// 1. eliminate (>< i (* t1 t2 (δ i->j) t3)) with
//     (* t1[i->j] t2[i->j] t3[i->j])
//
// 2. eliminate (>< i (δ i->j)) with (1)
struct EliminateContractionVisitor : public ExprVisitor {
    virtual Expr *visitContract(Contract *c) {
        // 2. eliminate (>< i (δ i->j)) with (1)
        if (Delta *d = dynamic_cast<Delta *>(c->inner)) {
            if (d->old == c->ix) {
                return new ConstantInt(1);
            }
        };

        // 1. eliminate products of dirac deltas
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
        for (Expr *e : m->inner) {
            if (e == d) continue;
            substInner.push_back(
                    ExprVisitor::visitExpr(e->subst(c->ix, d->new_)));
        }
        return new Mul(substInner);
    };
};

// convert (+ (+ a b) (+ c d)) to (+ a b c d) and similary
// for multiplication
struct FlattenVisitor : public ExprVisitor {
    virtual Expr *visitMul(Mul *m) {
        vector<Expr *> inner;
        for (Expr *e : m->inner) {
            Mul *me = dynamic_cast<Mul *>(e);
            if (!me) {
                inner.push_back(visitExpr(e));
            } else {
                // copy everything from the inner multiplication
                inner.insert(inner.end(), me->inner.begin(), me->inner.end());
            }
        }
        return new Mul(inner);
    }

    virtual Expr *visitAdd(Add *a) {
        vector<Expr *> inner;
        for (Expr *e : a->inner) {
            Add *ae = dynamic_cast<Add *>(e);
            if (!ae) {
                inner.push_back(visitExpr(e));
            } else {
                // copy everything from the inner multiplication
                inner.insert(inner.end(), ae->inner.begin(), ae->inner.end());
            }
        }
        return new Add(inner);
    }
};

Expr *simplify(Expr *e, bool debug) {
PushContractionInwardsVisitor pushv;
ConstantFoldVisitor cfv;
EliminateContractionVisitor ecv;
FlattenVisitor fv;
for (int i = 0; i < 100; ++i) {
if (debug) {
cout << "--\n";
cout << i << "|" << e->to_str() << "\n";
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

Expr *Expr::normalize() { return simplify(this, false); };

struct IndexGatherVisitor : public ExprVisitor {
    vector<Index *> indexes;

    virtual Expr *visitIndex(Index *i) {
        indexes.push_back(i);
        return ExprVisitor::visitIndex(i);
    }
};

struct ArrayGatherVisitor : public ExprVisitor {
    set<Arr> arrays;
    Expr *visitIndex(Index *i) {
        arrays.insert(i->arr);
        return ExprVisitor::visitIndex(i);
    }
};

map<Arr, set<pair<int, Index *>>> verifyIndexing(Expr *e) {
IndexGatherVisitor iv;
iv.visitExpr(e);

// a map from indeces to index expression
// eg. A[i] + B[j] + C[i, j]
//  i -> { A[i], C[i, j] } | j -> { B[j] + C[i, j] }
map<Arr, set<pair<int, Index *>>> index2indexes;

for (Index *ixe : iv.indexes) {
for (int i = 0; i < (int)ixe->ixs.size(); ++i) {
Arr ix = ixe->ixs[i];
index2indexes[ix].insert(make_pair(ixe->arr.sh[i], ixe));
}
}

for (auto it : index2indexes) {
set<pair<int, Index *>> equivclass = it.second;
assert(equivclass.size() > 0 && "cannot have empty equivalence class");

int refsize;
Index *refix;

std::tie(refsize, refix) = *equivclass.begin();

for (pair<int, Index *> sizeIx : equivclass) {
if (sizeIx.first == refsize) continue;

// error! found indeces that don't match
cerr << "*** ERROR: inconsistent use of index |" << it.first.name
<< "| with respect to sizes of arrays\n";
cerr << "  - size: |" << refsize << "| "
<< "ix: |" << refix->to_str_with_shape() << "|\n";
cerr << "  - size: |" << sizeIx.first << "| "
<< "ix: |" << sizeIx.second->to_str_with_shape() << "|\n";
cerr << "  - expr: |" << e->to_str_with_shape() << "| \n";
assert(false && "inconsistent use of index for sizes");
}
}

return index2indexes;
}


#endif //LANGUAGEMODELS_LANG_H
