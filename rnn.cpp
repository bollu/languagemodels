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

int last_word_index = 0;
set<string> vocab;
map<string, int> word2ix;
map<int, string> ix2word;

// end of sentence is denoted by -1
int num_sentences = 0;
static const long MAX_SENTENCES = 1e5;
static const long MAX_SENTENCE_LEN = 1e2;
int ss[MAX_SENTENCES][MAX_SENTENCE_LEN];
vector<vector<int>> sentences;

struct Shape {
    static const int MAXDIM = 10;
    int ndim = -42;
    DimSize vals[Shape::MAXDIM] = {-42};
    ;

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

void add_word_to_vocab(string w) {
    if (vocab.find(w) != vocab.end()) {
        return;
    }
    word2ix[w] = last_word_index;
    ix2word[last_word_index] = w;
    last_word_index++;
}

bool is_char_word_break(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '.';
}

// disable LSAN. Only keep ASAN. I leak far too much memory ;)
extern "C" int __lsan_is_turned_off() { return 1; }

// consume whitespace and get the next word, or empty string if run out.
string parse_word(FILE *f) {
    char c;
    do {
        c = fgetc(f);
    } while (c != EOF && is_char_word_break(c));

    if (c == EOF) {
        return "";
    }
    ungetc(c, f);

    string w;
    while (1) {
        c = fgetc(f);
        if (is_char_word_break(c)) break;
        w += c;
    }

    return w;
}

bool consume_till_sentence_end_or_word(FILE *f) {
    while (1) {
        char c = fgetc(f);
        if (c == '.') {
            return true;
        }
        if (c == ' ' || c == '\t' || c == '\n') {
            continue;
        }
        ungetc(c, f);
        return false;
    }
}

vector<int> parse_sentence(FILE *f) {
    vector<int> sentence;
    while (1) {
        // returns true if '.' was consumed
        if (consume_till_sentence_end_or_word(f)) {
            return sentence;
        }
        string w = parse_word(f);
        if (w == "") {
            return sentence;
        }
        auto it = word2ix.find(w);
        assert(it != word2ix.end());
        sentence.push_back(it->second);
    }
}

struct CodegenC {
    string program(Program &p) {
        set<Arr> allArrays;
        set<Arr> boundArrays;

        for (Assign a : p.assigns()) {
            boundArrays.insert(a.lhs->arr);
            ArrayGatherVisitor agv;
            agv.visitExpr(a.rhs);
            allArrays.insert(agv.arrays.begin(), agv.arrays.end());
        }

        set<Arr> freeArrays = subtract(allArrays, boundArrays);

        // generate free arrays as parameters
        string out;
        out += "void f(";
        int i = 0;
        for (Arr a : freeArrays) {
            out += (a.ty == NumType::Float ? "float" : "int");
            out += "* ";
            out += a.name;
            if (i < (int)freeArrays.size() - 1) {
                out += ", ";
            }
            ++i;
        }

        out += ") {\n";

        // start code generation of statements
        stmt(&p.stmts, out, 0);

        out += "}";

        return out;
    }

    void stmt(Stmt *s, string &out, int depth) {
        if (Assign *a = dynamic_cast<Assign *>(s)) {
            switch (a->type) {
                // ref = val
                case AssignType::Reference:
                    out += string(depth, ' ');
                    index(a->lhs, out, depth);
                    out += " = ";
                    expr(a->rhs, out, depth);
                    out += ";";
                    break;
                case AssignType::Copy:
                    out += string(depth, ' ');
                    out += "*";
                    index(a->lhs, out, depth);
                    out += " = ";
                    expr(a->rhs, out, depth);
                    out += ";";
                    break;
                case AssignType::Incr:
                    out += string(depth, ' ');
                    out += "*";
                    index(a->lhs, out, depth);
                    out += " += ";
                    expr(a->rhs, out, depth);
                    out += ";";
                    break;
            }
        } else if (Forall *fa = dynamic_cast<Forall *>(s)) {
            // we need to know the array limits
            for (Arr a : fa->ixs) {
                depth += 2;
                out += string(depth, ' ');
                out += "for (";
                out += "int " + a.name + " = 0; ";
                const int TODO_THRESOLD = 42;
                out += a.name + " < " + to_string(TODO_THRESOLD) + "; ";
                out += a.name + "+= 1";
                out += ") {\n";
            }

            stmt(&fa->inner, out, depth);

            for (Arr a : fa->ixs) {
                out += "\n" + string(depth, ' ') + "} /*end " + a.name + "*/";
                depth -= 2;
            }

        } else if (Block *b = dynamic_cast<Block *>(s)) {
            for (Stmt *bs : b->stmts) {
                stmt(bs, out, depth + 2);
                out += "\n";
            }
        } else {
            assert(false && "unknown stmt type.");
        }
    }

    void expr(Expr *e, string &out, int depth) {
        if (Index *ix = dynamic_cast<Index *>(e)) {
            out += "(*";
            index(ix, out, depth);
            out += ")";
        } else if (Mul *m = dynamic_cast<Mul *>(e)) {
            out += "(1";

            for (Expr *me : m->inner) {
                out += "*";
                expr(me, out, depth);
            }
            out += ")";
        } else if (Sub *s = dynamic_cast<Sub *>(e)) {
            out += "(";
            expr(s->l, out, depth);
            out += " - ";
            expr(s->r, out, depth);
            out += ")";
        } else if (ConstantInt *i = dynamic_cast<ConstantInt *>(e)) {
            out += to_string(i->i);
        } else if (ConstantFloat *f = dynamic_cast<ConstantFloat *>(e)) {
            out += to_string(f->f);
        } else if (Contract *c = dynamic_cast<Contract *>(e)) {
            // code generate contraction
            // we can codegen a _function_ that computes the contraction.
            // god I want a monad.
            (void)c;
            assert(false && "unimplemented codegen for Contract");
        } else {
            cerr << "\n\tunknown expr: " << e->to_str() << "\n" << std::flush;
            assert(false && "unknown expression type");
        }
    }

    // generate (arr + delta)
    void index(Index *ix, string &out, int depth) {
        out += "(";
        out += ix->arr.name;
        // now, build up the index expression
        int stride = 1;
        for (int i = 0; i < (int)ix->ixs.size(); ++i) {
            out += "+";
            out += "(";
            out += to_string(stride);
            out += "*";
            out += ix->ixs[i].name;
            out += ")";
            stride *= ix->arr.sh[i];
        }
        out += ")";
    }
};

void test_expr_dot() {
    int NDIMS = 3;
    Arr a("a", NDIMS);
    Arr b("b", NDIMS);
    Arr i("i");
    Expr *mul = new Mul(a.ix(i), b.ix(i));
    Expr *dot = new Contract(i, mul);

    cout << "******dot******\n";
    cout << dot->to_str_with_shape() << "\n";
    Arr k("k");
    Expr *grad = dot->grad("a", {k});
    cout << "## dot->grad[a k]: ##\n\t" << grad->to_str_with_shape() << "\n";
    cout << "## dot->grad[a k](normalized): ##\n\t"
         << grad->normalize()->to_str_with_shape() << "\n";
}

void test_expr_matvec() {
    const int NOUT = 3;
    const int NIN = 3;
    Arr m("m", NOUT, NIN);
    Arr v("v", NIN);
    Arr i("i"), j("j");
    Expr *mul = new Mul(m.ix(i, j), v.ix(j));
    Expr *matvec = new Contract(j, mul);

    cout << "******Mat @ vec******\n";
    cout << matvec->to_str_with_shape() << "\n";

    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *gradm = matvec->grad("m", {Arr("d0"), Arr("d1")});
    cout << "## mul->grad[a d0 d1]: ##\n\t" << gradm->to_str() << "\n";
    cout << "## mul->grad[a d0 d1](normalized): ##\n\t"
         << gradm->normalize()->to_str_with_shape() << "\n";

    Expr *gradv = matvec->grad("v", {Arr("d0")});
    cout << "## mul->grad[v d0]: ##\n\t" << gradv->to_str_with_shape() << "\n";
    cout << "## mul->grad[v d0](normalized): ##\n\t"
         << gradv->normalize()->to_str_with_shape() << "\n";
}

void test_expr_tanh() {
    const int NDIM = 3;
    Arr a("a", NDIM, NDIM);
    Arr i("i");
    Arr j("j");
    Expr *aixed = a.ix(i, j);
    Expr *matvec = new Tanh(aixed);

    cout << "*****tanh on matrix*****\n";
    cout << matvec->to_str_with_shape() << "\n";

    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *grada = matvec->grad("a", {Arr("k"), Arr("l")});
    cout << "## mul->grad[a k l]: ##\n\t" << grada->to_str_with_shape() << "\n";
    cout << "## mul->grad[a k l](normalized): ##\n\t"
         << grada->normalize()->to_str_with_shape() << "\n";
}

// construct Dot using program
void test_program_dot() {
    Program p;
    IRBuilder builder(p);

    static const int ARRSIZE = 10;
    Arr a("a", ARRSIZE);
    Arr b("b", ARRSIZE);
    Arr grada("grada", ARRSIZE);
    Arr gradb("gradb", ARRSIZE);
    Arr dot("dot");
    Arr i("i");
    Arr k("k");
    builder.copy(dot.ix(), new Contract(i, new Mul(a.ix(i), b.ix(i))));
    builder.copy(grada.ix(k), new Mul(new ConstantFloat(1e-2),
                                      p[dot].rhs->grad("a", {k})->normalize()));
    builder.copy(gradb.ix(k), new Mul(new ConstantFloat(1e-2),
                                      p[dot].rhs->grad("b", {k})->normalize()));

    // Incr is weird, since we should probably be the ones doing the
    // indexing...
    builder.incr(a.ix(k), grada.ix(k));
    builder.incr(b.ix(k), gradb.ix(k));

    cout << "*****program dot*****\n";
    cout << p.to_str();
}

void test_program_dot_batched() {
    static const int BATCHSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr focuses("focuses", BATCHSIZE, EMBEDSIZE);
    Arr ctxes("ctxes", BATCHSIZE, EMBEDSIZE);
    Arr total_loss("total_loss");

    Arr bi("bi");
    Arr i("i");

    Program p;

    IRBuilder builder(p);
    Forall *forbi = builder.insertFor({bi});

    builder.setInsertPoint(forbi);
    Expr *dots = new Contract(i, new Mul(focuses.ix(bi, i), ctxes.ix(bi, i)));
    Expr *losses = new Sub(new ConstantInt(1), dots);

    builder.copy(total_loss.ix(), new Contract(bi, losses));

    Expr *lr = new ConstantFloat(1e-2);

    Arr dbi("dbi"), dk("dk");
    builder.incr(
        focuses.ix(dbi, dk),
        new Mul(lr,
                p[total_loss].rhs->grad("focuses", {dbi, dk})->normalize()));
    builder.incr(
        ctxes.ix(dbi, dk),
        new Mul(lr, p[total_loss].rhs->grad("ctxes", {dbi, dk})->normalize()));

    cout << "*****program batched dot*****\n";
    cout << p.to_str() << "\n";
}

void test_program_dot_indicrect() {
    cout << "*****program indirect dot*****\n";
    static const int VOCABSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr embeds("embeds", VOCABSIZE, EMBEDSIZE);
    Arr focusix("focusix");
    Arr ctxix("ctxix");
    Arr focus("focus", EMBEDSIZE);
    Arr ctx("ctx", EMBEDSIZE);
    Arr dot("dot");
    Arr loss("loss");

    Arr i("i"), k("k");
    Program p;

    IRBuilder builder(p);

    builder.reference(focus.ix(i), embeds.ix(focusix, i));
    builder.reference(ctx.ix(i), embeds.ix(ctxix, i));

    builder.copy(dot.ix(), new Contract(i, new Mul(focus.ix(i), ctx.ix(i))));
    builder.copy(loss.ix(), new Sub(new ConstantInt(1), p[dot].rhs));

    Expr *lr = new ConstantFloat(1e-2);

    builder.incr(focus.ix(k),
                 new Mul(lr, p[loss].rhs->grad("focus", {k})->normalize()));
    builder.incr(ctx.ix(k),
                 new Mul(lr, p[loss].rhs->grad("ctx", {k})->normalize()));
    cout << p.to_str();
}

Expr *matvecmul(Arr m, Arr v, Arr ix) {
    Arr c("c");
    return new Contract(c, new Mul(m.ix(ix, c), v.ix(c)));
}

void test_program_dot_batched_indirect() {
    cout << "*****program batched, indirect addressed dot (final code that we "
            "need for word embeddings)*****\n";
    static const int BATCHSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr focuses("focuses", BATCHSIZE, EMBEDSIZE);
    Arr ctxes("ctxes", BATCHSIZE, EMBEDSIZE);
    Arr focus("focus");
    Arr ctx("ctx");
    Arr dot("dot");
    Arr loss("loss");

    Arr bi("bi");
    Arr i("i");
    // Arr *k = new Arr("k");
    Program p;

    IRBuilder builder(p);
    // builder.setInsertPoint(builder.insertFor(bi));
    Forall *forbi = builder.insertFor({bi, i});
    builder.setInsertPoint(forbi);

    builder.reference(focus.ix(), focuses.ix(bi, i));
    builder.reference(ctx.ix(), ctxes.ix(bi, i));

    builder.incr(dot.ix(), new Mul(focus.ix(), ctx.ix()));

    builder.setInsertPoint(p);
    forbi = builder.insertFor({bi, i});
    builder.setInsertPoint(forbi);
    builder.copy(loss.ix(), new Sub(new ConstantInt(1), p[dot].rhs));

    Expr *lr = new ConstantFloat(1e-2);

    Arr focusl("focusl");
    Arr ctxl("ctxl");
    builder.reference(focusl.ix(), focuses.ix(bi, i));
    builder.reference(ctxl.ix(), ctxes.ix(bi, i));
    builder.incr(focusl.ix(),
                 new Mul(lr, p[loss].rhs->grad("focus", {})->normalize()));
    builder.incr(ctxl.ix(),
                 new Mul(lr, p[loss].rhs->grad("ctx", {})->normalize()));
    cout << p.to_str();

    cout << "*****Codegened word embddings code:*****\n";
    CodegenC cc;
    cout << cc.program(p) << "\n";
}

Expr *l2(Arr v, Arr w) {
    Arr c("c");
    Expr *s = new Sub(v.ix(c), w.ix(c));
    return new Contract(c, new Mul(s, s));
}

// compute dy/dx

// get all paths from a source node to a target node, _backwards_.
// That is, expressions:
// x1 := x0 + 1
// x2 := x1 + 1
// x3 := x2 + 3
// with the call
// getPathsToArr(p, x3, x0, {x3}) will give
// - x3 x2 x1 x0
set<std::vector<Arr>> getPathsToArr(Program &p, Arr cur, Arr target,
                                    set<vector<Arr>> pathsToCur) {
    if (cur == target) {
        return pathsToCur;
    }

    ArrayGatherVisitor agv;

    // this array does not exist in the program, so just return.
    if (!p.is_array_assigned(cur)) {
        return {};
    }
    Expr *rhs = p[cur].rhs;

    // cout << "rhs for " << cur->name << " : " << (rhs ? rhs->to_str() :
    // "NULL") << "\n";
    if (!rhs) {
        return {};
    }

    // look in the RHS of this array
    agv.visitExpr(rhs);

    // cout << "arrays for |" << cur->name << "| := |"  << rhs->to_str() << "|";
    // for(Arr *a : agv.arrays) cout << a->name << " ";
    // cout << "|\n";

    set<vector<Arr>> ps;
    for (Arr child : agv.arrays) {
        set<vector<Arr>> pathsToChild;
        for (vector<Arr> path : pathsToCur) {
            path.push_back(child);
            pathsToChild.insert(path);
        }
        set<vector<Arr>> pathsToTarget =
            getPathsToArr(p, child, target, pathsToChild);
        ps.insert(pathsToTarget.begin(), pathsToTarget.end());
    }
    return ps;
}

// if array is [y intermediate x], compute:
// (>< i0 i1 i2  (* dy[outixs]/dintermediate[i0, i1, i2]  dindermediate[i0, i1,
// i2]/dx[inixs]))
Expr *codegenDerivativeChain(Program &p, vector<Arr> arrs, vector<Arr> inixs,
                             vector<Arr> outixs) {
    const int n = arrs.size();
    assert(arrs.size() >= 2);
    map<Arr, vector<Arr>> arr2ix;
    map<Arr, Expr *> arr2expr;
    arr2ix[arrs[0]] = outixs;
    arr2ix[arrs[n - 1]] = inixs;

    arr2expr[arrs[0]] = p[arrs[0]].rhs;

    for (int i = 1; i < n - 1; ++i) {
        struct Arr a = arrs[i];

        vector<Arr> aixs;

        for (int i = 0; i < a.sh.ndim; ++i) {
            aixs.push_back(Arr(a.name + "_" + to_string(i)));
        }
        arr2ix[arrs[i]] = aixs;

        Index *ix = p[arrs[i]].lhs;
        Expr *e = p[arrs[i]].rhs;
        // reindex
        for (int i = 0; i < a.sh.ndim; ++i) {
            e = e->subst(ix->ixs[i], aixs[i]);
        }

        arr2expr[arrs[i]] = e;
    }

    IRBuilder builder(p);
    Expr *chain = nullptr;
    vector<Arr> contractIxs;
    for (int i = 0; i < n - 1; ++i) {
        struct Arr cur = arrs[i];
        struct Arr next = arrs[i + 1];

        vector<Arr> ixcur = arr2ix[cur];
        vector<Arr> ixnext = arr2ix[next];
        vector<Arr> allixs;
        allixs.insert(allixs.end(), ixcur.begin(), ixcur.end());
        allixs.insert(allixs.end(), ixnext.begin(), ixnext.end());

        Expr *curval = arr2expr[cur];
        assert(curval);

        Expr *grad = curval->grad(next.name, ixnext)->normalize();
        Arr dcur_dnext("d" + cur.name + "_" + "d" + next.name,
                       cur.sh.appendInside(next.sh));
        builder.copy(dcur_dnext.ix(allixs), grad);

        if (!chain) {
            chain = dcur_dnext.ix(allixs);
        } else {
            cout << "prev chain: " << chain->to_str() << "\n";

            chain = new Mul(chain, dcur_dnext.ix(allixs));
            for (const Arr c : ixcur) {
                cout << "\tcontract " << chain->to_str() << "\n";
                chain = new Contract(c, chain);
            }
        }
        contractIxs = ixnext;
    }

    return chain;
}

// compute dy/dx, through any chain of expressions needed.
// TODO: keep a map of dy/dx -> array holding this value

Expr *takeIndirectDerivatives(Program &p, Arr y, Arr x, vector<Arr> inixs,
                              vector<Arr> outixs) {
    IRBuilder builder(p);
    set<vector<Arr>> yTox = getPathsToArr(p, y, x, {{y}});

    Expr *allsum = new ConstantInt(0);
    for (vector<Arr> path : yTox) {
        cout << "\n- path: ";
        for (Arr a : path) {
            cout << a.to_str() << " ";
        }
        allsum =
            new Add(allsum, codegenDerivativeChain(p, path, inixs, outixs));
    }
    return allsum;
}

// map an array to all uses of the array
map<Arr, set<Assign>> getUses(Program p) {
    map<Arr, set<Assign>> a2uses;
    for (Assign assign : p.assigns()) {
        ArrayGatherVisitor agv;
        agv.visitExpr(assign.rhs);
        for (Arr used : agv.arrays) {
            // map the used array on the RHS to the user on the LHS
            a2uses[used].insert(assign);
        }
    }

    return a2uses;
}

// map an array to all arrays used to compute this array
map<Arr, set<Arr>> getDependences(Program p) {
    map<Arr, set<Arr>> a2deps;
    for (Assign assign : p.assigns()) {
        ArrayGatherVisitor agv;
        agv.visitExpr(assign.rhs);
        for (Arr in : agv.arrays) {
            a2deps[assign.lhs->arr].insert(in);
        }
    }

    return a2deps;
}

// compute ds/dz for all variables reachable from $z$. Returns a mapping
// from each array to expressions
map<Arr, Arr> reverseDiff(Program &p, Arr z) {
    IRBuilder builder(p);
    assert(z.sh.ndim == 0);
    map<Arr, set<Arr>> deps = getDependences(p);

    cout << "=================================================\n";
    for (auto it : deps) {
        cout << it.first.name << " -> ";
        cout << "{ ";
        for (auto a : it.second) {
            cout << a.name << " ";
        }
        cout << "}";
        cout << "\n";
    }
    cout << "=================================================\n";

    map<Arr, Arr> dz_darr;
    Arr dz_dz("dz_dz");
    builder.copy(dz_dz.ix(), new ConstantInt(1));
    dz_darr[z] = dz_dz;

    queue<Arr> horizon;
    horizon.push(z);

    while (!horizon.empty()) {
        Arr o = horizon.front();
        horizon.pop();
        cout << "analyzing uses of: |" << o.name << "| \n";

        // array is not differentiable (not part of program)
        if (!p.is_array_assigned(o)) {
            continue;
        }

        // dz/do
        Assign oassign = p[o];
        Expr *oval = oassign.rhs;
        vector<Arr> oixs = oassign.lhs->ixs;

        assert(dz_darr.count(o));
        // array that hold derivative of ao wrt dz
        Arr dz_do = dz_darr.find(o)->second;

        // write out into every input
        // dz_di += dz_do x do_di [where o = f(i, ...) do_di = df_di]
        for (Arr i : deps[o]) {
            cout << "\tanalyzing |" << o.name << "| = f(" << i.name << ")\n";

            // needs to be initialized
            Arr dz_di;
            if (dz_darr.count(i)) {
                dz_di = dz_darr.find(i)->second;
            } else {
                dz_di = Arr("d" + z.name + "_d" + i.name, i.sh);
                dz_darr.insert(std::make_pair(i, dz_di));
                // TODO: write zero.
            }

            vector<Arr> iixs;
            for (int ix = 0; ix < i.sh.ndim; ++ix) {
                iixs.push_back(Arr(i.name + "_" + std::to_string(ix)));
            }

            // do/di
            Expr *do_di = oval->grad(i.name, iixs)->normalize();
            (void)(do_di);

            do_di = do_di->normalize();
            cout << "contracting along: ";
            for (Arr i : iixs) {
                cout << i.name << " ";
            }
            cout << "\n";
            cout << "  - dz/do:" << dz_do.to_str() << "\n";
            cout << "  - do/di:" << do_di->normalize()->to_str_with_shape()
                 << "\n";

            // write into dz_di
            builder.incr(
                dz_di.ix(iixs),
                Contract::contract(oixs, new Mul(dz_do.ix(oixs), do_di))
                    ->normalize());

            horizon.push(i);
        }
    }

    return dz_darr;
}

Expr *cell(Program &p, Arr I2H, Arr H2H, Arr i, Arr h, Arr I2Hi, Arr H2Hh,
           Arr ix) {
    IRBuilder builder(p);
    builder.copy(H2Hh.ix(ix), matvecmul(H2H, h, ix));
    builder.copy(I2Hi.ix(ix), matvecmul(I2H, i, ix));
    return new Tanh(new Add(I2Hi.ix(ix), H2Hh.ix(ix)));
}

void test_lstm() {
    // size of input/output arrays
    static const int IOSIZE = 5;
    static const int HIDDENSIZE = 10;
    static const int NINPUTS = 1;

    Arr inputs[NINPUTS];
    Arr I2Hi[NINPUTS];
    Arr hiddens[NINPUTS + 1];
    Arr H2Hh[NINPUTS];

    for (int i = 0; i < NINPUTS; ++i) {
        inputs[i] = Arr("i" + to_string(i), IOSIZE);
        I2Hi[i] = Arr("I2Hi" + to_string(i), HIDDENSIZE);
    }

    (void)(inputs);

    for (int i = 0; i < NINPUTS + 1; ++i) {
        hiddens[i] = Arr("h" + to_string(i), HIDDENSIZE);
    }

    for (int i = 0; i < NINPUTS; ++i) {
        H2Hh[i] = Arr("H2Hh" + to_string(i), HIDDENSIZE);
    }

    Arr ix("ix");
    Arr I2H("I2H", HIDDENSIZE, IOSIZE);
    Arr H2H("H2H", HIDDENSIZE, HIDDENSIZE);
    Arr H2O("H2O", IOSIZE, HIDDENSIZE);

    Program p;
    IRBuilder builder(p);

    Arr hprev = hiddens[0];
    for (int i = 0; i < NINPUTS; ++i) {
        Expr *c = cell(p, I2H, H2H, inputs[i], hprev, I2Hi[i], H2Hh[i], ix);
        cout << "program:\n" << p.to_str() << "\n";
        builder.copy(hiddens[i + 1].ix(ix), c);
        hprev = hiddens[i + 1];
    }

    Arr predict("p", IOSIZE);
    builder.copy(predict.ix(ix), matvecmul(H2O, hprev, ix));

    Arr output("o", IOSIZE);
    Arr loss("l");
    builder.copy(loss.ix(), l2(output, predict));

    cout << "*****LSTM*****:\n";
    cout << p.to_str();

    Arr i("i"), j("j");
    map<Arr, Arr> arr2der = reverseDiff(p, loss);

    // Arr *dH2H = new Arr("dH2H", HIDDENSIZE, HIDDENSIZE);
    // builder.incr(H2H.ix(i, j), der);

    cout << "*****LSTM(full)*****:\n";
    cout << p.to_str();

    CodegenC cc;
    cout << "******LSTM(codegen)******\n" << cc.program(p) << "\n";
}

int main(int argc, char **argv) {
    test_expr_dot();
    test_expr_matvec();
    test_expr_tanh();
    test_program_dot();
    test_program_dot_batched();
    test_program_dot_indicrect();
    test_program_dot_batched_indirect();
    test_lstm();
    // return 0;

    if (argc != 2) {
        cout << "usage: <input-path>\n";
        return 1;
    }
    FILE *f = fopen(argv[1], "r");

    // construct vocab
    while (1) {
        string w = parse_word(f);
        cout << "found word: " << w << "\n";
        if (feof(f)) {
            break;
        }
        add_word_to_vocab(w);
    }
    rewind(f);
    while (1) {
        vector<int> sentence = parse_sentence(f);
        sentences.push_back(sentence);
        if (feof(f)) {
            break;
        }
    }
    fclose(f);

    for (auto it : word2ix) {
        cout << "|" << it.first << "|->" << it.second << "\n";
    }

    cout << "SENTENCES:\n";
    for (auto s : sentences) {
        for (auto w : s) {
            cout << w << " ";
        }
        cout << "\n";
    }
}
