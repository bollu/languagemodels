#include <iostream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <map>
#include <list>
#include <stack>
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
        string s =  "<";
        for(int i = 0; i < ndim; ++i) {
            s += to_string(vals[i]) + (i < ndim - 1 ? " " : "");
        }
        s += ">";
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

struct Expr {
    // gives type of expression
    ExprType ty;
    Expr(ExprType ty): ty(ty) {};

    Expr *contract(const Arr* ix);

    // print as string
    virtual string to_str() const = 0;

    // return free indeces
    virtual set<const Arr *> free() const = 0;

    // substitute old indeces for new indeces.
    virtual Expr *subst(const Arr* old, const Arr* new_) const = 0;

    // creates dirac deltas by taking gradients
    virtual Expr *grad_(string name, vector<const Arr *> ixs) = 0;

    Expr *grad(string name, vector<const Arr *> ixs);

    string to_str_with_shape() const;

    // simplify the expression to normal form
    Expr *normalize();

};


struct ConstantInt : public Expr {
    int i;
    ConstantInt(int i) : Expr(ExprType::ConstantInt), i(i) {};

    string to_str() const { return std::to_string(i); }
    set<const Arr *> free() const { return {}; }
    Expr *grad_(string name, vector<const Arr *> ixs) { 
        return new ConstantInt(0);
    }

    Expr *subst(const Arr* old, const Arr* new_) const {
        return new ConstantInt(i);
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

    string to_str() const { return name + (sh.ndim > 0 ? sh.to_str() : ""); };

    set<const Arr *> free() const {
        return set<const Arr *>();
    }

    Expr *grad_(string garr, vector<const Arr *> ixs) {
        // if we ever get here, then we should be 0-dimensional
        assert(ixs.size() == 0);
        if (garr == name) { return new ConstantInt(1); };
        return new ConstantInt(0);
    }

    // do we not need to substitute the 1D array?
    Expr *subst(const Arr *old, const Arr *new_) const {
        // if we ever get here, then we should be 0-dimensional
        if (name == old->name) return new Arr(new_->name);
        return new Arr(name);
        //return new Arr(name);
    }

    Shape shape() const { return sh; }

    // helpers to index
    Expr *ix(vector<const Arr *> ix);
    Expr *ix(const Arr *ix1);
    Expr *ix(const Arr *ix, const Arr *ix2);

};

string Expr::to_str_with_shape() const {
    string s = "";
    s += "[";
    for(const Arr *i : free()) s += i->name + ",";
    s += "]";
    s += to_str();
    return s;
}



Expr *Expr::grad(string name, vector<const Arr *> ixs) {
    set<string> freeNames;
    for (const Arr *f : this->free()) { 
        freeNames.insert(f->name);
    }
    for(const Arr * ix : ixs) {
        if (freeNames.count(ix->name)) {
            cerr << "\nreused name for gradient index: " << ix->name << "\n";
            assert(false && "gradient indexing names must be fresh");
        }
    }

    return grad_(name, ixs);
};



struct ConstantFloat : public Expr {
    float f;
    ConstantFloat(float f) : Expr(ExprType::ConstantFloat), f(f) {};

    string to_str() const { return std::to_string(f); }
    set<const Arr*> free() const { return {}; }
    Expr *grad_(string name, vector<const Arr*> ixs) { 
        return new ConstantInt(0);
    }

    Expr *subst(const Arr* old, const Arr* new_) const {
        return new ConstantFloat(f);
    }

};


struct Delta : public Expr {
    const Arr* old;
    const Arr* new_;

    Delta(const Arr* old, const Arr* new_) : 
        Expr(ExprType::Delta), 
    old(old), new_(new_) { };

    string to_str() const {
        string s = "(δ ";
        s +=  old->to_str() + "->" + new_->to_str();
        s += ")";
        return s;
    }

    virtual set<const Arr*> free() const {
        return {old, new_};
    }

    Expr *grad_(string arr, vector<const Arr*> ixs) {
        return new ConstantInt(0);
    };

    Expr *subst(const Arr* sold, const Arr* snew) const {
        // (i->k) delta_ij : delta_kj
        // (i->k) delta_ji : delta_jk
        // (i->k) delta_ii : delta_kk
        // (i->k) delta_jl : delta_jl
        return new Delta(old == sold ? snew : old,
                    new_ == sold ? snew : new_);
    }


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

    set<const Arr*> free() const {
        set<const Arr*> f;

        for(Expr *i: inner) {
            set<const Arr*> ifree = i->free();
            f.insert(ifree.begin(), ifree.end());
        }
        return f;
    }

    Expr *grad_(string name, vector<const Arr*> ixs) {
        vector<Expr *>dinner;
        for(Expr *e : inner) {
            dinner.push_back(e->grad(name, ixs));
        }
        return new Add(dinner);
    }

    Expr *subst(const Arr* old, const Arr* new_) const {
        vector<Expr *> sinner;
        for(Expr *e : inner) {
            sinner.push_back(e->subst(old, new_));
        }
        return new Add(sinner);
    }
};

struct Sub : public Expr {
    Expr *l, *r;

    Sub (Expr *l, Expr *r) : Expr(ExprType::Sub), l(l), r(r) { }

    string to_str() const {
        return "(- " + l->to_str() + " " + r->to_str() + ")";
    }

    set<const Arr*> free() const {
        set<const Arr*> f;
        set<const Arr*> lf = l->free();
        f.insert(lf.begin(), lf.end());

        set<const Arr*> rf = r->free();
        f.insert(rf.begin(), rf.end());
        return f;
    }

    Expr *grad_(string name, vector<const Arr*> ixs) {
        return new Sub(l->grad(name, ixs), r->grad(name, ixs));
    }

    Expr *subst(const Arr* old, const Arr* new_) const {
        return new Sub(l->subst(old, new_), r->subst(old, new_));
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
        s += ")";
        return s;
    }

    set<const Arr*> free() const {
        set<const Arr*> f;

        for(Expr *i: inner) {
            set<const Arr*> ifree = i->free();
            f.insert(ifree.begin(), ifree.end());
        }
        return f;
    }

    Expr *grad_(string name, vector<const Arr*> ixs) {

        // d(fgh) = df gh + f dg h + fg dh
        vector<Expr *> dsum;

        for(int i = 0; i < (int)inner.size(); ++i) {
            vector<Expr *> dinner_i = this->inner;
            dinner_i[i] = dinner_i[i]->grad(name, ixs);
            dsum.push_back(new Mul(dinner_i));
        }

        return new Add(dsum);
    }

    Expr *subst(const Arr* old, const Arr* new_) const {
        vector<Expr *> sinner;
        for(Expr *e : inner) {
            sinner.push_back(e->subst(old, new_));
        }
        return new Mul(sinner);
    }
};


struct Index : public Expr {
    Arr *arr;
    vector<const Arr*> ixs;

    Index(Arr *arr, vector<const Arr*> ixs): 
        Expr(ExprType::Index), arr(arr), ixs(ixs) {
            for (const Arr *ix : ixs) {
                assert(ix->shape().ndim == 0 &&
                        "slices must be zero-dimensional");
            }

            // ensure that we are fully indexing the array.
            if((int)ixs.size() > arr->shape().ndim) {
                cout << "array " << arr->to_str() << "| shape: " 
                    << arr->shape().to_str() << " | nixs: " << ixs.size()
                    << "\n";
            };
            assert((int)ixs.size() <= arr->shape().ndim);
        }; 

    string to_str() const  {
        string s =  arr->to_str() + "[";

        for(int i = 0; i < (int)ixs.size(); ++i) {
            s += ixs[i]->to_str() + (i < (int)ixs.size() - 1 ? " " : "");
        }

        s += "]";
        return s;
    }

    set<const Arr*> free() const {
        set<const Arr*> s = arr->free();
        s.insert(ixs.begin(), ixs.end());
        return s;
    }

    Expr *grad_(string name, vector<const Arr*> gixs) {
        // can only take gradients if the slice is fully saturated
        assert((int)ixs.size() == arr->shape().ndim);

        // this IS the Index of an array, but not the array we are
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
    virtual Expr *subst(const Arr* old, const Arr* new_) const {
        vector<const Arr*> ixsnew;
        for(int i = 0; i < (int)ixs.size(); ++i) {
            ixsnew.push_back(ixs[i] == old ? new_ : ixs[i]);
        }
        return new Index(arr, ixsnew);
    }

};



Expr *Arr::ix(vector<const Arr*> ixs) {
    return new Index(this, ixs);
}

Expr *Arr::ix(const Arr* ix1) {
    return new Index(this, {ix1});
}

Expr *Arr::ix(const Arr* ix1, const Arr* ix2) {
    return new Index(this, {ix1, ix2});
}



struct Contract : public Expr {
    const Arr* ix;
    Expr *inner;
    Contract (const Arr* ix, Expr *inner) : Expr(ExprType::Contract), ix(ix),
        inner(inner) {};

    string to_str() const {
        return "(>< " + ix->to_str() + " " + inner->to_str() +  ")";
    }

    set<const Arr*> free() const {
        set<const Arr*> infree = inner->free();
        auto it = infree.find(ix);
        if (it != infree.end()) infree.erase(it);
        return infree;
    }

    Expr *grad_(string name, vector<const Arr*> ixs) {
        return new Contract(ix, inner->grad(name, ixs));
    }

    Expr *subst(const Arr* old, const Arr* new_) const {
        return new Contract(ix == old ? new_ : ix, inner);
    }

};


Expr *Expr::contract(const Arr* ix) {
    return new Contract(ix, this);
}

struct Tanh : public Expr {
    Expr *inner;
    Tanh(Expr *inner) : Expr(ExprType::Tanh), inner(inner) {};

    string to_str() const {
        return "(tanh " + inner->to_str() + ")";
    }

    set<const Arr*> free() const {
        return inner->free();
    }


    Expr *subst(const Arr* old, const Arr* new_) const { 
        return new Tanh(inner->subst(old, new_));
    }

    Expr *grad_(string name, vector<const Arr*> ixs) {
        Expr *dtan = new Sub(new ConstantInt(1), new Mul(new Tanh(inner), new
                    Tanh(inner)));
        Expr *dinner = inner->grad(name, ixs);
        return new Mul(dtan, dinner);
    }

};

struct ExprVisitor {
    Expr *visitExpr(Expr *e) {
        if (Add *a = dynamic_cast<Add *>(e)) {
            return visitAdd(a);
        } if (Sub *s = dynamic_cast<Sub *>(e)) {
            return visitSub(s);
        } else if (Mul *m = dynamic_cast<Mul *>(e)) {
            return visitMul(m);
        } else if (Contract *c = dynamic_cast<Contract *>(e)) {
            return visitContract(c);
        } else if (Index *i = dynamic_cast<Index *>(e)) {
            return visitIndex(i);
        } else if (Arr *a = dynamic_cast<Arr *>(e)) {
            return visitArr(a);
        } else if (Tanh *tanh = dynamic_cast<Tanh *>(e)) {
            return visitExpr(tanh->inner);
        } else {
            return e;
        }
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

    virtual Expr *visitSub(Sub *s) { 
        return new Sub(visitExpr(s->l), visitExpr(s->r));
    };

    virtual Expr *visitContract(Contract *c) { 
        Expr *inner = visitExpr(c->inner);
        return new Contract(c->ix, inner);
    }

    virtual Expr *visitArr(Arr *a) {
        return new Arr(a->name, a->sh);
    }

    virtual Expr *visitIndex(Index *i) {
        Expr *e = visitArr(i->arr);
        if (Arr *a = dynamic_cast<Arr *>(e)) {
            return new Index(a, i->ixs);
        }
        return nullptr;
    }
};

enum class AssignType {
    Copy,
    Reference,
    Incr,
};

struct Stmt {
    virtual string to_str(int depth=0) const = 0;

    // return the RHS of the expression if the statement has it.
    // return nullptr otherwise;
    virtual void findArr(Arr *arr, Expr **arrptr)  = 0;
    virtual std::map<Arr *, Expr *> arrays() = 0;
};

struct Assign : public Stmt {
    AssignType type;
    Arr *lhs;
    vector<const Arr *> indeces;
    Expr *rhs;

    Assign(AssignType type, Arr *lhs, vector<const Arr *> indeces, Expr *rhs) 
        : type(type), lhs(lhs), indeces(indeces), rhs(rhs) {

            // TODO: add an exhaustiveness check that the set of indeces
            // is equal to the set of free indeces.

            for (const Arr * ix: indeces) {
                assert(ix->sh.ndim == 0);
            }

            if (type == AssignType::Reference) {
                if (indeces.size() >= rhs->free().size()) {
                    cerr << "Taking a slice of zero or negative dimension\n";
                    cerr << "\tlhs: " << lhs->to_str_with_shape() << "\n";
                    cerr << "\tindeces: ";
                    for (const Arr *a : indeces) cerr << a->name << " ";
                    cerr << "\n\texpression: ";
                    cerr << rhs->to_str_with_shape() << "\n";
                }

                assert(indeces.size() < rhs->free().size());
            } else {
                if (rhs->free().size() != indeces.size()) {
                    cerr << "mismatch between indeces and number of free variables\n";
                    cerr << "\tlhs: " << lhs->to_str_with_shape() << "\n";
                    cerr << "\tindeces: ";
                    for (const Arr *a : indeces) cerr << a->name << " ";
                    cerr << "\n\texpression: ";
                    cerr << rhs->to_str_with_shape() << "\n";
                }

                assert(rhs->free().size() == indeces.size() && 
                        "need those many free variables as free indices");
            }
        };

    string to_str(int depth) const {
        cout << string(depth, ' ');

        string s = "(";
        s += lhs->name;
        s += "[";
        for (const Arr * ix: indeces) {
            s += ix->name + ",";
        }
        s += "] ";

        switch(type) {
            case AssignType::Copy: s += ":="; break;
            case AssignType::Reference: s += "&="; break;
            case AssignType::Incr: s += "+="; break;
        }

        s += " " + rhs->to_str_with_shape() + ")";
        return s;
    }

    virtual void findArr(Arr *arr, Expr **arrptr) {
        assert(arrptr);
        if (arr == lhs) {  *arrptr = rhs; }
        else { *arrptr = nullptr; }
    }

    std::map<Arr *, Expr *> arrays() {
        return { {lhs, rhs}};

    }
};

struct Block : public Stmt {
    list<Stmt *> stmts;

    string to_str(int depth) const {
        string str = "";
        for (Stmt *s: stmts) {
            str += string(depth, ' ') + s->to_str() + "\n";
        }
        return str;
    }

    virtual void findArr(Arr *arr, Expr **arrptr) {
        assert(arrptr);
        *arrptr = nullptr;
        for (auto it = stmts.rbegin(); it != stmts.rend(); ++it) {
            (*it)->findArr(arr, arrptr);
            if (*arrptr) { return; }
        }
    }

    std::map<Arr *, Expr *> arrays() {
        std::map<Arr *, Expr *> m;
        for(Stmt *s : stmts) {
            std::map<Arr *, Expr *> sm = s->arrays();
            m.insert(sm.begin(), sm.end());
        }
        return m;
    }


};

struct Forall : public Stmt {
    const Arr* ix;
    Block inner;

    Forall (const Arr* ix) : ix(ix) {};

    string to_str(int depth) const {
        string s = "";
        s += "forall " + ix->name + " {\n";
        s += inner.to_str(depth + 1);
        s += "\n" + string(' ', depth) + "}";
        return s;
    }

    virtual void findArr(Arr *arr, Expr **arrptr) {
        return inner.findArr(arr, arrptr);
    }

    std::map<Arr *, Expr *> arrays() {
        return inner.arrays();
    }
};

struct Program {
    Block stmts;

    string to_str() {
        return stmts.to_str(0);
    }

    Expr *operator [](Arr *arr) {
        Expr *arrptr;
        stmts.findArr(arr, &arrptr);
        return arrptr;
    }

    std::map<Arr *, Expr *> arrays() {
        return stmts.arrays();
    }
};

// welcome LLVM my old friend
struct IRBuilder {
    Program &p;
    Block *insertPoint;

    IRBuilder(Program &p) : p(p), insertPoint(&p.stmts) {};

    void setInsertPoint(Program &p) {
        insertPoint = &p.stmts;
    }

    void setInsertPoint(Forall &f) {
        insertPoint = &f.inner;
    }

    void setInsertPoint(Forall *f) {
        insertPoint = &f->inner;
    }

    void copy(Arr *arr, vector<const Arr *> indeces, Expr *rhs) {
        assert(arr != nullptr);
        insertPoint->stmts.push_back(new Assign(AssignType::Copy, arr, indeces,
                    rhs));
    }

    void incr(Arr *arr, vector<const Arr *> indeces, Expr *rhs) {
        assert(arr != nullptr && "creating an array");
        insertPoint->stmts.push_back(new Assign(AssignType::Incr, arr, indeces,
                    rhs));
    }


    void reference(Arr *arr, vector<const Arr *> indeces, Expr *rhs) {
        assert(arr != nullptr && "creating an array");
        insertPoint->stmts.push_back(new Assign(AssignType::Reference, arr,
                    indeces, rhs));
    }

    Forall *insertFor(const Arr* ix) {
        Forall *f = new Forall(ix);
        insertPoint->stmts.push_back(f);
        return f;
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
        if (inner.size() == 0) { return new ConstantInt(1); }


        m = new Mul(inner);

        // Used to expose dirac deltas.
        // convert (* x[i] (- 0 y)) into (* x[i] y -1)
        // this is useful so that we can have 
        // (>< i (* x[i] (- 0 delta_i_j)))
        // => (>< i (* x[i] delta_i_j -1))
        // => (* x[j] -1)
        inner.clear();
        for(Expr *e : m->inner) {
            // cout << "inspecting: " << m->to_str() << " | " << e->to_str() << "\n";
            if(auto *s = dynamic_cast<Sub *>(e)) {
                if (is_const_zero(s->l)) {
                    inner.push_back(new ConstantInt(-1));
                    inner.push_back(s->r);
                    continue;
                }
            } 

            inner.push_back(e);
        }

        if (inner.size() == 1) { return inner[0]; }
        if (inner.size() == 0) { return new ConstantInt(1); }

        m = new Mul(inner);
        return m;
    
    }

    virtual Expr *visitAdd(Add *a) {
        vector<Expr *> inner;
        for(Expr *e : a->inner) {
            if (is_const_zero(e)) continue;
            inner.push_back(visitExpr(e));
        }

        if (inner.size() == 1) return inner[0];
        if (inner.size() == 0) { return new ConstantInt(0); }

        return new Add(inner);
    }

     virtual Expr *visitSub(Sub *b) {
        if(is_const_zero(b->r)) { return visitExpr(b->l); };

        // convert (- 0 x) into (* (-1) x)
        // Used to convert
        //(>< i (- 0 (* (δ i->j) v[i]))
        //=> (>< i (* -1 (δ i j) v[i]))
        //=> (* -1 v[j])
        if (is_const_zero(b->l)) {
            if(Mul *m = dynamic_cast<Mul *>(b->r)) {
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
int findDeltaForIndex(const Arr* ix, vector<Expr *> es) {
    for(int i = 0; i < (int)es.size(); ++i) {
        Delta *d = dynamic_cast<Delta *>(es[i]);
        if(d && d->old == ix) return i;
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
        for(Expr *e : m->inner) {
            if (e == d) continue;
            substInner.push_back(ExprVisitor::visitExpr(e->subst(c->ix,
                            d->new_)));
        }
        return new Mul(substInner);
    };
};


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


Expr *simplify(Expr *e, bool debug) {
    PushContractionInwardsVisitor pushv;
    ConstantFoldVisitor cfv;
    EliminateContractionVisitor ecv;
    FlattenVisitor fv;
    for(int i = 0; i < 100; ++i) {
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
    Arr* i = new Arr("i");
    Expr *mul = new Mul(a->ix(i), b->ix(i));
    Expr *dot = new Contract(i, mul);

    cout << "******dot******\n";
    cout << dot->to_str_with_shape() << "\n";
    Arr* k = new Arr("k");
    Expr *grad = dot->grad("a", {k});
    cout << "## dot->grad[a k]: ##\n\t" << grad->to_str_with_shape() << "\n";
    cout << "## dot->grad[a k](normalized): ##\n\t" << 
        grad->normalize()->to_str_with_shape() << "\n";
}
void test_expr_matvec() {
    const int NOUT = 3;
    const int NIN = 3;
    Arr *m = new Arr("m", NOUT, NIN);
    Arr *v = new Arr("v", NIN);
    Arr* i = new Arr("i"), *j = new Arr("j");
    Expr *mul = new Mul(m->ix(i, j), v->ix(j));
    Expr *matvec = new Contract(j, mul);

    cout << "******Mat @ vec******\n";
    cout << matvec->to_str_with_shape() << "\n";


    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *gradm = matvec->grad("m", {new Arr("d0"), new Arr("d1")});
    cout << "## mul->grad[a d0 d1]: ##\n\t" << gradm->to_str() << "\n";
    cout << "## mul->grad[a d0 d1](normalized): ##\n\t" <<
        gradm->normalize()->to_str_with_shape() << "\n";

    Expr *gradv = matvec->grad("v", {new Arr("d0")});
    cout << "## mul->grad[v d0]: ##\n\t" << gradv->to_str_with_shape() << "\n";
    cout << "## mul->grad[v d0](normalized): ##\n\t" <<
        gradv->normalize()->to_str_with_shape() << "\n";
}

void test_expr_tanh() {
    const int NDIM = 3;
    Arr *a = new Arr("a", NDIM, NDIM);
    Arr * i = new Arr("i");
    Arr *j = new Arr("j");
    Expr *aixed = a->ix(i, j);
    Expr *matvec = new Tanh(aixed);

    cout << "*****tanh on matrix*****\n";
    cout << matvec->to_str_with_shape() << "\n";


    // before we can take the gradient, we need to put a sum over all
    // free parameters
    Expr *grada = matvec->grad("a", 
            {new Arr("k"), new Arr("l")});
    cout << "## mul->grad[a k l]: ##\n\t" << grada->to_str_with_shape() << "\n";
    cout << "## mul->grad[a k l](normalized): ##\n\t" << 
        grada->normalize()->to_str_with_shape() << "\n";
}

// construct Dot using program
void test_program_dot() {
    Program p;
    IRBuilder builder(p);

    static const int ARRSIZE = 10;
    Arr *a = new Arr("a", ARRSIZE);
    Arr *b = new Arr("b", ARRSIZE);
    Arr *grada = new Arr("grada", ARRSIZE);
    Arr *gradb = new Arr("gradb", ARRSIZE);
    Arr *dot = new Arr("dot");
    Arr *i = new Arr("i");
    Arr *k = new Arr("k");
    builder.copy(dot, {}, new Contract(i, new Mul(a->ix(i), b->ix(i))));
    builder.copy(grada,{k}, 
            new Mul(new ConstantFloat(1e-2), 
                p[dot]->grad("a", {k})->normalize()));
    builder.copy(gradb, {k}, 
            new Mul(new ConstantFloat(1e-2),
               p[dot]->grad("b", {k})->normalize()));

    // Incr is weird, since we should probably be the ones doing the
    // indexing...
    builder.incr(a, {k}, grada->ix(k));
    builder.incr(b, {k}, gradb->ix(k));

    cout << "*****program dot*****\n";
    cout << p.to_str();
}

void test_program_dot_batched() {
    static const int BATCHSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr *focuses = new Arr("focuses", BATCHSIZE, EMBEDSIZE);
    Arr *ctxes = new Arr("ctxes", BATCHSIZE, EMBEDSIZE);
    Arr *total_loss = new Arr("total_loss");

    Arr *bi = new Arr("bi");
    Arr *i = new Arr("i");

    Program p;

    IRBuilder builder(p);
    Forall *forbi = builder.insertFor(bi);

    builder.setInsertPoint(forbi);
    Expr *dots = new Contract(i, new Mul(focuses->ix(bi, i), ctxes->ix(bi, i)));
    Expr *losses = new Sub(new ConstantInt(1), dots);

    builder.copy(total_loss, {}, new Contract(bi, losses));

    Expr *lr = new ConstantFloat(1e-2);

    Arr *dbi = new Arr("dbi"), *dk = new Arr("dk");
    builder.incr(focuses, {dbi, dk},
        new Mul (lr, p[total_loss]->grad("focuses", {dbi, dk})->normalize()));
    builder.incr(ctxes, {dbi, dk},
        new Mul (lr, p[total_loss]->grad("ctxes", {dbi, dk})->normalize()));

    cout << "*****program batched dot*****\n";
    cout << p.to_str() << "\n";


}

void test_program_dot_batched_indirect() {
    cout << "*****program batched, indirect addressed dot (final code that we need for word embeddings)*****\n";
    static const int BATCHSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr *focuses = new Arr("focuses", BATCHSIZE, EMBEDSIZE);
    Arr *ctxes = new Arr("ctxes", BATCHSIZE, EMBEDSIZE);
    Arr *focus = new Arr("focus", EMBEDSIZE);
    Arr *ctx = new Arr("ctx", EMBEDSIZE);
    Arr *dot = new Arr("dot");
    Arr *loss = new Arr("loss");

    Arr *bi = new Arr("bi");
    Arr *i = new Arr("i"), *k = new Arr("k");
    Program p;

    IRBuilder builder(p);
    // builder.setInsertPoint(builder.insertFor(bi));
    Forall *forbi = builder.insertFor(bi);
    builder.setInsertPoint(forbi);
    
    builder.reference(focus, {i}, focuses->ix(bi, i));
    builder.reference(ctx, {i}, ctxes->ix(bi, i));

    builder.copy(dot, {}, new Contract(i, new Mul(focus->ix(i), ctx->ix(i))));
    builder.copy(loss, {}, new Sub(new ConstantInt(1), p[dot]));

    Expr *lr = new ConstantFloat(1e-2);

    builder.incr(focus, {k},
        new Mul (lr, p[loss]->grad("focus", {k})->normalize()));
    builder.incr(ctx, {k},
            new Mul (lr, p[loss]->grad("ctx", {k})->normalize()));
    cout << p.to_str();
}


void test_program_dot_indicrect() {
    cout << "*****program indirect dot*****\n";
    static const int VOCABSIZE = 10;
    static const int EMBEDSIZE = 4;
    Arr *embeds = new Arr("embeds", VOCABSIZE, EMBEDSIZE);
    Arr *focusix = new Arr("focusix");
    Arr *ctxix = new Arr("ctxix");
    Arr *focus = new Arr("focus", EMBEDSIZE);
    Arr *ctx = new Arr("ctx", EMBEDSIZE);
    Arr *dot = new Arr("dot");
    Arr *loss = new Arr("loss");

    Arr *i = new Arr("i"), *k = new Arr("k");
    Program p;

    IRBuilder builder(p);
    

    builder.reference(focus, {i},  embeds->ix(focusix, i));
    builder.reference(ctx, {i}, embeds->ix(ctxix, i));

    builder.copy(dot, {}, new Contract(i, new Mul(focus->ix(i), ctx->ix(i))));
    builder.copy(loss, {}, new Sub(new ConstantInt(1), p[dot]));


    Expr *lr = new ConstantFloat(1e-2);

    builder.incr(focus, {k}, 
        new Mul (lr, p[loss]->grad("focus", {k})->normalize()));
    builder.incr(ctx, {k}, 
            new Mul (lr, p[loss]->grad("ctx", {k})->normalize()));
    cout << p.to_str();
}

Expr *matvecmul(Arr *m, Arr *v, Arr *ix) {
    Arr *c = new Arr("c");
    return new Contract(c, new Mul(m->ix(ix, c), v->ix(c)));
}

Expr *l2(Arr *v, Arr *w) {
    Arr *c = new Arr("c");
    Expr *s = new Sub(v->ix(c), w->ix(c));
    return new Contract(c, new Mul(s, s));    
}

Expr *cell(Arr *I2H, Arr *H2H, Arr *i, Arr *h, Arr *ix) {
    return new Tanh(new Add(matvecmul(I2H, i, ix), matvecmul(H2H, h, ix)));
}

// compute dy/dx
Expr *gradProgram(Program &p, Expr *y, Arr *x) {
    const map<Arr *, Expr *> arr2expr = p.arrays();

    // if we have l in the program, then recursively grad the expression
    Arr *a = dynamic_cast<Arr *>(y);
    if (a) {
        // a naked array can only be 0-dimensional. Otherwise,
        // it must have been sliced.
        assert(a->sh.ndim == 0);
        auto it = arr2expr.find(a);
        // this array does not have an expression associated to it,
        // so we can differentiate it as if it were a leaf.
        if (it == arr2expr.end()) {
            if (a->name == x->name) { 
                return new ConstantInt(1);
            } else {
                return new ConstantInt(0);
            }
        } else {
            // we differentiate the inner expression
            // d(scalar)/dx = d(scalar value)/dx
        }
    }
    
    return nullptr;

}

struct ArrayGatherVisitor : public ExprVisitor {
    set<Arr *> arrays;
    Expr *visitArr(Arr *a) {
        arrays.insert(a);
        return nullptr;
    }
};

// take the derivatives of this expression with all arrays in "params"
// that occur directly within it
void takeDirectDerivatives(Program &p, Expr *e, string name, set<const Arr*> params) {
    ArrayGatherVisitor agv;
    agv.visitExpr(e);


    IRBuilder builder(p);
    for(Arr *a : agv.arrays) {
        if (params.find(a) == params.end()) continue;

        vector<const Arr *> ixs;
        for (int i = 0; i < a->sh.ndim; ++i) {
            ixs.push_back(new Arr("d" + to_string(i)));
        }

        Expr *grad = e->grad(a->name, ixs);
        cout << "d" << name << "/d" << a->name << ":\n\t" << grad->to_str_with_shape() << "\n";
        // builder.copy(new Arr("d" + name + "_d" + a->name), grad->normalize());
    }

}


// get all paths from a source node to a target node, _backwards_. 
// That is, expressions:
// x1 := x0 + 1
// x2 := x1 + 1
// x3 := x2 + 3
// with the call
// getPathsToArr(p, x3, x0, {x3}) will give
// - x3 x2 x1 x0
set<std::vector<Arr *>> getPathsToArr(Program &p, Arr *cur, Arr *target, set<vector<Arr *>> pathsToCur) {
    if (cur == target) {
        return pathsToCur;
    }

    ArrayGatherVisitor agv;
    Expr *rhs = p[cur];

    // cout << "rhs for " << cur->name << " : " << (rhs ? rhs->to_str() : "NULL") << "\n";
    if (!rhs) { return {}; }

    // look in the RHS of this array
    agv.visitExpr(rhs);

    // cout << "arrays for |" << cur->name << "| := |"  << rhs->to_str() << "|";
    // for(Arr *a : agv.arrays) cout << a->name << " ";
    // cout << "|\n";

    set<vector<Arr *>> ps;
    for (Arr *child : agv.arrays) { 
        set<vector<Arr *>> pathsToChild;
        for (vector<Arr *> path: pathsToCur) {
            path.push_back(child);
            pathsToChild.insert(path);
        }
        set<vector<Arr *>> pathsToTarget = getPathsToArr(p, child, target, pathsToChild);
        ps.insert(pathsToTarget.begin(), pathsToTarget.end());
    }
    return ps;
}

// compute dy/dx, through any chain of expressions needed.
// TODO: keep a map of dy/dx -> array holding this value
Expr* takeIndirectDerivatives(Program &p, Arr *y, Arr *x, map<pair<Arr *, Arr*>, Arr*> &ders) {
    (void) ders;
    IRBuilder builder(p);
    set<vector<Arr *>> yTox = getPathsToArr(p, y, x, {{y}});

    Expr *allsum = new ConstantInt(0);
    for(vector<Arr *> path : yTox) {
        // y3 -> y2 -> y1
        // y3/y2 . y2/y1
        Expr *chainprod = new ConstantInt(1);
        for(int i = 0; i < (int)path.size() - 1; ++i) {

            Expr *rhs = p[path[i]];
            if (!rhs) continue;

            vector<const Arr *> ixs;
            for (int j = 0; j < path[i+1]->sh.ndim; ++j) {
                ixs.push_back(new Arr("j" + to_string(j)));
            }
            // Expr *der = rhs->grad(path[i+1]->name, ixs)->normalize();
            // Arr *arrder = new Arr("d" + path[i]->name + "_" + "d" + path[i+1]->name);
            // // builder.copy(arrder, der);
            // chainprod = new Mul(chainprod, arrder);
        }
        allsum = new Add(allsum, chainprod);
    }
    return allsum;
}

void test_lstm() {
    static const int EMBEDSIZE = 5;
    static const int HIDDENSIZE = 10;
    static const int NINPUTS = 1;

    Arr *inputs[NINPUTS];
    Arr *hiddens[NINPUTS+1];

    for(int i = 0; i < NINPUTS; ++i) {
        inputs[i] = new Arr("i" + to_string(i), EMBEDSIZE);
    }

    (void)(inputs);

    for(int i = 0; i < NINPUTS+1; ++i) {
        hiddens[i] = new Arr("h" + to_string(i), EMBEDSIZE);
    }

    Arr *ix = new Arr("ix");

    Arr *I2H = new Arr("I2H", EMBEDSIZE, HIDDENSIZE);
    Arr *H2H = new Arr("H2H", HIDDENSIZE, HIDDENSIZE);
    Arr *H2O = new Arr("H2O", HIDDENSIZE, EMBEDSIZE);

    Program p;
    IRBuilder builder(p);

    Arr *hprev = hiddens[0];
    for(int i = 0; i < NINPUTS; ++i) {
        builder.copy(hiddens[i+1], {ix}, cell(I2H, H2H, inputs[i], hprev, ix)); //new Tanh(matvecmul(H2H, hprev, ix)));
        hprev = hiddens[i+1];
    }

    Arr *predict = new Arr("p", EMBEDSIZE);
    builder.copy(predict, {ix}, matvecmul(H2O, hprev, ix));


    Arr *output = new Arr("o", EMBEDSIZE);
    Arr *loss = new Arr("l");
    builder.copy(loss, {}, l2(output, predict));

    cout << "*****LSTM*****:\n";
    cout << p.to_str();

    // cout << "dl_dH2H: |" << p[loss]->grad("H2H", {new Arr("i'"), new Arr("j'")})->normalize()->to_str_with_shape() << "|\n";
    // cout << "dl_dH2H: |" << p[loss]->grad("H2H", {new Arr("i'"), new Arr("j'")})->normalize()->to_str_with_shape() << "|\n";
    takeDirectDerivatives(p, p[loss], "l", {predict});
    takeDirectDerivatives(p, p[hiddens[1]], "h1", {H2H});



    /*
    map<pair<Arr *, Arr *>, Arr*> ders;
    Expr *finalDer = takeIndirectDerivatives(p, loss, H2H, ders);
    builder.copy(new Arr("dl/dH2H"), finalDer->normalize());

    cout << "\n\n paths from loss to H2H:\n";
    set<vector<Arr *>> paths = getPathsToArr(p, loss, H2H, {{loss}});
    for(vector<Arr *> path : paths) {
        cout << "- ";
        for(Arr *a : path) {
            cout << a->name << " ";
        }
        cout << "\n";
    }

    // map<pair<Arr *, Arr *>, Arr*> ders;
    // Expr *finalDer = takeIndirectDerivatives(p, loss, H2H, ders);
    // builder.copy(new Arr("dloss_dH2H"), finalDer->normalize());
    */

    cout << "*****LSTM(full)*****:\n";
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

    test_expr_dot();
    test_expr_matvec();
    test_expr_tanh();
    test_program_dot();
    test_program_dot_batched();
    test_program_dot_indicrect();
    test_program_dot_batched_indirect();

    test_lstm();
}
