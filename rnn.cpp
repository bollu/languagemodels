#include <iostream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdio.h>
#include <map>
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

    static Shape unify(Shape sh1, Shape sh2) {
        Shape smaller, larger;
        if (sh1.ndim < sh2.ndim) { smaller = sh1; larger = sh2; }
        else { smaller = sh2; larger = sh1; }

        for(int i = 0; i < smaller.ndim; ++i) {
            assert(smaller[i] == larger[i]);
        }
        return larger;
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

// only 1D frrays
/*
struct Arr {
    Shape sh;
    float *data = nullptr;
    std::string name = "undef";

    Arr() = default;
    Arr(const Arr &other) = default;
    Arr (Shape sh, string name) : 
        sh(sh), data(new float[sh.nelem()]), name(name) {
        };

    Arr (int n, string name) : 
        sh(Shape::oned(n)), data(new float[n]), name(name) {};
    Arr (int n, int m, string name) :
        sh(Shape::twod(n, m)), data(new float[n*m]), name(name) {};

    float &operator [](int ix) {
        assert(ix < sh.nelem());
        assert(ix >= 0);
        return data[ix];
    }
    
    void print_data() {
        cout <<name <<  "[";
        for(int i = 0; i < sh.nelem(); ++i) {
            cout << data[i] << (i < sh.nelem() - 1 ? " " : "");
        }
        cout << "]\n";
    }

    bool operator < (const Arr &other) const {
        // lex compare, first on name, then on shape
        return name < other.name || 
            ((name == other.name) && (sh < other.sh));
    }

    bool operator == (const Arr &other) const {
        return name == other.name && (sh == other.sh);
    }

    bool operator != (const Arr &other) const {
        return !(*this == other);
    }
};
*/

enum class ExprType {
    Add,
    Mul,
    Undef,
    Arr,
    Slice,
    Contract,
    Delta,
    ConstantInt
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

    Expr *ix(vector<Index> ix);
    Expr *ix(Index ix1);
    Expr *ix(Index ix, Index ix2);

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
        string s = to_str();
        s += "[";
        for(Index i : free()) s += i.name + ",";
        s += "]";
        return s;
    }

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

struct Delta : public Expr {
    Index old;
    Index new_;

    Delta(Index old, Index new_) : Expr(ExprType::Delta), 
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
    Arr(string name) : Expr(ExprType::Arr), name(name) {};
    string to_str() const { return name; };

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


};


struct Add : public Expr {
    Expr *l, *r;

    Add (Expr *l, Expr *r) : Expr(ExprType::Add), l(l), r(r) {};

    string to_str() const {
        return "(+ " + l->to_str() + " " + r->to_str() + ")";
    }

    set<Index> free() const {
        set<Index> lf = l->free();
        set<Index> rf = l->free();
        rf.insert(lf.begin(), lf.end());
        return rf;
    }

    Expr *grad(string name, vector<Index> ixs) {
        return new Add(l->grad(name, ixs), r->grad(name, ixs));
    }

    Expr *subst(Index old, Index new_) const {
        return new Add(l->subst(old, new_),
                r->subst(old, new_));
    }
};

struct Mul : public Expr {
    Expr *l, *r;

    Mul (Expr *l, Expr *r) : Expr(ExprType::Mul), l(l), r(r) {};

    string to_str() const {
        return "(* " + l->to_str() + " " + r->to_str() + ")";
    }

    set<Index> free() const {
        set<Index> lf = l->free();
        set<Index> rf = l->free();
        rf.insert(lf.begin(), lf.end());
        return rf;
    }

    Expr *grad(string name, vector<Index> ixs) {
        return new Add(new Mul(l->grad(name, ixs), r),
                    new Mul(l, r->grad(name, ixs)));
    }

    Expr *subst(Index old, Index new_) const {
        return new Mul(l->subst(old, new_),
                r->subst(old, new_));
    }
};

struct Slice : public Expr {
    Expr *inner;
    vector<Index> ixs;

    Slice(Expr *inner, vector<Index> ixs): 
        Expr(ExprType::Slice), inner(inner), ixs(ixs) {}; 

    string to_str() const  {
        string s =  "(! " + inner->to_str() + " ";

        for(int i = 0; i < (int)ixs.size(); ++i) {
            s += ixs[i].to_str() + (i < (int)ixs.size() - 1 ? " " : "");
        }

        s += ")";
        return s;
    }

    set<Index> free() const {
        set<Index> s = inner->free();
        for (Index ix :ixs) { s.insert(ix); };
        return s;
    }

    Expr *grad(string name, vector<Index> gixs) {
        // we are a slice of an expression, allow the inner part
        // to continue uninhibited
        Arr *a = dynamic_cast<Arr *>(inner);
        if (!a) { return new Slice(inner->grad(name, gixs), ixs); }

        // this IS the slice of an array, but not the array we are
        // looking for. Return zero
        if(name != a->name) { return new ConstantInt(0); }

        // we ARE the slce of the array we were looking for.
        assert(name == a->name);

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
        return new Slice(inner, ixsnew);
    }

};


Expr *Expr::ix(vector<Index> ixs) {
    return new Slice(this, ixs);
}

Expr *Expr::ix(Index ix1) {
    return new Slice(this, {ix1});
}

Expr *Expr::ix(Index ix1, Index ix2) {
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

struct Stmt {
    string lhs;
    Expr *rhs;
};


Expr *pushContractionsInwards(Expr *e) {
    if (Add *a = dynamic_cast<Add *>(e)) {
        return new Add(pushContractionsInwards(a->l),
                pushContractionsInwards(a->r));
    }
    else if (Contract *c = dynamic_cast<Contract *>(e)) {
        Add *a = dynamic_cast<Add *>(c->inner);
        if (!a) return e;
        Expr *l = new Contract(c->ix, a->l);
        Expr *r = new Contract(c->ix, a->r);
        return new Add(l, r);

    }
    return e;
}

// move dirac deltas leftwards
Expr *reassocDelta(Expr *e) {
    return e;
}

// eliminate (>< i (* (δ i->j) t1)) with t1[i/j]
Expr *eliminateContractions(Expr *e)  {
    Contract *c = dynamic_cast<Contract *>(e);
    if (!c) return e;
    Mul *m = dynamic_cast<Mul *>(c->inner);
    if (!m) return e;
    Delta *d = dynamic_cast<Delta *>(m->l);
    if (!d) return e;

    if (c->ix == d->old) {
        return m->r->subst(d->old, d->new_);
    }

    return e;
};


bool is_const_zero(Expr *e) {
    ConstantInt *i = dynamic_cast<ConstantInt*>(e);
    if (!i) return false;
    return i->i == 0;
}

bool is_const_one(Expr *e) {
    ConstantInt *i = dynamic_cast<ConstantInt*>(e);
    if (!i) return false;
    return i->i == 1;
}

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

Expr *simplify(Expr *e) {
    for(int i = 0; i < 4; ++i) {
        cout << "--\n";
        cout << i << "|"  << e->to_str() << "\n";
        e = pushContractionsInwards(e);
        cout << i << "|PUSH|" << e->to_str() << "\n";
        e = constantFold(e);
        cout << i << "|FOLD|" << e->to_str() << "\n";
        e = eliminateContractions(e);
        cout << i << "|ELIM|" << e->to_str() << "\n";
    }
    return e;
}

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
    Expr *a = new Arr("a");
    Expr *b = new Arr("b");
    Index i("i");
    Expr *mul = new Mul(a->ix(i), b->ix(i));
    Expr *dot = new Contract(i, mul);

    cout << "***dot***\n";
    cout << dot->detailed_to_str() << "\n";
    Index k("k");
    Expr *grad = dot->grad("a", {k});
    cout << "dot->grad[a k]: " << grad->detailed_to_str() << "\n";
    simplify(grad);
}

void test_expr_matvec() {
    Expr *a = new Arr("a");
    Expr *b = new Arr("b");
    Index i("i"), j("j");
    Expr *mul = new Mul(a->ix(i, j), b->ix(j));
    Expr *matvec = new Contract(j, mul);

    cout << "***mul***\n";
    cout << matvec->detailed_to_str() << "\n";
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
    // test_expr_tanh_matvec();
    // test_expr_rnn_batched();
}
