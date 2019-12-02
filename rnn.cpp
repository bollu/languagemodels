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

enum class ExprType {
    Add,
    Sub,
    Dot, 
    MatMatMul, 
    MatVecMul,
    Replicate,
    PointwiseMul,
    // constant aray
    Constant,
    Tanh,
    DerTanh,
    Arr, 
    Undef, 
    AllOnes, 
    Identity, 
    AllZeros,
    // select a sub array from a given array. Used for embedding indexing.
    Index,
    // define a batch dimensions
    Batch,
    // unbatch a batch dimension to specify final computation of gradient update
    Unbatch,
    // let bindings. 
    // Let, 
};

static const int MAXARGS = 30;
static const int MAXPRED = 30;

struct Expr {
    ExprType ty = ExprType::Undef;


    // if it's an Expr::Arr, this is the name of the array.
    string arrname;

    // if it's a virtual node such as AllZeros, AllOnes, this will be its
    // length.
    Shape sh;

    Expr *args[MAXARGS] = { nullptr };
    int nargs = 0;

    // constant float for Constant
    float constval;

    void addarg(Expr *e) {
        assert(nargs < MAXARGS);
        args[nargs++] = e;
        // assert(e->npred < MAXPRED);
        // e->pred[e->npred++] = this;
    }

    Expr() = default;
    Expr(const Expr &other) = default;

    bool operator > (const Expr &other) const {
        return (other < *this) && (other != *this);
    }

    bool operator < (const Expr &other) const {
        if (ty < other.ty) { return true; }
        if (ty > other.ty) { return false; }
        
        /*
        if ((ty == ExprType::AllOnes || ty == ExprType::AllZeros || ty ==
                    ExprType::Index || ty == ExprType::Unbatch || ty ==
                    ExprType :: Replicate) && (sh < other.sh)) {
            return true;
        }
        */

        if (nargs < other.nargs) return true;
        if (nargs > other.nargs) return false;
        assert(nargs == other.nargs);
        for(int i = 0; i < nargs; ++i) {
            if (*args[i] < *other.args[i]) return true;
            if (*args[i] > *other.args[i]) return false;
        }
        // predecessors don't matter since we never use them.

        if (ty == ExprType::Constant && constval < other.constval) { 
            return true; 
        }
        if (ty == ExprType::Constant && constval > other.constval) { 
            return false; 
        }

        return false;
    } 

    bool operator != (const Expr &other) const {
        if (ty != other.ty) { return true; }
        if ((ty == ExprType::AllOnes || ty == ExprType::AllZeros || ty ==
                    ExprType::Index || ty == ExprType::Unbatch || ty ==
                    ExprType :: Replicate) && sh != other.sh) {
            return true;
        }

        if (nargs != other.nargs) return true;
        for(int i = 0; i < nargs; ++i) {
            if (*args[i] != *other.args[i]) return true;
        }
        // predecessors don't matter since we never use them.

        if (ty == ExprType::Constant && constval != other.constval) { 
            return true; 
        }

        return false;
    } 

    bool operator == (const Expr &other) {
        return !(*this != other);
    }

    static Expr *arr(string name, Shape sh) {
        Expr *e = new Expr;
        e->arrname = name;
        e->sh = sh;
        e->ty = ExprType::Arr;
        return e;
    }

    static Expr *arr(string name, int dim1) {
        return Expr::arr(name, Shape::oned(dim1));
    }

    static Expr *arr(string name, int dim1, int dim2) {
        return Expr::arr(name, Shape::twod(dim1, dim2));
    }

    static Expr* add(Expr *l, Expr *r) {
        if (l->ty == ExprType::AllZeros) return r;
        if (r->ty == ExprType::AllZeros) return l;

        Shape shunified = Shape::unify(l->sh, r->sh);

        l = Expr::replicate(l, shunified);
        r = Expr::replicate(r, shunified);

        Expr *e = new Expr;
        e->sh = shunified;
        e->ty = ExprType::Add;
        e->addarg(l); 
        e->addarg(r);
        // e->val = Arr(shunified, e->to_str());
        return e;
    }

    static Expr* sub(Expr *l, Expr *r) {
        Shape shunified = Shape::unify(l->sh, r->sh);

        l = Expr::replicate(l, shunified);
        r = Expr::replicate(r, shunified);

        Expr *e = new Expr;
        e->sh = shunified;
        e->ty = ExprType::Sub;
        e->addarg(l); 
        e->addarg(r);
        return e;
    }

    static Expr *pointwisemul(Expr *l,  Expr *r) {
        Shape shunified = Shape::unify(l->sh, r->sh);
        l = Expr::replicate(l, shunified);
        r = Expr::replicate(r, shunified);

        Expr *e = new Expr;
        e->ty = ExprType::PointwiseMul;
        e->sh = shunified;
        e->addarg(l);
        e->addarg(r);
        return e;
    }

    static Expr *matmatmul(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::MatMatMul;
        e->addarg(l);
        e->addarg(r);
        assert(l->sh.ndim == 2);
        assert(r->sh.ndim == 2);
        assert(l->sh[1] == r->sh[0]);
        e->sh = Shape::twod(l->sh[0], r->sh[1]);
        return e;
    }

    static Expr *matvecmul(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::MatVecMul;
        e->addarg(l);
        e->addarg(r);

        assert(l->sh.ndim == 2);
        assert(r->sh.ndim == 1);
        assert(l->sh[1] == r->sh[0]);
        e->sh = Shape::oned(l->sh[0]);
        return e;

    }

    static Expr *replicate(Expr *inner, Shape replicatesh) {
        // constant fold directly in the replicate()
        if (inner->sh == replicatesh) return inner;

        Expr *e = new Expr;
        e->ty = ExprType::Replicate;
        e->addarg(inner);
        // the new shape is that of the shape we want to replicate to
        e->sh = replicatesh;
        return e;
    }


    static Expr *dot(Expr *l, Expr *r) {
        Expr *e = new Expr;
        e->ty = ExprType::Dot;
        e->addarg(l); 
        e->addarg(r);

        // assert(this->sh == other->sh);
        e->sh = Shape::zerod();
        // e->val = Arr(1, e->to_str());
        return e;
    }

    static Expr *tanh(Expr *inner) {
        Expr *e = new Expr;
        e->ty = ExprType::Tanh;
        e->addarg(inner);
        e->sh = inner->sh;
        return e;
    }

    static Expr *dertanh(Expr *inner) {
        Expr *e = new Expr;
        e->ty = ExprType::DerTanh;
        e->addarg(inner);
        e->sh = inner->sh;
        return e;
    }


    static Expr *allones(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::AllOnes;
        e->sh = sh;
        return e;
    }


    static Expr *identity(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::Identity;
        e->sh = sh;
        return e;
    }


    static Expr *allzeros(Shape sh) {
        Expr *e = new Expr;
        e->ty = ExprType::AllZeros;
        e->sh = sh;
        return e;
    }

    static Expr *index(Expr *arr, Expr *index) {
        // indexing array is 1D. Contains offsets into array.
        assert(index->sh.ndim == 1);

        Expr *e = new Expr;
        e->addarg(arr);
        e->addarg(index);
        e->ty = ExprType::Index;


        // array shape: 3 3 [1 0 0; 0 1 0; 0 0 1]
        // index set: 1 [1 2]
        // out[index[0]] = [1 0 0]
        // out[index[1]] = [0 1 0]
        // size of out: inner * len(index)
        e->sh = arr->sh.removeOutermost().addOutermost(index->sh.vals[0]);
        return e;
        
    }

    static Expr *batch(Expr *arr) {
        Expr *e = new Expr;
        e->addarg(arr);
        e->ty = ExprType::Batch;
        // e->val = Arr(arr->sh, e->to_str());
        e->sh = arr->sh.removeOutermost();
        return e;
    }

    static Expr *unbatch(Expr *arr, int batchsize) {
        Expr *e = new Expr;
        e->addarg(arr);
        e->ty = ExprType::Unbatch;
        // TODO: check that this matches the inner Batch() sizes.
        e->sh = arr->sh.addOutermost(batchsize); 
        // e->val = Arr(e->sh, e->to_str());
        return e;
    }

    static Expr *constant(float c) {
        Expr *e = new Expr;
        e->ty = ExprType::Constant;
        e->constval = c;
        e->sh = Shape::zerod();
        // e->val = Arr(e->sh, e->to_str());
        return e;
    }

    Expr *grad(Expr *dx) {
        assert(dx->ty == ExprType::Arr);
        return grad_(dx, dx->sh, {});
    }


    // return the expression for the gradient with the other array
    Expr *grad_(Expr *dx, Shape outsh, map<string, Expr *> dermap) {
        switch(ty) {
            case ExprType::Arr:  {
                assert(dx->ty == ExprType::Arr);
                if (dx->arrname == this->arrname) {
                    return Expr::identity(outsh);
                }
                else {
                    return Expr::allzeros(outsh);
                }
                // find this array in the derivative map.
                // auto it = dermap.find(this->arrname);                
                //  // if it's in the map, return the value
                // if (it != dermap.end()) { return new Expr(*it->second); }
                // else { return Expr::allzeros(sh); }

             }
            case ExprType::Add:
                return Expr::add(args[0]->grad_(dx, outsh, dermap), args[1]->grad_(dx, outsh, dermap));
            case ExprType::Sub:
                return Expr::sub(args[0]->grad_(dx, outsh, dermap), args[1]->grad_(dx, outsh, dermap));
            case ExprType::Dot: {
                // dl x r = outsh
                // dl = outsh x r[0]
                Expr *l = args[0], *r = args[1];
                // dr x l = outsh
                // dr = outsh x l[0]
                Expr *dl = l->grad_(dx, Shape::twod(outsh[0], r->sh[0]), dermap);
                Expr *dr = r->grad_(dx, Shape::twod(outsh[0], l->sh[0]), dermap);
                return Expr::add(Expr::matvecmul(dl, r), Expr::matvecmul(dr, l));
            }
                // return Expr::add(Expr::pointwisemul(args[0]->grad_(dx, outsh, dermap), args[1]),
                //         Expr::pointwisemul(args[0], args[1]->grad_(dx, outsh, dermap)));
            // d/dx [tanh(i)] = tanh'(i) . di/dx
            case ExprType::Tanh: {
                Expr *dinner = args[0]->grad_(dx, args[0]->sh, dermap);
                return Expr::pointwisemul(Expr::dertanh(args[0]), dinner);
             }
            case ExprType::MatMatMul: {
                    assert(false && "need to implement replicate");
                   return Expr::add(Expr::matmatmul(args[0]->grad_(dx, args[0]->sh, dermap), args[1]),
                           Expr::matmatmul(args[0], args[1]->grad_(dx, args[1]->sh, dermap)));

               }
            case ExprType::MatVecMul: {
                   // TODO!! What is the justification for the "fit shape"??
                   return Expr::add(
                           Expr::replicate(Expr::matvecmul(args[0]->grad_(dx, args[0]->sh, dermap), args[1]), outsh),
                           Expr::replicate(Expr::matvecmul(args[0], args[1]->grad_(dx, args[1]->sh, dermap)), outsh));
                   return Expr::allzeros(outsh);
               }
            case ExprType::Batch: {
                return Expr::batch(args[0]->grad_(dx, args[0]->sh, dermap));
            }

            default:
                cerr << "\nUnimplemented gradient:\n|" << to_str() << "|\n";
                assert(false && "unimplemented");
        }
    }

    string to_str() const {
        switch(ty) {
            case ExprType::Arr: return arrname;
            case ExprType::AllOnes: return "11..1";
            case ExprType::Identity: return "id(1010)";
            case ExprType::AllZeros: return "00..0";
            case ExprType::Add: 
                return "(+ " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Sub: 
                return "(- " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Dot: 
                return "(dot " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Tanh:
                return "(tanh " + args[0]->to_str() + ")";

            case ExprType::DerTanh:
                return "(tanh' " + args[0]->to_str() + ")";

            case ExprType::PointwiseMul: 
                return "(.* " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::MatMatMul: 
                return "(@mm " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::MatVecMul: 
                return "(@mv " + args[0]->to_str() + " " + 
                        args[1]->to_str() + ")";

            case ExprType::Replicate: 
                return "(replicate " + sh.to_str() + " " + 
                    args[0]->to_str()+ ")";

            case ExprType::Index: 
                return "(!! " + args[0]->to_str() + " " +
                            args[1]->to_str() + ")";
            case ExprType::Batch:
                return "(batch " + args[0]->to_str() + ")";
            case ExprType::Unbatch:
                return "(unbatch " + to_string(sh[0]) + " " + 
                    args[0]->to_str() + ")";
            case ExprType::Constant:
                return "(constant " + to_string(constval) + ")";
            default:
                assert(false && "unimplemented to_str()");

        }
    }
    
};

enum class StmtType {
    Assign,
    AddAssign,
};

struct Stmt {
    Arr lhs;
    Expr *rhs;
    StmtType ty;

    Stmt assign(Arr lhs, Expr *rhs) const {
        Stmt s;
        s.lhs = lhs;
        s.rhs = rhs;
        s.ty = StmtType::Assign;
        return s;
    }

    Stmt addassign(Arr lhs, Expr *rhs) const {
        Stmt s;
        s.lhs = lhs;
        s.rhs = rhs;
        s.ty = StmtType::AddAssign;
        return s;
    }
};

using StmtList = vector<Stmt>;


bool isexprconstant(const Expr *e) {
    switch(e->ty) {
        case ExprType::AllOnes:
        case ExprType::AllZeros:
            return true;
        default: return false;
    }
}

void getSubexpressions(Expr *e, map<Expr, int> &s) {

    switch(e->ty) {
        case ExprType::Arr: return;
        case ExprType::Add:
        case ExprType::Sub:
        case ExprType::PointwiseMul:
        case ExprType::Dot:
        case ExprType::MatMatMul:
        case ExprType::MatVecMul:
            s[*e] += 1;
            getSubexpressions(e->args[0], s);
            getSubexpressions(e->args[1], s);
            return;

        case ExprType::Tanh:
        case ExprType::DerTanh:
            s[*e] += 1;
            getSubexpressions(e->args[0], s);
            return;

        default:
            cerr << "unknown expr:\n|" << e->to_str() << "|\n";
            assert(false && "unknown expr for getSubexpressions");
    }
    
}

Expr *commonSubexpressionElimination(Expr *e) {
    map<Expr, int> s;
    getSubexpressions(e, s);

    cout << "subexpressions:\n";
    for(auto it : s) {
        cout << "**--- " << it.second << ":" << it.first.to_str() << "\n";
    }
    exit(0);

    return e;
}

// move all constants to the left. If both params are constants, then fold
Expr *constantfold(Expr *e) {
    switch(e->ty) {
        case ExprType::PointwiseMul:
            if (e->args[0]->ty == ExprType::AllZeros || 
                    e->args[1]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh);
            }
            if (e->args[0]->ty == ExprType::AllOnes) {
                return constantfold(e->args[1]);
            }
            if (e->args[1]->ty == ExprType::AllOnes) {
                return constantfold(e->args[0]);
            }
            return Expr::pointwisemul(constantfold(e->args[0]), constantfold(e->args[1]));

        case ExprType::MatVecMul:
            if (e->args[0]->ty == ExprType::AllZeros || 
                    e->args[1]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh);
            }
            if (e->args[0]->ty == ExprType::Identity) {
                return constantfold(e->args[1]);
            }
            if (e->args[1]->ty == ExprType::Identity) {
                return constantfold(e->args[0]);
            }
            return Expr::matvecmul(constantfold(e->args[0]), constantfold(e->args[1]));



        case ExprType::Add:
            if(e->args[0]->ty == ExprType::AllZeros && 
                    e->args[1]->ty == ExprType::AllOnes) {
                return Expr::allones(e->sh);
            }
            if(e->args[0]->ty == ExprType::AllZeros) {
                return constantfold(e->args[1]);
            }
            if(e->args[1]->ty == ExprType::AllZeros) {
                return constantfold(e->args[0]);
            }
            return Expr::add(constantfold(e->args[0]), constantfold(e->args[1]));

        case ExprType::Sub:
            if(e->args[1]->ty == ExprType::AllZeros) {
                return constantfold(e->args[0]);
            }
            return Expr::sub(constantfold(e->args[0]), constantfold(e->args[1]));

        case ExprType::Replicate:
            if (e->args[0]->sh == e->sh) {
                return constantfold(e->args[0]);
            } 
            else if (e->args[0]->ty == ExprType::AllOnes) {
                return Expr::allones(e->sh);
            }
            else if (e->args[0]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh);
            }
            else {
                return Expr::replicate(constantfold(e->args[0]), e->sh);
            }

        case ExprType::Batch:
            if (e->args[0]->ty == ExprType::AllZeros) {
                return Expr::allzeros(e->sh);
            }
            else if (e->args[0]->ty == ExprType::AllOnes) {
                return Expr::allones(e->sh);
            }
            return Expr::batch(constantfold(e->args[0]));

        default: return e;
    }
}

Expr *simplify(Expr *e) {
    for(int i = 0; i < 10; ++i) e = constantfold(e);
    return e;
}


void test_expr_dot() {
    cout << "***dot***\n";
    const int N = 3;
    Expr *a = Expr::arr("a", Shape::oned(N));
    Expr *b = Expr::arr("b", Shape::oned(N));

    // Expr *add = Expr::add(a, b);
    Expr *dot = Expr::dot(a, b);
    cout << "dot: " << dot->to_str();

    Expr *dotder = dot->grad(b);
    cout << "\ngrad of dot wrt b: " << dotder->to_str();

    for(int i = 0; i < 3; ++i) {
        dotder = constantfold(dotder);
    }
    cout << " | simpl " << dotder->to_str();

    dotder = dot->grad(a);
    cout << "\ngrad of dot wrt a:" << dotder->to_str();
    for(int i = 0; i < 3; ++i) {
        dotder = constantfold(dotder);
    }
    cout << " | simpl " << dotder->to_str();
    cout << "\n";
}

void test_expr_matvec() {
    cout << "***Ax***\n";
    static const int N = 3;
    static const int M = 5;
    Expr *A = Expr::arr("A", Shape::twod(N, M));
    Expr *x = Expr::arr("x", Shape::oned(M));

    Expr *out = Expr::matvecmul(A, x);
    cout << "out: " << out->to_str() << "\n";
    cout << "out->grad[A]: " << out->grad(A)->to_str() << "\n";
    cout << "out->grad[A]: " << simplify(out->grad(A))->to_str() << "\n";
    cout << "out->grad[x]: " << simplify(out->grad(x))->to_str() << "\n";
}


void test_expr_tanh_matvec() {
    cout << "***Tanh(Ax)***\n";
    static const int N = 3;
    static const int M = 5;
    Expr *A = Expr::arr("A", Shape::twod(N, M));
    Expr *x = Expr::arr("x", Shape::oned(M));

    Expr *out = Expr::tanh(Expr::matvecmul(A, x));
    cout << "out: " << out->to_str() << "\n";
    cout << "out->grad[A]: " << out->grad(A)->to_str() << "\n";
    cout << "out->grad[A]: " << simplify(out->grad(A))->to_str() << "\n";
    cout << "out->grad[x]: " << simplify(out->grad(x))->to_str() << "\n";
}

void test_expr_rnn() {
    // joint modelling of words and corpus
    static const int windowsize = 3;
    static const int embedsize = 4;
    static const int hiddensize = 10;
    vector<int> sentence;
    vector<Expr *> embeds;
    Expr *inputs[windowsize];
    Expr *hiddens[windowsize+1];

    Expr *H2H = Expr::arr("H2H", hiddensize, hiddensize);
    Expr *I2H = Expr::arr("I2h", hiddensize, embedsize);

    Expr *H2O = Expr::arr("H2O", embedsize, hiddensize);
    Expr *H2OBias = Expr::arr("H2OBias", embedsize);

    for(int i = 0; i < (int)vocab.size(); ++i) {
        embeds[i] = Expr::arr(ix2word[i], embedsize);
    }

    for(int i = 0; i < windowsize; ++i) {
        inputs[i] = Expr::arr("i" + std::to_string(i), embedsize);
    }

    hiddens[0] = Expr::arr("hinit", hiddensize);
    for(int i = 1; i <= windowsize; ++i) {
        hiddens[i] = Expr::tanh(Expr::add(Expr::matvecmul(I2H, inputs[i-1]),
                    Expr::matvecmul(H2H, hiddens[i-1])));
    }

    Expr *out = Expr::tanh(Expr::add(Expr::matvecmul(H2O, hiddens[windowsize]), H2OBias));
    cout << "out: " << out->to_str() << "\n";

    Expr *H2Hgrad = out->grad(H2H);
    for(int i = 0; i < 6; ++i) {
        cout << "\n" << i << "| out->grad[H2H]:" << H2Hgrad->to_str();
        H2Hgrad = constantfold(H2Hgrad);
    }

    commonSubexpressionElimination(H2Hgrad);

    Expr *H2OGrad = out->grad(H2O);
    for(int i = 0; i < 6; ++i) {
        cout << "\n" << i << "| out->grad[H2O]:" << H2OGrad->to_str();
        H2OGrad = constantfold(H2OGrad);
    }

}

void test_expr_rnn_batched() {
    cout << "\n***RNN Computation with batching***\n";
    // joint modelling of words and corpus
    static const int batchsize = 2;
    static const int windowsize = 1;
    static const int embedsize = 4;
    static const int hiddensize = 10;
    vector<int> sentence;
    Expr *embeds[vocab.size()];


    Expr *H2H = Expr::arr("H2H", hiddensize, hiddensize);
    Expr *I2H = Expr::arr("I2H", hiddensize, embedsize);
    // Expr *H2HBias = Expr::arr("H2HBias", hiddensize);

    Expr *H2O = Expr::arr("H2O", embedsize, hiddensize);
    Expr *H2OBias = Expr::arr("H2OBias", embedsize);

    for(int i = 0; i < (int)vocab.size(); ++i) {
        embeds[i] = Expr::arr(ix2word[i], embedsize);
        (void)embeds[i];
    }

    // inputs: batchsize x embedsize
    Expr *inputs[windowsize];
    for(int i = 0; i < windowsize; ++i) {
        inputs[i] = Expr::arr( "i_batched" + std::to_string(i), batchsize,
                embedsize);
    }

    // outputs array, batchsize x embedsize
    Expr *outputs = Expr::arr("output_batched", batchsize, embedsize);

    Expr *hiddens[windowsize+1];

    // create the compute kernel
    hiddens[0] = Expr::arr("hinit", hiddensize);
    for(int i = 1; i <= windowsize; ++i) {
        hiddens[i] = Expr::tanh(Expr::add(Expr::matvecmul(I2H,
                        Expr::batch(inputs[i-1])),
                    Expr::matvecmul(H2H, hiddens[i-1])));
    }

    // should be batchsize x embedsize, but our kernel pretends it is embedsize
    Expr *predict = Expr::tanh(Expr::add(Expr::matvecmul(H2O,
                    hiddens[windowsize]), H2OBias));

    cout << "RNN prediction:\n" << predict->to_str();

    // full loss
    Expr *loss = Expr::sub(Expr::batch(outputs), predict);
    cout << "\n\nRRN loss:\n" << loss->to_str();

    // find derivative of loss wrt H2H, H2O, I2H, H2OBias
    Expr *H2Hgrad = loss->grad(H2H);
    for(int i = 0; i < 6; ++i) {
        cout << "\n" << i << "| out->grad[H2H]:" << H2Hgrad->to_str();
        H2Hgrad = constantfold(H2Hgrad);
    }
    cout << "\n\n";


    Expr *H2Ograd = loss->grad(H2O);
    cout << "\n\nloss->grad[H2O]: " << H2Ograd->to_str();

    Expr *I2Hgrad = loss->grad(I2H);
    cout << "\n\nloss->grad[I2H]: " << I2Hgrad->to_str();

    Expr *H2OBiasgrad = loss->grad(H2OBias);
    cout << "\n\nloss->grad[H2OBias]: " << H2OBiasgrad->to_str();


    // scaled gradient
    const float learningrate = 1e-2;
    Expr *H2hgradscaled = Expr::pointwisemul(Expr::constant(learningrate), H2Hgrad);
    cout << "\n\nlearningrate * loss->grad[H2H]:\n" << H2hgradscaled->to_str();

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
    test_expr_tanh_matvec();
    test_expr_rnn_batched();
}
