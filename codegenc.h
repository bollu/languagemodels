//
// Created by bollu on 18/03/20.
//

#ifndef LANGUAGEMODELS_CODEGENC_H
#define LANGUAGEMODELS_CODEGENC_H

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
#include "lang.h"

using namespace std;

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


#endif //LANGUAGEMODELS_CODEGENC_H
