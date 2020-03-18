//
// Created by bollu on 18/03/20.
//

#ifndef LANGUAGEMODELS_CODEGENMLIR_H
#define LANGUAGEMODELS_CODEGENMLIR_H

#include "lang.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;


using namespace std;

struct CodegenMLIR {

    mlir::ModuleOp program(mlir::MLIRContext &ctx, Program p) {
        mlir::Builder builder(&ctx);
        mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
        cout << "created module!\n";

        mlir::FuncOp main = mlir::FuncOp::create(UnknownLoc::get(&ctx), "main", builder.getFunctionType({}, llvm::None));
        cout << "created function!\n";

        cout << "verifying module..."  << std::flush;
        assert(!failed(mlir::verify(theModule)));
        cout << "VERIFIED!\n";
        for (Assign a: p.stmts.assigns()) {
            // genAssign(builder, a);
        };
        cout << "returned the module\n";
        return theModule;

    }
};
#endif //LANGUAGEMODELS_CODEGENMLIR_H
