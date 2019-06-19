//=- RISCVMachineFunctionInfo.cpp - RISCV machine function info ---*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines functionality for RISCVMachineFunctionInfo.
//
//===----------------------------------------------------------------------===//

#include "RISCVMachineFunctionInfo.h"

using namespace llvm;

static cl::opt<bool> EnableSaveRestore(
    "enable-save-restore", cl::init(false),
    cl::desc("Enable save/restore of callee-saved registers via libcalls"));

bool RISCVMachineFunctionInfo::useSaveRestoreLibCalls() const {
  // We cannot use fixed locations for the callee saved spill slots if the
  // function uses a varargs save area.
  return EnableSaveRestore && VarArgsSaveSize == 0;
}
