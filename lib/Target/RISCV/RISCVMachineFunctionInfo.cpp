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
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  if (!EnableSaveRestore)
    return false;

  // We cannot use fixed locations for the callee saved spill slots if there is
  // any chance of other objects also requiring fixed locations in the stack
  // frame. This is called before adding the fixed spill slots so there should
  // be no fixed objects at all.
  if (MFI.getNumFixedObjects())
    return false;

  return true;
}
