//===-- RISCVFrameLowering.cpp - RISCV Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the RISCV implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "RISCVFrameLowering.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;

bool RISCVFrameLowering::hasFP(const MachineFunction &MF) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         RegInfo->needsStackRealignment(MF) || MFI.hasVarSizedObjects() ||
         MFI.isFrameAddressTaken();
}

// Determines the size of the frame and maximum call frame size.
void RISCVFrameLowering::determineFrameLayout(MachineFunction &MF) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();

  // Get the number of bytes to allocate from the FrameInfo.
  uint64_t FrameSize = MFI.getStackSize();

  // Get the alignment.
  uint64_t StackAlign = RI->needsStackRealignment(MF) ? MFI.getMaxAlignment()
                                                      : getStackAlignment();

  // Make sure the frame is aligned.
  FrameSize = alignTo(FrameSize, StackAlign);

  // Update frame info.
  MFI.setStackSize(FrameSize);
}

void RISCVFrameLowering::adjustReg(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   const DebugLoc &DL, unsigned DestReg,
                                   unsigned SrcReg, int64_t Val,
                                   MachineInstr::MIFlag Flag) const {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  const RISCVInstrInfo *TII = STI.getInstrInfo();

  if (DestReg == SrcReg && Val == 0)
    return;

  if (isInt<12>(Val)) {
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DestReg)
        .addReg(SrcReg)
        .addImm(Val)
        .setMIFlag(Flag);
  } else if (isInt<32>(Val)) {
    unsigned Opc = RISCV::ADD;
    bool isSub = Val < 0;
    if (isSub) {
      Val = -Val;
      Opc = RISCV::SUB;
    }

    unsigned ScratchReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
    TII->movImm32(MBB, MBBI, DL, ScratchReg, Val, Flag);
    BuildMI(MBB, MBBI, DL, TII->get(Opc), DestReg)
        .addReg(SrcReg)
        .addReg(ScratchReg, RegState::Kill)
        .setMIFlag(Flag);
  } else {
    report_fatal_error("adjustReg cannot yet handle adjustments >32 bits");
  }
}

// Returns the register used to hold the frame pointer.
static unsigned getFPReg(const RISCVSubtarget &STI) { return RISCV::X8; }

// Returns the register used to hold the stack pointer.
static unsigned getSPReg(const RISCVSubtarget &STI) { return RISCV::X2; }

static bool EnableMSaveRestore = true;

static bool useSaveRestoreLibCalls(const MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  if (!EnableMSaveRestore)
    return false;

  // We cannot use fixed locations for the callee saved spill slots if there is
  // any chance of other objects also requiring fixed locations in the stack
  // frame. This is called before adding the fixed spill slots so there should
  // be no fixed objects at all.
  if (MFI.getNumFixedObjects())
    return false;

  return true;
}

void RISCVFrameLowering::emitPrologue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  MachineBasicBlock::iterator MBBI = MBB.begin();

  unsigned FPReg = getFPReg(STI);
  unsigned SPReg = getSPReg(STI);

  // Since spillCalleeSavedRegisters may have inserted a libcall, skip past
  // any instructions marked as FrameSetup
  while (MBBI != MBB.end() && MBBI->getFlag(MachineInstr::FrameSetup))
    ++MBBI;

  // Debug location must be unknown since the first debug location is used
  // to determine the end of the prologue.
  DebugLoc DL;

  // Determine the correct frame layout
  determineFrameLayout(MF);

  // FIXME (note copied from Lanai): This appears to be overallocating.  Needs
  // investigation. Get the number of bytes to allocate from the FrameInfo.
  // FIXME: Adjust for callee saved registers that are saved via libcall.
  uint64_t StackSize = MFI.getStackSize();

  // Early exit if there is no need to allocate on the stack
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  // Allocate space on the stack if necessary.
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, -StackSize, MachineInstr::FrameSetup);

  // The frame pointer is callee-saved, and code has been generated for us to
  // save it to the stack. We need to skip over the storing of callee-saved
  // registers as the frame pointer must be modified after it has been saved
  // to the stack, not before.
  // FIXME: assumes exactly one instruction is used to save each callee-saved
  // register.
  std::advance(MBBI, RVFI->getNonFixedCSRs().size());

  // Generate new FP.
  if (hasFP(MF))
    adjustReg(MBB, MBBI, DL, FPReg, SPReg,
              StackSize - RVFI->getVarArgsSaveSize(), MachineInstr::FrameSetup);
}

void RISCVFrameLowering::emitEpilogue(MachineFunction &MF,
                                      MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  const RISCVRegisterInfo *RI = STI.getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  DebugLoc DL = MBBI->getDebugLoc();
  unsigned FPReg = getFPReg(STI);
  unsigned SPReg = getSPReg(STI);

  // If callee-saved registers are saved via libcall, place stack adjustment
  // before this call.
  while (MBBI != MBB.begin() &&
        std::prev(MBBI)->getFlag(MachineInstr::FrameDestroy))
    --MBBI;

  // Skip to before the restores of callee-saved registers
  // FIXME: assumes exactly one instruction is used to restore each
  // callee-saved register.
  auto LastFrameDestroy = std::prev(MBBI, RVFI->getNonFixedCSRs().size());

  // FIXME: Adjust for callee saved registers that are saved via libcall.
  uint64_t StackSize = MFI.getStackSize();

  // Restore the stack pointer using the value of the frame pointer. Only
  // necessary if the stack pointer was modified, meaning the stack size is
  // unknown.
  if (RI->needsStackRealignment(MF) || MFI.hasVarSizedObjects()) {
    assert(hasFP(MF) && "frame pointer should not have been eliminated");
    adjustReg(MBB, LastFrameDestroy, DL, SPReg, FPReg,
              -StackSize + RVFI->getVarArgsSaveSize(),
              MachineInstr::FrameDestroy);
  }

  // Deallocate stack
  adjustReg(MBB, MBBI, DL, SPReg, SPReg, StackSize, MachineInstr::FrameDestroy);
}

int RISCVFrameLowering::getFrameIndexReference(const MachineFunction &MF,
                                               int FI,
                                               unsigned &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterInfo *RI = MF.getSubtarget().getRegisterInfo();
  const auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();

  // Callee-saved registers should be referenced relative to the stack
  // pointer (positive offset), otherwise use the frame pointer (negative
  // offset).
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  int Offset = MFI.getObjectOffset(FI) - getOffsetOfLocalArea() +
               MFI.getOffsetAdjustment();

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }

  if (FI >= MinCSFI && FI <= MaxCSFI) {
    FrameReg = RISCV::X2;
    Offset += MF.getFrameInfo().getStackSize();
  } else {
    FrameReg = RI->getFrameRegister(MF);
    if (hasFP(MF))
      Offset += RVFI->getVarArgsSaveSize();
    else
      Offset += MF.getFrameInfo().getStackSize();
  }
  return Offset;
}

void RISCVFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                              BitVector &SavedRegs,
                                              RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  // Unconditionally spill RA and FP only if the function uses a frame
  // pointer.
  if (hasFP(MF)) {
    SavedRegs.set(RISCV::X1);
    SavedRegs.set(RISCV::X8);
  }

  // If interrupt is enabled and there are calls in the handler,
  // unconditionally save all Caller-saved registers and
  // all FP registers, regardless whether they are used.
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (MF.getFunction().hasFnAttribute("interrupt") && MFI.hasCalls()) {

    static const MCPhysReg CSRegs[] = { RISCV::X1,      /* ra */
      RISCV::X5, RISCV::X6, RISCV::X7,                  /* t0-t2 */
      RISCV::X10, RISCV::X11,                           /* a0-a1, a2-a7 */
      RISCV::X12, RISCV::X13, RISCV::X14, RISCV::X15, RISCV::X16, RISCV::X17,
      RISCV::X28, RISCV::X29, RISCV::X30, RISCV::X31, 0 /* t3-t6 */
    };

    for (unsigned i = 0; CSRegs[i]; ++i)
      SavedRegs.set(CSRegs[i]);

    if (MF.getSubtarget<RISCVSubtarget>().hasStdExtD() ||
        MF.getSubtarget<RISCVSubtarget>().hasStdExtF()) {

      // If interrupt is enabled, this list contains all FP registers.
      const MCPhysReg * Regs = MF.getRegInfo().getCalleeSavedRegs();

      for (unsigned i = 0; Regs[i]; ++i)
        if (RISCV::FPR32RegClass.contains(Regs[i]) ||
            RISCV::FPR64RegClass.contains(Regs[i]))
          SavedRegs.set(Regs[i]);
    }
  }
}

void RISCVFrameLowering::processFunctionBeforeFrameFinalized(
    MachineFunction &MF, RegScavenger *RS) const {
  const TargetRegisterInfo *RegInfo = MF.getSubtarget().getRegisterInfo();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const TargetRegisterClass *RC = &RISCV::GPRRegClass;
  // estimateStackSize has been observed to under-estimate the final stack
  // size, so give ourselves wiggle-room by checking for stack size
  // representable an 11-bit signed field rather than 12-bits.
  // FIXME: It may be possible to craft a function with a small stack that
  // still needs an emergency spill slot for branch relaxation. This case
  // would currently be missed.
  if (!isInt<11>(MFI.estimateStackSize(MF))) {
    int RegScavFI = MFI.CreateStackObject(
        RegInfo->getSpillSize(*RC), RegInfo->getSpillAlignment(*RC), false);
    RS->addScavengingFrameIndex(RegScavFI);
  }
}

// Not preserve stack space within prologue for outgoing variables when the
// function contains variable size objects and let eliminateCallFramePseudoInstr
// preserve stack space for it.
bool RISCVFrameLowering::hasReservedCallFrame(const MachineFunction &MF) const {
  return !MF.getFrameInfo().hasVarSizedObjects();
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions.
MachineBasicBlock::iterator RISCVFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator MI) const {
  unsigned SPReg = RISCV::X2;
  DebugLoc DL = MI->getDebugLoc();

  if (!hasReservedCallFrame(MF)) {
    // If space has not been reserved for a call frame, ADJCALLSTACKDOWN and
    // ADJCALLSTACKUP must be converted to instructions manipulating the stack
    // pointer. This is necessary when there is a variable length stack
    // allocation (e.g. alloca), which means it's not possible to allocate
    // space for outgoing arguments from within the function prologue.
    int64_t Amount = MI->getOperand(0).getImm();

    if (Amount != 0) {
      // Ensure the stack remains aligned after adjustment.
      Amount = alignSPAdjust(Amount);

      if (MI->getOpcode() == RISCV::ADJCALLSTACKDOWN)
        Amount = -Amount;

      adjustReg(MBB, MI, DL, SPReg, SPReg, Amount, MachineInstr::NoFlags);
    }
  }

  return MBB.erase(MI);
}

bool RISCVFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<llvm::CalleeSavedInfo> &CSI) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  // This code is a workaround to be able to check whether save/restore libcalls
  // with fixed spill slots will be used for this function. If so, the generic
  // code can be used, which makes use of getCalleeSavedSpillSlots. The rest of
  // this function is used when fixed spill slots cannot be used, and is mostly
  // copied from the generic code.
  if (useSaveRestoreLibCalls(MF))
    return false;

  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  for (auto &CS : CSI) {
    // If the target has spilled this register to another register, we don't
    // need to allocate a stack slot.
    if (CS.isSpilledToReg())
      continue;

    unsigned Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);

    int FrameIdx;
    if (TRI->hasReservedSpillSlot(MF, Reg, FrameIdx)) {
      CS.setFrameIdx(FrameIdx);
      continue;
    }

    unsigned Size = TRI->getSpillSize(*RC);
    // Nope, just spill it anywhere convenient.
    unsigned Align = TRI->getSpillAlignment(*RC);
    unsigned StackAlign = getStackAlignment();

    // We may not be able to satisfy the desired alignment specification of
    // the TargetRegisterClass if the stack alignment is smaller. Use the
    // min.
    Align = std::min(Align, StackAlign);
    FrameIdx = MFI.CreateStackObject(Size, Align, true);

    CS.setFrameIdx(FrameIdx);
  }
  return true;
}

static const TargetFrameLowering::SpillSlot spillSlotTable64[] = {
  {/*ra*/  RISCV::X1,   -8},
  {/*s0*/  RISCV::X8,   -16},
  {/*s1*/  RISCV::X9,   -24},
  {/*s2*/  RISCV::X18,  -32},
  {/*s3*/  RISCV::X19,  -40},
  {/*s4*/  RISCV::X20,  -48},
  {/*s5*/  RISCV::X21,  -56},
  {/*s6*/  RISCV::X22,  -64},
  {/*s7*/  RISCV::X23,  -72},
  {/*s8*/  RISCV::X24,  -80},
  {/*s9*/  RISCV::X25,  -88},
  {/*s10*/ RISCV::X26,  -96},
  {/*s11*/ RISCV::X27, -104}};

static const TargetFrameLowering::SpillSlot spillSlotTable32[] = {
  {/*ra*/  RISCV::X1,   -4},
  {/*s0*/  RISCV::X8,   -8},
  {/*s1*/  RISCV::X9,  -12},
  {/*s2*/  RISCV::X18, -16},
  {/*s3*/  RISCV::X19, -20},
  {/*s4*/  RISCV::X20, -24},
  {/*s5*/  RISCV::X21, -28},
  {/*s6*/  RISCV::X22, -32},
  {/*s7*/  RISCV::X23, -36},
  {/*s8*/  RISCV::X24, -40},
  {/*s9*/  RISCV::X25, -44},
  {/*s10*/ RISCV::X26, -48},
  {/*s11*/ RISCV::X27, -52}};

static const TargetFrameLowering::SpillSlot spillSlotTable32e[] = {
  {/*ra*/ RISCV::X1,  -4},
  {/*s0*/ RISCV::X8,  -8},
  {/*s1*/ RISCV::X9, -12}};

const TargetFrameLowering::SpillSlot *
RISCVFrameLowering::getCalleeSavedSpillSlots(unsigned &NumEntries) const {
  if (STI.is64Bit()) {
    NumEntries = array_lengthof(spillSlotTable64);
    return spillSlotTable64;
  } else if (STI.isRV32E()) {
    NumEntries = array_lengthof(spillSlotTable32e);
    return spillSlotTable32e;
  } else {
    // RV32.
    NumEntries = array_lengthof(spillSlotTable32);
    return spillSlotTable32;
  }
}

static const char *
getSpillLibCallName(MachineFunction &MF) {
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const std::vector<CalleeSavedInfo> FixedCSRs = RVFI->getFixedCSRs();

  if (FixedCSRs.empty())
    return nullptr;

  using std::max;
  unsigned MaxReg = 0;
  for (auto &CS : FixedCSRs)
    MaxReg = std::max(MaxReg, CS.getReg());

  switch (MaxReg) {
  default:
    llvm_unreachable("Something has gone wrong!");
  case /*s11*/ RISCV::X27: return "__riscv_save_12";
  case /*s10*/ RISCV::X26: return "__riscv_save_11";
  case /*s9*/ RISCV::X25: return "__riscv_save_10";
  case /*s8*/ RISCV::X24: return "__riscv_save_9";
  case /*s7*/ RISCV::X23: return "__riscv_save_8";
  case /*s6*/ RISCV::X22: return "__riscv_save_7";
  case /*s5*/ RISCV::X21: return "__riscv_save_6";
  case /*s4*/ RISCV::X20: return "__riscv_save_5";
  case /*s3*/ RISCV::X19: return "__riscv_save_4";
  case /*s2*/ RISCV::X18: return "__riscv_save_3";
  case /*s1*/ RISCV::X9: return "__riscv_save_2";
  case /*s0*/ RISCV::X8: return "__riscv_save_1";
  case /*ra*/ RISCV::X1: return "__riscv_save_0";
  }
}

static const char *
getRestoreLibCallName(MachineFunction &MF) {
  auto *RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
  const std::vector<CalleeSavedInfo> FixedCSRs = RVFI->getFixedCSRs();

  if (FixedCSRs.empty())
    return nullptr;

  using std::max;
  unsigned MaxReg = 0;
  for (auto &CS : FixedCSRs)
    MaxReg = std::max(MaxReg, CS.getReg());

  switch (MaxReg) {
  default:
    llvm_unreachable("Something has gone wrong!");
  case /*s11*/ RISCV::X27: return "__riscv_restore_12";
  case /*s10*/ RISCV::X26: return "__riscv_restore_11";
  case /*s9*/ RISCV::X25: return "__riscv_restore_10";
  case /*s8*/ RISCV::X24: return "__riscv_restore_9";
  case /*s7*/ RISCV::X23: return "__riscv_restore_8";
  case /*s6*/ RISCV::X22: return "__riscv_restore_7";
  case /*s5*/ RISCV::X21: return "__riscv_restore_6";
  case /*s4*/ RISCV::X20: return "__riscv_restore_5";
  case /*s3*/ RISCV::X19: return "__riscv_restore_4";
  case /*s2*/ RISCV::X18: return "__riscv_restore_3";
  case /*s1*/ RISCV::X9: return "__riscv_restore_2";
  case /*s0*/ RISCV::X8: return "__riscv_restore_1";
  case /*ra*/ RISCV::X1: return "__riscv_restore_0";
  }
}

bool RISCVFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    const std::vector<CalleeSavedInfo> &CSI,
    const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  auto *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
  DebugLoc DL;
  if (MI != MBB.end() && !MI->isDebugInstr())
    DL = MI->getDebugLoc();

  const char *SpillLibCallName = getSpillLibCallName(*MF);
  if (SpillLibCallName)
    BuildMI(MBB, MI, DL, TII.get(RISCV::PseudoCALLReg), RISCV::X5)
        .addExternalSymbol(SpillLibCallName, RISCVII::MO_CALL)
        .setMIFlag(MachineInstr::FrameSetup);

  // Add registers spilt in libcall as liveins, spill other values manually.
  for (auto &CS : RVFI->getFixedCSRs())
    MBB.addLiveIn(CS.getReg());
  for (auto &CS : RVFI->getNonFixedCSRs()) {
    // Insert the spill to the stack frame.
    unsigned Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(MBB, MI, Reg, true, CS.getFrameIdx(), RC, TRI);
  }

  return true;
}

bool RISCVFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    std::vector<CalleeSavedInfo> &CSI, const TargetRegisterInfo *TRI) const {
  if (CSI.empty())
    return true;

  MachineFunction *MF = MBB.getParent();
  auto *RVFI = MF->getInfo<RISCVMachineFunctionInfo>();
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
  DebugLoc DL;
  if (MI != MBB.end() && !MI->isDebugInstr())
    DL = MI->getDebugLoc();

  // Manually restore values not restored by libcall. Insert in reverse order.
  // loadRegFromStackSlot can insert multiple instructions.
  for (auto &CS : reverse(RVFI->getNonFixedCSRs())) {
    unsigned Reg = CS.getReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.loadRegFromStackSlot(MBB, MI, Reg, CS.getFrameIdx(), RC, TRI);
    assert(MI != MBB.begin() &&
            "loadRegFromStackSlot didn't insert any code!");
  }

  const char *RestoreLibCallName = getRestoreLibCallName(*MF);
  if (RestoreLibCallName) {
    // Replace terminating tail calls with a simple call. This is valid because
    // the return address register is always callee saved as part of the
    // save/restore libcalls.
    if (MI != MBB.end() && MI->getOpcode() == RISCV::PseudoTAIL) {
      MachineBasicBlock::iterator NewMI = BuildMI(MBB, MI, DL, TII.get(RISCV::PseudoCALL))
          .add(MI->getOperand(0));
      NewMI->copyImplicitOps(*MF, *MI);
      MI->eraseFromParent();
      MI = ++NewMI;
    }

    MachineBasicBlock::iterator NewMI = BuildMI(MBB, MI, DL, TII.get(RISCV::PseudoTAIL))
        .addExternalSymbol(RestoreLibCallName, RISCVII::MO_CALL)
        .setMIFlag(MachineInstr::FrameDestroy);

    // Remove trailing returns, since the terminator is now a tail call to the
    // restore function.
    if (MI != MBB.end() && MI->getOpcode() == RISCV::PseudoRET) {
      NewMI->copyImplicitOps(*MF, *MI);
      MI->eraseFromParent();
    }
  }

  return true;
}
