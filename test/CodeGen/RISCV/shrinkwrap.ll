; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple riscv32 < %s | FileCheck %s -check-prefix=RV32I-NOSW
; RUN: llc -mtriple riscv32 -enable-shrink-wrap < %s | FileCheck %s -check-prefix=RV32I-SW


declare void @abort()

define void @eliminate_restore(i32 %n) {
; RV32I-NOSW-LABEL: eliminate_restore:
; RV32I-NOSW:       # %bb.0:
; RV32I-NOSW-NEXT:    addi sp, sp, -16
; RV32I-NOSW-NEXT:    sw ra, 12(sp)
; RV32I-NOSW-NEXT:    addi a1, zero, 32
; RV32I-NOSW-NEXT:    bgeu a1, a0, .LBB0_2
; RV32I-NOSW-NEXT:  # %bb.1: # %if.end
; RV32I-NOSW-NEXT:    lw ra, 12(sp)
; RV32I-NOSW-NEXT:    addi sp, sp, 16
; RV32I-NOSW-NEXT:    ret
; RV32I-NOSW-NEXT:  .LBB0_2: # %if.then
; RV32I-NOSW-NEXT:    call abort
;
; RV32I-SW-LABEL: eliminate_restore:
; RV32I-SW:       # %bb.0:
; RV32I-SW-NEXT:    addi a1, zero, 32
; RV32I-SW-NEXT:    bgeu a1, a0, .LBB0_2
; RV32I-SW-NEXT:  # %bb.1: # %if.end
; RV32I-SW-NEXT:    ret
; RV32I-SW-NEXT:  .LBB0_2: # %if.then
; RV32I-SW-NEXT:    addi sp, sp, -16
; RV32I-SW-NEXT:    sw ra, 12(sp)
; RV32I-SW-NEXT:    call abort
  %cmp = icmp ule i32 %n, 32
  br i1 %cmp, label %if.then, label %if.end

if.then:
  call void @abort()
  unreachable

if.end:
  ret void
}

declare void @notdead(i8*)

define void @conditional_alloca(i32 %n) nounwind {
; RV32I-NOSW-LABEL: conditional_alloca:
; RV32I-NOSW:       # %bb.0:
; RV32I-NOSW-NEXT:    addi sp, sp, -16
; RV32I-NOSW-NEXT:    sw ra, 12(sp)
; RV32I-NOSW-NEXT:    sw s0, 8(sp)
; RV32I-NOSW-NEXT:    addi s0, sp, 16
; RV32I-NOSW-NEXT:    addi a1, zero, 32
; RV32I-NOSW-NEXT:    bltu a1, a0, .LBB1_2
; RV32I-NOSW-NEXT:  # %bb.1: # %if.then
; RV32I-NOSW-NEXT:    addi a0, a0, 15
; RV32I-NOSW-NEXT:    andi a0, a0, -16
; RV32I-NOSW-NEXT:    sub a0, sp, a0
; RV32I-NOSW-NEXT:    mv sp, a0
; RV32I-NOSW-NEXT:    call notdead
; RV32I-NOSW-NEXT:  .LBB1_2: # %if.end
; RV32I-NOSW-NEXT:    addi sp, s0, -16
; RV32I-NOSW-NEXT:    lw s0, 8(sp)
; RV32I-NOSW-NEXT:    lw ra, 12(sp)
; RV32I-NOSW-NEXT:    addi sp, sp, 16
; RV32I-NOSW-NEXT:    ret
;
; RV32I-SW-LABEL: conditional_alloca:
; RV32I-SW:       # %bb.0:
; RV32I-SW-NEXT:    addi a1, zero, 32
; RV32I-SW-NEXT:    bltu a1, a0, .LBB1_2
; RV32I-SW-NEXT:  # %bb.1: # %if.then
; RV32I-SW-NEXT:    addi sp, sp, -16
; RV32I-SW-NEXT:    sw ra, 12(sp)
; RV32I-SW-NEXT:    sw s0, 8(sp)
; RV32I-SW-NEXT:    addi s0, sp, 16
; RV32I-SW-NEXT:    addi a0, a0, 15
; RV32I-SW-NEXT:    andi a0, a0, -16
; RV32I-SW-NEXT:    sub a0, sp, a0
; RV32I-SW-NEXT:    mv sp, a0
; RV32I-SW-NEXT:    call notdead
; RV32I-SW-NEXT:    addi sp, s0, -16
; RV32I-SW-NEXT:    lw s0, 8(sp)
; RV32I-SW-NEXT:    lw ra, 12(sp)
; RV32I-SW-NEXT:    addi sp, sp, 16
; RV32I-SW-NEXT:  .LBB1_2: # %if.end
; RV32I-SW-NEXT:    ret
  %cmp = icmp ule i32 %n, 32
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %addr = alloca i8, i32 %n
  call void @notdead(i8* %addr)
  br label %if.end

if.end:
  ret void
}
