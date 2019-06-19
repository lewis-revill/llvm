; RUN: llc -mtriple=riscv32 < %s | FileCheck %s -check-prefix=RV32I
; RUN: llc -mtriple=riscv64 < %s | FileCheck %s -check-prefix=RV64I
; RUN: llc -mtriple=riscv32 -enable-save-restore < %s | FileCheck %s -check-prefix=RV32I-SR
; RUN: llc -mtriple=riscv64 -enable-save-restore < %s | FileCheck %s -check-prefix=RV64I-SR


; Check that the correct save/restore libcalls are generated.

@var0 = global [18 x i32] zeroinitializer
@var1 = global [24 x i32] zeroinitializer
@var2 = global [30 x i32] zeroinitializer

define void @callee_saved0() nounwind {
; RV32I-LABEL: callee_saved0:
; RV32I:         addi sp, sp, -32
; RV32I-NEXT:    sw s0, 28(sp)
; RV32I-NEXT:    sw s1, 24(sp)
; RV32I-NEXT:    sw s2, 20(sp)
; RV32I-NEXT:    sw s3, 16(sp)
; RV32I-NEXT:    sw s4, 12(sp)
; RV32I:         lw s4, 12(sp)
; RV32I-NEXT:    lw s3, 16(sp)
; RV32I-NEXT:    lw s2, 20(sp)
; RV32I-NEXT:    lw s1, 24(sp)
; RV32I-NEXT:    lw s0, 28(sp)
; RV32I-NEXT:    addi sp, sp, 32
; RV32I-NEXT:    ret
;
; RV64I-LABEL: callee_saved0:
; RV64I:         addi sp, sp, -48
; RV64I-NEXT:    sd s0, 40(sp)
; RV64I-NEXT:    sd s1, 32(sp)
; RV64I-NEXT:    sd s2, 24(sp)
; RV64I-NEXT:    sd s3, 16(sp)
; RV64I:         ld s4, 8(sp)
; RV64I-NEXT:    ld s3, 16(sp)
; RV64I-NEXT:    ld s2, 24(sp)
; RV64I-NEXT:    ld s1, 32(sp)
; RV64I-NEXT:    ld s0, 40(sp)
; RV64I-NEXT:    addi sp, sp, 48
; RV64I-NEXT:    ret
;
; RV32I-SR-LABEL: callee_saved0:
; RV32I-SR:         call t0, __riscv_save_5
; RV32I-SR:         tail __riscv_restore_5
;
; RV64I-SR-LABEL: callee_saved0:
; RV64I-SR:         call t0, __riscv_save_5
; RV64I-SR:         tail __riscv_restore_5
  %val = load [18 x i32], [18 x i32]* @var0
  store volatile [18 x i32] %val, [18 x i32]* @var0
  ret void
}

define void @callee_saved1() nounwind {
; RV32I-LABEL: callee_saved1:
; RV32I:         addi sp, sp, -48
; RV32I-NEXT:    sw s0, 44(sp)
; RV32I-NEXT:    sw s1, 40(sp)
; RV32I-NEXT:    sw s2, 36(sp)
; RV32I-NEXT:    sw s3, 32(sp)
; RV32I-NEXT:    sw s4, 28(sp)
; RV32I-NEXT:    sw s5, 24(sp)
; RV32I-NEXT:    sw s6, 20(sp)
; RV32I-NEXT:    sw s7, 16(sp)
; RV32I-NEXT:    sw s8, 12(sp)
; RV32I-NEXT:    sw s9, 8(sp)
; RV32I-NEXT:    sw s10, 4(sp)
; RV32I:         lw s10, 4(sp)
; RV32I-NEXT:    lw s9, 8(sp)
; RV32I-NEXT:    lw s8, 12(sp)
; RV32I-NEXT:    lw s7, 16(sp)
; RV32I-NEXT:    lw s6, 20(sp)
; RV32I-NEXT:    lw s5, 24(sp)
; RV32I-NEXT:    lw s4, 28(sp)
; RV32I-NEXT:    lw s3, 32(sp)
; RV32I-NEXT:    lw s2, 36(sp)
; RV32I-NEXT:    lw s1, 40(sp)
; RV32I-NEXT:    lw s0, 44(sp)
; RV32I-NEXT:    addi sp, sp, 48
; RV32I-NEXT:    ret
;
; RV64I-LABEL: callee_saved1:
; RV64I:         addi sp, sp, -96
; RV64I-NEXT:    sd s0, 88(sp)
; RV64I-NEXT:    sd s1, 80(sp)
; RV64I-NEXT:    sd s2, 72(sp)
; RV64I-NEXT:    sd s3, 64(sp)
; RV64I-NEXT:    sd s4, 56(sp)
; RV64I-NEXT:    sd s5, 48(sp)
; RV64I-NEXT:    sd s6, 40(sp)
; RV64I-NEXT:    sd s7, 32(sp)
; RV64I-NEXT:    sd s8, 24(sp)
; RV64I-NEXT:    sd s9, 16(sp)
; RV64I-NEXT:    sd s10, 8(sp)
; RV64I:         ld s10, 8(sp)
; RV64I-NEXT:    ld s9, 16(sp)
; RV64I-NEXT:    ld s8, 24(sp)
; RV64I-NEXT:    ld s7, 32(sp)
; RV64I-NEXT:    ld s6, 40(sp)
; RV64I-NEXT:    ld s5, 48(sp)
; RV64I-NEXT:    ld s4, 56(sp)
; RV64I-NEXT:    ld s3, 64(sp)
; RV64I-NEXT:    ld s2, 72(sp)
; RV64I-NEXT:    ld s1, 80(sp)
; RV64I-NEXT:    ld s0, 88(sp)
; RV64I-NEXT:    addi sp, sp, 96
; RV64I-NEXT:    ret
;
; RV32I-SR-LABEL: callee_saved1:
; RV32I-SR:         call t0, __riscv_save_11
; RV32I-SR:         tail __riscv_restore_11
;
; RV64I-SR-LABEL: callee_saved1:
; RV64I-SR:         call t0, __riscv_save_11
; RV64I-SR:         tail __riscv_restore_11
  %val = load [24 x i32], [24 x i32]* @var1
  store volatile [24 x i32] %val, [24 x i32]* @var1
  ret void
}

define void @callee_saved2() nounwind {
; RV32I-LABEL: callee_saved2:
; RV32I:         addi sp, sp, -64
; RV32I-NEXT:    sw s0, 60(sp)
; RV32I-NEXT:    sw s1, 56(sp)
; RV32I-NEXT:    sw s2, 52(sp)
; RV32I-NEXT:    sw s3, 48(sp)
; RV32I-NEXT:    sw s4, 44(sp)
; RV32I-NEXT:    sw s5, 40(sp)
; RV32I-NEXT:    sw s6, 36(sp)
; RV32I-NEXT:    sw s7, 32(sp)
; RV32I-NEXT:    sw s8, 28(sp)
; RV32I-NEXT:    sw s9, 24(sp)
; RV32I-NEXT:    sw s10, 20(sp)
; RV32I-NEXT:    sw s11, 16(sp)
; RV32I:         lw s11, 16(sp)
; RV32I-NEXT:    lw s10, 20(sp)
; RV32I-NEXT:    lw s9, 24(sp)
; RV32I-NEXT:    lw s8, 28(sp)
; RV32I-NEXT:    lw s7, 32(sp)
; RV32I-NEXT:    lw s6, 36(sp)
; RV32I-NEXT:    lw s5, 40(sp)
; RV32I-NEXT:    lw s4, 44(sp)
; RV32I-NEXT:    lw s3, 48(sp)
; RV32I-NEXT:    lw s2, 52(sp)
; RV32I-NEXT:    lw s1, 56(sp)
; RV32I-NEXT:    lw s0, 60(sp)
; RV32I-NEXT:    addi sp, sp, 64
; RV32I-NEXT:    ret
;
; RV64I-LABEL: callee_saved2:
; RV64I:         addi sp, sp, -128
; RV64I-NEXT:    sd s0, 120(sp)
; RV64I-NEXT:    sd s1, 112(sp)
; RV64I-NEXT:    sd s2, 104(sp)
; RV64I-NEXT:    sd s3, 96(sp)
; RV64I-NEXT:    sd s4, 88(sp)
; RV64I-NEXT:    sd s5, 80(sp)
; RV64I-NEXT:    sd s6, 72(sp)
; RV64I-NEXT:    sd s7, 64(sp)
; RV64I-NEXT:    sd s8, 56(sp)
; RV64I-NEXT:    sd s9, 48(sp)
; RV64I-NEXT:    sd s10, 40(sp)
; RV64I-NEXT:    sd s11, 32(sp)
; RV64I:         ld s11, 32(sp)
; RV64I-NEXT:    ld s10, 40(sp)
; RV64I-NEXT:    ld s9, 48(sp)
; RV64I-NEXT:    ld s8, 56(sp)
; RV64I-NEXT:    ld s7, 64(sp)
; RV64I-NEXT:    ld s6, 72(sp)
; RV64I-NEXT:    ld s5, 80(sp)
; RV64I-NEXT:    ld s4, 88(sp)
; RV64I-NEXT:    ld s3, 96(sp)
; RV64I-NEXT:    ld s2, 104(sp)
; RV64I-NEXT:    ld s1, 112(sp)
; RV64I-NEXT:    ld s0, 120(sp)
; RV64I-NEXT:    addi sp, sp, 128
; RV64I-NEXT:    ret
;
; RV32I-SR-LABEL: callee_saved2:
; RV32I-SR:         call t0, __riscv_save_12
; RV32I-SR:         tail __riscv_restore_12
;
; RV64I-SR-LABEL: callee_saved2:
; RV64I-SR:         call t0, __riscv_save_12
; RV64I-SR:         tail __riscv_restore_12
  %val = load [30 x i32], [30 x i32]* @var2
  store volatile [30 x i32] %val, [30 x i32]* @var2
  ret void
}

; Check that tail calls are updated correctly by save/restore

declare i32 @tail_callee(i32 %i)

define i32 @tail_call(i32 %i) nounwind {
; RV32I-LABEL: tail_call:
; RV32I:         addi sp, sp, -32
; RV32I-NEXT:    sw s0, 28(sp)
; RV32I-NEXT:    sw s1, 24(sp)
; RV32I-NEXT:    sw s2, 20(sp)
; RV32I-NEXT:    sw s3, 16(sp)
; RV32I-NEXT:    sw s4, 12(sp)
; RV32I-NEXT:    sw s5, 8(sp)
; RV32I:         lw s5, 8(sp)
; RV32I-NEXT:    lw s4, 12(sp)
; RV32I-NEXT:    lw s3, 16(sp)
; RV32I-NEXT:    lw s2, 20(sp)
; RV32I-NEXT:    lw s1, 24(sp)
; RV32I-NEXT:    lw s0, 28(sp)
; RV32I-NEXT:    addi sp, sp, 32
; RV32I-NEXT:    tail tail_callee
;
; RV64I-LABEL: tail_call:
; RV64I:         addi sp, sp, -48
; RV64I-NEXT:    sd s0, 40(sp)
; RV64I-NEXT:    sd s1, 32(sp)
; RV64I-NEXT:    sd s2, 24(sp)
; RV64I-NEXT:    sd s3, 16(sp)
; RV64I-NEXT:    sd s4, 8(sp)
; RV64I-NEXT:    sd s5, 0(sp)
; RV64I:         ld s5, 0(sp)
; RV64I-NEXT:    ld s4, 8(sp)
; RV64I-NEXT:    ld s3, 16(sp)
; RV64I-NEXT:    ld s2, 24(sp)
; RV64I-NEXT:    ld s1, 32(sp)
; RV64I-NEXT:    ld s0, 40(sp)
; RV64I-NEXT:    addi sp, sp, 48
; RV64I-NEXT:    tail tail_callee
;
; RV32I-SR-LABEL: tail_call:
; RV32I-SR:         call t0, __riscv_save_6
; RV32I-SR:         call tail_callee
; RV32I-SR-NEXT:    tail __riscv_restore_6
;
; RV64I-SR-LABEL: tail_call:
; RV64I-SR:         call t0, __riscv_save_6
; RV64I-SR:         call tail_callee
; RV64I-SR-NEXT:    tail __riscv_restore_6
entry:
  %val = load [18 x i32], [18 x i32]* @var0
  store volatile [18 x i32] %val, [18 x i32]* @var0
  %r = tail call i32 @tail_callee(i32 %i)
  ret i32 %r
}

; Check that functions with varargs do not use save/restore code

declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)

define i32 @varargs(i8* %fmt, ...) nounwind {
; RV32I-LABEL: varargs:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -48
; RV32I-NEXT:    mv a0, a1
; RV32I-NEXT:    sw a7, 44(sp)
; RV32I-NEXT:    sw a6, 40(sp)
; RV32I-NEXT:    sw a5, 36(sp)
; RV32I-NEXT:    sw a4, 32(sp)
; RV32I-NEXT:    sw a3, 28(sp)
; RV32I-NEXT:    sw a2, 24(sp)
; RV32I-NEXT:    addi a1, sp, 24
; RV32I-NEXT:    sw a1, 12(sp)
; RV32I-NEXT:    sw a0, 20(sp)
; RV32I-NEXT:    addi sp, sp, 48
; RV32I-NEXT:    ret
;
; RV64I-LABEL: varargs:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -80
; RV64I-NEXT:    sd a1, 24(sp)
; RV64I-NEXT:    sd a7, 72(sp)
; RV64I-NEXT:    sd a6, 64(sp)
; RV64I-NEXT:    sd a5, 56(sp)
; RV64I-NEXT:    sd a4, 48(sp)
; RV64I-NEXT:    sd a3, 40(sp)
; RV64I-NEXT:    sd a2, 32(sp)
; RV64I-NEXT:    addi a0, sp, 24
; RV64I-NEXT:    ori a0, a0, 4
; RV64I-NEXT:    sd a0, 8(sp)
; RV64I-NEXT:    lw a0, 24(sp)
; RV64I-NEXT:    addi sp, sp, 80
; RV64I-NEXT:    ret
;
; RV32I-SR-LABEL: varargs:
; RV32I-SR:       # %bb.0:
; RV32I-SR-NEXT:    addi sp, sp, -48
; RV32I-SR-NEXT:    mv a0, a1
; RV32I-SR-NEXT:    sw a7, 44(sp)
; RV32I-SR-NEXT:    sw a6, 40(sp)
; RV32I-SR-NEXT:    sw a5, 36(sp)
; RV32I-SR-NEXT:    sw a4, 32(sp)
; RV32I-SR-NEXT:    sw a3, 28(sp)
; RV32I-SR-NEXT:    sw a2, 24(sp)
; RV32I-SR-NEXT:    addi a1, sp, 24
; RV32I-SR-NEXT:    sw a1, 12(sp)
; RV32I-SR-NEXT:    sw a0, 20(sp)
; RV32I-SR-NEXT:    addi sp, sp, 48
; RV32I-SR-NEXT:    ret
;
; RV64I-SR-LABEL: varargs:
; RV64I-SR:       # %bb.0:
; RV64I-SR-NEXT:    addi sp, sp, -80
; RV64I-SR-NEXT:    sd a1, 24(sp)
; RV64I-SR-NEXT:    sd a7, 72(sp)
; RV64I-SR-NEXT:    sd a6, 64(sp)
; RV64I-SR-NEXT:    sd a5, 56(sp)
; RV64I-SR-NEXT:    sd a4, 48(sp)
; RV64I-SR-NEXT:    sd a3, 40(sp)
; RV64I-SR-NEXT:    sd a2, 32(sp)
; RV64I-SR-NEXT:    addi a0, sp, 24
; RV64I-SR-NEXT:    ori a0, a0, 4
; RV64I-SR-NEXT:    sd a0, 8(sp)
; RV64I-SR-NEXT:    lw a0, 24(sp)
; RV64I-SR-NEXT:    addi sp, sp, 80
; RV64I-SR-NEXT:    ret
  %va = alloca i8*, align 4
  %1 = bitcast i8** %va to i8*
  call void @llvm.va_start(i8* %1)
  %argp.cur = load i8*, i8** %va, align 4
  %argp.next = getelementptr inbounds i8, i8* %argp.cur, i32 4
  store i8* %argp.next, i8** %va, align 4
  %2 = bitcast i8* %argp.cur to i32*
  %3 = load i32, i32* %2, align 4
  call void @llvm.va_end(i8* %1)
  ret i32 %3
}

define void @many_args(i32, i32, i32, i32, i32, i32, i32, i32, i32) nounwind {
; RV32I-LABEL: many_args:
; RV32I:         addi sp, sp, -32
; RV32I-NEXT:    sw s0, 28(sp)
; RV32I-NEXT:    sw s1, 24(sp)
; RV32I-NEXT:    sw s2, 20(sp)
; RV32I-NEXT:    sw s3, 16(sp)
; RV32I-NEXT:    sw s4, 12(sp)
; RV32I:         lw s4, 12(sp)
; RV32I-NEXT:    lw s3, 16(sp)
; RV32I-NEXT:    lw s2, 20(sp)
; RV32I-NEXT:    lw s1, 24(sp)
; RV32I-NEXT:    lw s0, 28(sp)
; RV32I-NEXT:    addi sp, sp, 32
; RV32I-NEXT:    ret
;
; RV64I-LABEL: many_args:
; RV64I:         addi sp, sp, -48
; RV64I-NEXT:    sd s0, 40(sp)
; RV64I-NEXT:    sd s1, 32(sp)
; RV64I-NEXT:    sd s2, 24(sp)
; RV64I-NEXT:    sd s3, 16(sp)
; RV64I-NEXT:    sd s4, 8(sp)
; RV64I:         ld s4, 8(sp)
; RV64I-NEXT:    ld s3, 16(sp)
; RV64I-NEXT:    ld s2, 24(sp)
; RV64I-NEXT:    ld s1, 32(sp)
; RV64I-NEXT:    ld s0, 40(sp)
; RV64I-NEXT:    addi sp, sp, 48
; RV64I-NEXT:    ret
;
; RV32I-SR-LABEL: many_args:
; RV32I-SR:         call t0, __riscv_save_5
; RV32I-SR:         tail __riscv_restore_5
;
; RV64I-SR-LABEL: many_args:
; RV64I-SR:         call t0, __riscv_save_5
; RV64I-SR:         tail __riscv_restore_5
entry:
  %val = load [18 x i32], [18 x i32]* @var0
  store volatile [18 x i32] %val, [18 x i32]* @var0
  ret void
}

; Check that dynamic allocation calculations remain correct

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)
declare void @notdead(i8*)

define void @alloca(i32 %n) nounwind {
; RV32I-LABEL: alloca:
; RV32I:       # %bb.0:
; RV32I-NEXT:    addi sp, sp, -16
; RV32I-NEXT:    sw ra, 12(sp)
; RV32I-NEXT:    sw s0, 8(sp)
; RV32I-NEXT:    sw s1, 4(sp)
; RV32I-NEXT:    addi s0, sp, 16
; RV32I-NEXT:    mv s1, sp
; RV32I-NEXT:    addi a0, a0, 15
; RV32I-NEXT:    andi a0, a0, -16
; RV32I-NEXT:    sub a0, sp, a0
; RV32I-NEXT:    mv sp, a0
; RV32I-NEXT:    call notdead
; RV32I-NEXT:    mv sp, s1
; RV32I-NEXT:    addi sp, s0, -16
; RV32I-NEXT:    lw s1, 4(sp)
; RV32I-NEXT:    lw s0, 8(sp)
; RV32I-NEXT:    lw ra, 12(sp)
; RV32I-NEXT:    addi sp, sp, 16
; RV32I-NEXT:    ret
;
; RV64I-LABEL: alloca:
; RV64I:       # %bb.0:
; RV64I-NEXT:    addi sp, sp, -32
; RV64I-NEXT:    sd ra, 24(sp)
; RV64I-NEXT:    sd s0, 16(sp)
; RV64I-NEXT:    sd s1, 8(sp)
; RV64I-NEXT:    addi s0, sp, 32
; RV64I-NEXT:    addi a1, zero, 1
; RV64I-NEXT:    slli a1, a1, 33
; RV64I-NEXT:    addi a1, a1, -16
; RV64I-NEXT:    slli a0, a0, 32
; RV64I-NEXT:    srli a0, a0, 32
; RV64I-NEXT:    addi a0, a0, 15
; RV64I-NEXT:    and a0, a0, a1
; RV64I-NEXT:    mv s1, sp
; RV64I-NEXT:    sub a0, sp, a0
; RV64I-NEXT:    mv sp, a0
; RV64I-NEXT:    call notdead
; RV64I-NEXT:    mv sp, s1
; RV64I-NEXT:    addi sp, s0, -32
; RV64I-NEXT:    ld s1, 8(sp)
; RV64I-NEXT:    ld s0, 16(sp)
; RV64I-NEXT:    ld ra, 24(sp)
; RV64I-NEXT:    addi sp, sp, 32
; RV64I-NEXT:    ret
;
; RV32I-SR-LABEL: alloca:
; RV32I-SR:       # %bb.0:
; RV32I-SR-NEXT:    call t0, __riscv_save_2
; RV32I-SR-NEXT:    mv s0, sp
; RV32I-SR-NEXT:    mv s1, sp
; RV32I-SR-NEXT:    addi a0, a0, 15
; RV32I-SR-NEXT:    andi a0, a0, -16
; RV32I-SR-NEXT:    sub a0, sp, a0
; RV32I-SR-NEXT:    mv sp, a0
; RV32I-SR-NEXT:    call notdead
; RV32I-SR-NEXT:    mv sp, s1
; RV32I-SR-NEXT:    mv sp, s0
; RV32I-SR-NEXT:    tail __riscv_restore_2
;
; RV64I-SR-LABEL: alloca:
; RV64I-SR:       # %bb.0:
; RV64I-SR-NEXT:    call t0, __riscv_save_2
; RV64I-SR-NEXT:    mv s0, sp
; RV64I-SR-NEXT:    addi a1, zero, 1
; RV64I-SR-NEXT:    slli a1, a1, 33
; RV64I-SR-NEXT:    addi a1, a1, -16
; RV64I-SR-NEXT:    slli a0, a0, 32
; RV64I-SR-NEXT:    srli a0, a0, 32
; RV64I-SR-NEXT:    addi a0, a0, 15
; RV64I-SR-NEXT:    and a0, a0, a1
; RV64I-SR-NEXT:    mv s1, sp
; RV64I-SR-NEXT:    sub a0, sp, a0
; RV64I-SR-NEXT:    mv sp, a0
; RV64I-SR-NEXT:    call notdead
; RV64I-SR-NEXT:    mv sp, s1
; RV64I-SR-NEXT:    mv sp, s0
; RV64I-SR-NEXT:    tail __riscv_restore_2
  %sp = call i8* @llvm.stacksave()
  %addr = alloca i8, i32 %n
  call void @notdead(i8* %addr)
  call void @llvm.stackrestore(i8* %sp)
  ret void
}
