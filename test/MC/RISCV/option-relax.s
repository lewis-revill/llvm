# RUN: llvm-mc -triple riscv32 < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv64 < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=CHECK-RELOC %s

# CHECK-INST: call foo
# CHECK-RELOC: R_RISCV_CALL foo 0x0
call foo

.option relax
# CHECK-INST: .option relax
# CHECK-INST: call bar
# CHECK-RELOC-NEXT: R_RISCV_CALL bar 0x0
# CHECK-RELOC-NEXT: R_RISCV_RELAX bar 0x0
call bar

.option norelax
# CHECK-INST: .option norelax
# CHECK-INST: call baz
# CHECK-RELOC: R_RISCV_CALL baz 0x0
# CHECK-RELOC-NOT: R_RISCV_RELAX baz 0x0
call baz
