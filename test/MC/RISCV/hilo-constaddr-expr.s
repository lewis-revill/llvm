# RUN: not llvm-mc -filetype=obj -triple=riscv32 %s 2>&1 \
# RUN:     | FileCheck %s -check-prefix=CHECK-RELAX

# Check the assembler rejects hi and lo expressions with constant expressions
# involving labels when diff expressions are emitted as relocation pairs.
# Test case derived from test/MC/Mips/hilo-addressing.s

tmp1:
  # Emit zeros so that difference between tmp1 and tmp3 is 0x30124 bytes.
  .fill 0x30124-8
tmp2:
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lui t0, %hi(tmp3-tmp1)
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lw ra, %lo(tmp3-tmp1)(t0)

tmp3:
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lui t1, %hi(tmp2-tmp3)
# CHECK-RELAX: :[[@LINE+1]]:{{[0-9]+}}: error: expected relocatable expression
  lw sp, %lo(tmp2-tmp3)(t1)
