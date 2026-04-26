//go:build amd64 && !purego

#include "textflag.h"

// func squaredL2(a, b []float32) float32
TEXT ·squaredL2(SB), NOSPLIT, $16-56
	MOVQ a_base+0(FP), AX
	MOVQ a_len+8(FP), CX
	MOVQ b_base+24(FP), BX

	XORPS X0, X0

	MOVQ CX, DX
	SHRQ $2, DX
	JZ reduce

vector_loop:
	MOVUPS (AX), X1
	MOVUPS (BX), X2
	SUBPS X2, X1
	MULPS X1, X1
	ADDPS X1, X0
	ADDQ $16, AX
	ADDQ $16, BX
	DECQ DX
	JNZ vector_loop

reduce:
	MOVUPS X0, 0(SP)
	MOVSS 0(SP), X3
	ADDSS 4(SP), X3
	ADDSS 8(SP), X3
	ADDSS 12(SP), X3

	ANDQ $3, CX
	JZ done

tail_loop:
	MOVSS (AX), X1
	SUBSS (BX), X1
	MULSS X1, X1
	ADDSS X1, X3
	ADDQ $4, AX
	ADDQ $4, BX
	DECQ CX
	JNZ tail_loop

done:
	MOVSS X3, ret+48(FP)
	RET
