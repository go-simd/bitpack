//go:build ignore

// Part of the pack_ppc64le.s generator (run with pack_ppc64le_gen.go). Emits the
// unpack kernels unpackBits{bits}_VSX, the inverse of the pack kernels. See
// pack_ppc64le_gen.go for the layout, VSX-VMX aliasing and LXVW4X word-order
// notes.
//
// Unpack schedule for width `bits`: integer k occupies per-lane bits
// [k*bits, k*bits+bits). With word w=(k*bits)/32 and off=(k*bits)%32, the value
// is (W[w]>>off) | (W[w+1]<<(32-off) when it straddles), masked to `bits`. The
// schedule is fixed by `bits`, so each kernel is a branch-free unrolled sequence
// of LXVW4X/VSRW/VSLW/VOR/VAND/STXVW4X.
package main

import (
	"fmt"

	"github.com/go-asmgen/asmgen/ppc64"
)

// genUnpackPPC emits unpackBits{bits}_VSX. R3=dst, R4=src, R5=blocks.
func genUnpackPPC(s *ppcState, bits int, maskSym string) {
	name := fmt.Sprintf("unpackBits%d_VSX", bits)
	loop, done := "loop_"+name, "done_"+name
	fn := ppc64.NewFunc(name, ppcSig(), 0)
	fn.LoadArg("dst_base", "R3").LoadArg("src_base", "R4").LoadArg("blocks", "R5").
		Raw("MOVD $%s(SB), R6", maskSym).
		Raw("LXVW4X (R6), %s", vsOf("V15")). // V15 = mask
		Raw("CMP R5, $0").Raw("BEQ %s", done).
		Label(loop)

	s.emitUnpackBodyPPC(fn, bits)

	fn.Raw("ADD $%d, R4", 16*bits). // bits words * 16 bytes
		Raw("ADD $512, R3").        // 128 ints * 4 bytes
		Raw("ADD $-1, R5").Raw("CMP R5, $0").Raw("BNE %s", loop).
		Label(done).Ret()
	s.f.Add(fn.Func())
}

// emitUnpackBodyPPC mirrors the amd64 SSE emitUnpackBody.
//
//	Cur = V3   current source word (already loaded)
//	Nxt = V4   next source word (for straddling integers)
//	V   = V1   extracted value being assembled
//	scratch V13 for the straddle merge, V14 for count.
func (s *ppcState) emitUnpackBodyPPC(fn *ppc64.Builder, bits int) {
	const cur, nxt, v = "V3", "V4", "V1"
	loaded := -1

	ensure := func(w int) {
		if loaded != w {
			ppcLoadWord(fn, cur, w)
			loaded = w
		}
	}

	for k := 0; k < 32; k++ {
		start := k * bits
		word := start / 32
		off := start % 32
		end := off + bits

		ensure(word)
		// V = Cur >> off
		if off == 0 {
			fn.Raw("VOR %s, %s, %s", cur, cur, v) // V = Cur
		} else {
			s.countShiftPPC(fn, "VSRW", cur, off, v)
		}
		if end > 32 {
			ppcLoadWord(fn, nxt, word+1)
			// V |= Nxt << (32-off)  (scratch V13)
			s.countShiftPPC(fn, "VSLW", nxt, 32-off, "V13")
			fn.Raw("VOR V13, %s, %s", v, v)
			// pre-cache next word as current
			fn.Raw("VOR %s, %s, %s", nxt, nxt, cur)
			loaded = word + 1
		}
		fn.Raw("VAND V15, %s, %s", v, v) // V &= mask
		ppcStoreVec(fn, v, k)
	}
}

// countShiftPPC loads the count constant for n into V14 and emits
// `op V14, src, dst` (dst = src shifted by n, per word).
func (s *ppcState) countShiftPPC(fn *ppc64.Builder, op, src string, n int, dst string) {
	sym := s.count(n)
	fn.Raw("MOVD $%s(SB), R7", sym).
		Raw("LXVW4X (R7), %s", vsOf("V14")).
		Raw("%s %s, V14, %s", op, src, dst) // Plan9: data, count, dst
}

// ppcLoadWord loads source word w (byte offset 16w) into reg.
func ppcLoadWord(fn *ppc64.Builder, reg string, w int) {
	if w == 0 {
		fn.Raw("LXVW4X (R4), %s", vsOf(reg))
		return
	}
	fn.Raw("MOVD $%d, R8", 16*w).
		Raw("LXVW4X (R4)(R8), %s", vsOf(reg))
}

// ppcStoreVec stores reg to output vector k (dst[4k..4k+3], byte offset 16k).
func ppcStoreVec(fn *ppc64.Builder, reg string, k int) {
	if k == 0 {
		fn.Raw("STXVW4X %s, (R3)", vsOf(reg))
		return
	}
	fn.Raw("MOVD $%d, R8", 16*k).
		Raw("STXVW4X %s, (R3)(R8)", vsOf(reg))
}
